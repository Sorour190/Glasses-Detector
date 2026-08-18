"""Calibrate the production operating point on the CAL split.

The cal split is disjoint from both train and the val split used for model
selection — thresholds fitted on the selection split are optimistically
biased.

Produces threshold.json:
    T        temperature (scales log-probs; >1 softens overconfident outputs)
    t_high   highest threshold with eyeglasses recall >= --target-recall
    t_low    lowest threshold with FPR <= --target-fpr
    [t_low, t_high) is the abstain / retry-capture band (empty if crossed)

Usage:
    python -m glasses_detector.calibrate --manifest data/manifest_r5.csv \
        --checkpoint runs/R5/best.pt --out runs/R5/threshold.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ManifestDataset
from .metrics import GLASSES_CLASS, binary_metrics, ece, evaluate
from .model import load_checkpoint


def fit_temperature(log_probs: np.ndarray, labels: np.ndarray) -> float:
    """Scalar temperature minimizing NLL on the cal split."""
    logits = torch.tensor(log_probs)
    y = torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    cal = ManifestDataset(args.manifest, split="cal", mode="eval")
    probs, labels, _ = evaluate(model, DataLoader(cal, batch_size=256,
                                                  num_workers=4), device)

    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    temp = fit_temperature(log_p, labels)
    probs_t = torch.softmax(torch.tensor(log_p) / temp, dim=1).numpy()

    y = (labels == GLASSES_CLASS).astype(int)
    p = probs_t[:, GLASSES_CLASS]
    print(f"T={temp:.3f}  ECE before={ece(probs[:, GLASSES_CLASS], y):.4f} "
          f"after={ece(p, y):.4f}")

    grid = np.linspace(0.01, 0.99, 197)
    recalls = np.array([(p[y == 1] >= t).mean() for t in grid])
    fprs = np.array([(p[y == 0] >= t).mean() for t in grid])
    ok_recall = grid[recalls >= args.target_recall]
    ok_fpr = grid[fprs <= args.target_fpr]
    if len(ok_recall) == 0 or len(ok_fpr) == 0:
        raise RuntimeError(
            f"targets unreachable on cal: max recall {recalls.max():.4f}, "
            f"min fpr {fprs.min():.4f} — do NOT ship a default threshold")
    t_high = float(ok_recall.max())
    t_low = float(ok_fpr.min())

    abstain = float(((p >= min(t_low, t_high)) & (p < max(t_low, t_high))).mean())
    m = binary_metrics(probs_t, labels, thresh=0.5)
    out = {
        "T": round(temp, 4), "t_low": round(t_low, 4), "t_high": round(t_high, 4),
        "band_crossed": t_low >= t_high, "abstain_rate_cal": round(abstain, 4),
        "cal_n": len(labels), "checkpoint": args.checkpoint,
        "cal_metrics_at_0.5": str(m),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
