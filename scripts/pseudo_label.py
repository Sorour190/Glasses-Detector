"""Pseudo-label 'unknown' eyewear rows with the current best model.

Policy (conservative):
    p_eyeglasses >= hi  -> eyeglasses
    p_sunglasses >= hi  -> sunglasses
    otherwise           -> 'review' (kept out of training until a human pass;
                           these are concentrated hard positives)

Usage:
    python scripts/pseudo_label.py --manifest data/manifest_anyglasses_pos.csv \
        --checkpoint runs/R4/best.pt --hi 0.85 --out data/manifest_anyglasses_labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glasses_detector.dataset import ManifestDataset  # noqa: E402
from glasses_detector.metrics import contact_sheet, evaluate  # noqa: E402
from glasses_detector.model import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets-dir", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    ds = ManifestDataset(df, mode="eval")
    probs, _, idx = evaluate(model, DataLoader(ds, batch_size=256, num_workers=6,
                                               pin_memory=True), device)
    probs = probs[np.argsort(idx)]

    p_none, p_eye, p_sun = probs[:, 0], probs[:, 1], probs[:, 2]
    unknown = df["label"].eq("unknown")
    df.loc[unknown, "label"] = "review"
    df.loc[unknown & (p_eye >= args.hi), "label"] = "eyeglasses"
    df.loc[unknown & (p_sun >= args.hi), "label"] = "sunglasses"
    df["label_id"] = df["label"].map({"none": 0, "eyeglasses": 1,
                                      "sunglasses": 2, "review": -1})
    df["p_eyeglasses"] = p_eye.round(4)
    df.to_csv(args.out, index=False)
    print(df["label"].value_counts().to_string())

    if args.sheets_dir:
        out = Path(args.sheets_dir)
        review = np.where(df["label"].to_numpy() == "review")[0]
        order = review[np.argsort(-p_eye[review])]
        for i in range(0, min(len(order), 360), 60):
            contact_sheet(df, order[i:i + 60], p_eye[order[i:i + 60]],
                          out / f"review_{i // 60}.png", "review", max_tiles=60)
        print(f"{len(review)} review images -> sheets in {out}")


if __name__ == "__main__":
    main()
