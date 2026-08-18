"""Train the glasses classifier (3-class head, binary product output).

Staged transfer learning:
  1. head warmup (backbone frozen)         lr 1e-3, 3 epochs
  2. last 4 backbone blocks + head         lr 3e-4 cosine, 8 epochs
  3. full unfreeze (param groups)          lr 1e-4/5e-4 cosine, optional

Selection + early stopping use worst_condition_accuracy on the degraded val
suite — never clean accuracy or loss, which are dominated by easy clean
examples and stop training at a non-robust optimum.

Usage:
    python -m glasses_detector.train --manifest data/manifest_tier0.csv \
        --run-name R1 --aug-severity 0.4
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ManifestDataset
from .metrics import (binary_metrics, make_error_sheets, per_condition_eval,
                      worst_condition_accuracy)
from .model import (backbone_head_param_groups, build_model, unfreeze_all,
                    unfreeze_backbone)

CLASS_WEIGHTS = [1.0, 1.0, 2.0]  # none, eyeglasses, sunglasses (hard negatives)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "none"
    except OSError:
        return "none"


def run_stage(model, stage_name, train_loader, manifest, device, epochs, optimizer,
              patience=5, min_delta=0.002, eval_workers=2):
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, device=device), label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = {"worst_acc": -1.0, "state": None, "epoch": -1, "clean": None, "cond": None}
    stale = 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
        scheduler.step()

        from .metrics import _cached_crops, _eval_crops
        crops, labels = _cached_crops(manifest, "val")
        probs = _eval_crops(model, crops, device)
        clean = binary_metrics(probs, labels)
        cond = per_condition_eval(model, manifest, "val", device,
                                  num_workers=eval_workers)
        worst = worst_condition_accuracy(cond)
        print(f"[{stage_name}] epoch {epoch + 1}/{epochs} "
              f"loss={running / len(train_loader.dataset):.4f} "
              f"clean: {clean} worst_cond={worst:.4f}", flush=True)

        if worst > best["worst_acc"] + min_delta:
            best.update(worst_acc=worst, epoch=epoch,
                        state=copy.deepcopy(model.state_dict()),
                        clean=clean, cond=cond)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{stage_name}] early stop at epoch {epoch + 1}", flush=True)
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=8)
    ap.add_argument("--full-epochs", type=int, default=0,
                    help="stage-3 full-unfreeze epochs (0 = skip)")
    ap.add_argument("--aug-severity", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA device — training on CPU")

    manifest = pd.read_csv(args.manifest)
    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ManifestDataset(manifest, split="train", mode="train",
                               aug_severity=args.aug_severity, seed=args.seed)
    # If eyeglasses positives are rare (<15%), oversample them mildly rather
    # than to 50/50 — heavy oversampling distorts calibration.
    labels = train_ds.df["label_id"].to_numpy()
    pos_frac = (labels == 1).mean()
    sampler = None
    if pos_frac < 0.15:
        w = np.where(labels == 1, 1.8, 1.0)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(labels),
            replacement=True)
        print(f"eyeglasses fraction {pos_frac:.3f} < 0.15 -> weighted sampler on")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)
    print(f"train={len(train_ds)} "
          f"val={(manifest['split'] == 'val').sum()} "
          f"cal={(manifest['split'] == 'cal').sum()}")

    model = build_model(pretrained=True, freeze_backbone=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    print("== stage 1: head warmup ==")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3, weight_decay=1e-4)
    run_stage(model, "s1", train_loader, manifest, device, args.head_epochs, opt)

    print("== stage 2: last-4-block fine-tune ==")
    unfreeze_backbone(model, last_n_blocks=4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=1e-4)
    best = run_stage(model, "s2", train_loader, manifest, device,
                     args.finetune_epochs, opt)

    if args.full_epochs > 0:
        print("== stage 3: full unfreeze ==")
        unfreeze_all(model)
        opt = torch.optim.AdamW(
            backbone_head_param_groups(model, backbone_lr=1e-4, head_lr=5e-4),
            weight_decay=0.05)
        best = run_stage(model, "s3", train_loader, manifest, device,
                         args.full_epochs, opt)

    # ---- final artifacts ----
    ckpt = run_dir / "best.pt"
    torch.save({"model": model.state_dict(),
                "worst_condition_acc": best["worst_acc"]}, ckpt)

    clean, cond = best["clean"], best["cond"]
    cond.to_csv(run_dir / "per_condition.csv", index=False)
    sheets = make_error_sheets(model, manifest, "val", device, run_dir / "sheets")

    report = [
        f"# Run {args.run_name}",
        f"date: {datetime.datetime.now():%Y-%m-%d %H:%M} | commit: {git_commit()} "
        f"| seed: {args.seed} | aug_severity: {args.aug_severity} "
        f"| train_n: {len(train_ds)}",
        "", "## Clean val (binary, thresh 0.5)", str(clean),
        "", "## Per-condition binary accuracy",
        cond.pivot(index="condition", columns="severity",
                   values="binary_acc").round(4).to_markdown(),
        "", f"**worst_condition_accuracy = {best['worst_acc']:.4f}**",
        f"errors on val: {sheets['n_fp']} FP / {sheets['n_fn']} FN "
        "(see sheets/*.png)",
    ]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    runs_csv = Path(args.out_dir) / "runs.csv"
    write_header = not runs_csv.exists()
    with open(runs_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run", "date", "commit", "seed", "train_n", "aug_severity",
                        "clean_acc", "recall", "sun_fpr", "auc", "ece",
                        "worst_cond_acc"])
        w.writerow([args.run_name, f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
                    git_commit(), args.seed, len(train_ds), args.aug_severity,
                    f"{clean.accuracy:.4f}", f"{clean.recall:.4f}",
                    f"{clean.sunglasses_fpr:.4f}", f"{clean.auc:.4f}",
                    f"{clean.ece:.4f}", f"{best['worst_acc']:.4f}"])

    print(f"\nsaved {ckpt}\nreport: {run_dir / 'report.md'}")
    print(f"worst_condition_acc={best['worst_acc']:.4f} clean: {clean}")


if __name__ == "__main__":
    main()
