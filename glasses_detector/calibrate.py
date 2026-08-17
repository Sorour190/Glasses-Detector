"""Pick a decision threshold from validation data.

For onboarding, missing glasses (false negative) is usually worse than
asking a user to re-capture (false positive). This script finds the
highest threshold that still catches a target fraction of glasses frames
on the validation set, and reports the false-rejection cost.

Usage:
    python -m glasses_detector.calibrate \
        --checkpoint checkpoints/glasses.pt --data-dir data --target-recall 0.99
"""

import argparse
import json
from pathlib import Path

import torch

from .dataset import make_loaders
from .metrics import compute_metrics, confusion
from .model import load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    probs, labels = [], []
    for images, batch_labels in loader:
        batch_probs = torch.sigmoid(model(images.to(device))).squeeze(1)
        probs.extend(batch_probs.cpu().tolist())
        labels.extend(float(y) for y in batch_labels)
    return probs, labels


def calibrate(probs, labels, target_recall):
    """Highest threshold whose recall still meets the target.

    Higher thresholds only lose positives, so recall is monotonically
    non-increasing in the threshold — take the highest candidate that
    still qualifies to minimize false positives.
    """
    candidates = sorted(set(probs), reverse=True)
    best = 0.5
    for threshold in candidates:
        tp, fp, tn, fn = confusion(probs, labels, threshold)
        recall = tp / (tp + fn) if tp + fn else 0.0
        if recall >= target_recall:
            best = threshold
            break
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--target-recall", type=float, default=0.99,
                        help="Fraction of glasses frames that must be caught")
    parser.add_argument("--out", default=None,
                        help="Optional JSON file to write the chosen threshold to")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    _, val_loader = make_loaders(args.data_dir)
    probs, labels = collect_predictions(model, val_loader, device)

    threshold = calibrate(probs, labels, args.target_recall)
    print(f"threshold={threshold:.4f} for target recall {args.target_recall}")
    print(f"at this threshold: {compute_metrics(probs, labels, threshold)}")

    print("\noperating points:")
    for t in (0.2, 0.35, 0.5, 0.65, 0.8):
        print(f"  threshold {t:.2f}: {compute_metrics(probs, labels, t)}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "threshold": threshold,
            "target_recall": args.target_recall,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
