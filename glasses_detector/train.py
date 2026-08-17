"""Train the glasses classifier.

Two-stage transfer learning:
  1. Warm up the new head with the backbone frozen.
  2. Unfreeze the last backbone blocks and fine-tune at a lower LR.

Class imbalance is handled with a positive-class weight in the loss, and
model selection / early stopping use validation AUC rather than accuracy
(accuracy is misleading when classes are imbalanced).

Usage:
    python -m glasses_detector.train --data-dir data --out checkpoints/glasses.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .calibrate import collect_predictions
from .dataset import POSITIVE_CLASS, make_loaders
from .metrics import compute_metrics
from .model import build_model, unfreeze_backbone


def positive_class_weight(train_loader) -> float:
    """neg/pos ratio of the training set, used as BCE pos_weight."""
    dataset = train_loader.dataset
    positive_idx = dataset.class_to_idx[POSITIVE_CLASS]
    n_pos = sum(1 for t in dataset.targets if t == positive_idx)
    n_neg = len(dataset.targets) - n_pos
    if n_pos == 0:
        raise ValueError("Training set has no glasses examples")
    return n_neg / n_pos


def train_epochs(model, loaders, device, epochs, lr, pos_weight, patience=3):
    train_loader, val_loader = loaders
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        probs, labels = collect_predictions(model, val_loader, device)
        metrics = compute_metrics(probs, labels)
        print(f"epoch {epoch + 1}/{epochs}  {metrics}")

        if metrics.auc > best_auc:
            best_auc = metrics.auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stop: no val AUC improvement in {patience} epochs")
                break
    return best_auc, best_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="checkpoints/glasses.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    pos_weight = positive_class_weight(loaders[0])
    print(f"pos_weight={pos_weight:.2f}")

    model = build_model(pretrained=True, freeze_backbone=True).to(device)

    print("stage 1: head warmup")
    train_epochs(model, loaders, device, epochs=args.head_epochs, lr=1e-3,
                 pos_weight=pos_weight, patience=args.patience)

    print("stage 2: fine-tune last backbone blocks")
    unfreeze_backbone(model, last_n_blocks=4)
    best_auc, best_state = train_epochs(
        model, loaders, device, epochs=args.finetune_epochs, lr=1e-4,
        pos_weight=pos_weight, patience=args.patience,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "val_auc": best_auc}, out)
    print(f"saved best model (val_auc={best_auc:.4f}) to {out}")
    print("next: pick a production threshold with `python -m glasses_detector.calibrate`")


if __name__ == "__main__":
    main()
