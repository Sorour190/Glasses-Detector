"""Train the glasses classifier.

Two-stage transfer learning:
  1. Warm up the new head with the backbone frozen.
  2. Unfreeze the last backbone blocks and fine-tune at a lower LR.

Usage:
    python -m glasses_detector.train --data-dir data --out checkpoints/glasses.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .dataset import make_loaders
from .model import build_model, unfreeze_backbone


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return loss_sum / total, correct / total


def train_epochs(model, loaders, device, epochs, lr):
    train_loader, val_loader = loaders
    criterion = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None
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

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_acc, best_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="checkpoints/glasses.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(pretrained=True, freeze_backbone=True).to(device)

    print("stage 1: head warmup")
    train_epochs(model, loaders, device, epochs=args.head_epochs, lr=1e-3)

    print("stage 2: fine-tune last backbone blocks")
    unfreeze_backbone(model, last_n_blocks=4)
    best_acc, best_state = train_epochs(
        model, loaders, device, epochs=args.finetune_epochs, lr=1e-4
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "val_acc": best_acc}, out)
    print(f"saved best model (val_acc={best_acc:.4f}) to {out}")


if __name__ == "__main__":
    main()
