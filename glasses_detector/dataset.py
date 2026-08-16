"""Dataset utilities.

Expects a simple folder layout of face crops:

    data/
      train/
        glasses/
        no_glasses/
      val/
        glasses/
        no_glasses/

`scripts/prepare_celeba.py` can generate this layout from CelebA.
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model import IMAGE_SIZE, MEAN, STD

# Positive class. ImageFolder sorts class dirs alphabetically:
# glasses -> 0, no_glasses -> 1, so we remap below to glasses=1.
POSITIVE_CLASS = "glasses"


def _target_transform(class_to_idx: dict):
    positive_idx = class_to_idx[POSITIVE_CLASS]
    return lambda y: 1.0 if y == positive_idx else 0.0


def train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        # Simulate webcam variance during onboarding: lighting, blur, slight tilt.
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        # Simulate partial occlusion (hair, hands, cropped frames) so the
        # model can't rely on any single region always being visible.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def make_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 4):
    root = Path(data_dir)
    train_ds = datasets.ImageFolder(root / "train", transform=train_transforms())
    val_ds = datasets.ImageFolder(root / "val", transform=eval_transforms())

    for ds in (train_ds, val_ds):
        if POSITIVE_CLASS not in ds.class_to_idx:
            raise ValueError(
                f"Expected a '{POSITIVE_CLASS}' class directory in {ds.root}, "
                f"found {list(ds.class_to_idx)}"
            )
        ds.target_transform = _target_transform(ds.class_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
