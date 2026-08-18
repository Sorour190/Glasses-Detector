"""Manifest-driven dataset with ROI-v1 cropping and two-stage augmentation.

Stage A (geometric, pre-crop): landmark jitter + horizontal flip, applied to
the SCRFD landmarks BEFORE the ROI warp — simulates detector localization
error, the real production noise source.

Stage B (pixel-level): albumentations on the 160x160 crop, scaled by a
severity factor `s` (0.4 early runs -> 1.0 later) with a 15% always-clean
branch so clean accuracy never sags.

Manifest columns (see scripts/build_manifest.py): path, source, label,
label_id, det_score, kp0x..kp4y, phash, cluster, split.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import degrade
from .preprocess import IMAGE_SIZE, flip_landmarks, jitter_landmarks, roi_crop

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ("none", "eyeglasses", "sunglasses")


def build_train_aug(s: float = 1.0) -> A.Compose:
    """Stage-B pixel augmentation; probabilities of degradations scale with s."""
    return A.Compose([
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=0.05,
                 shear=(-8, 8), p=0.6),
        A.Perspective(scale=(0.02, 0.07), p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 15)),
            A.GaussianBlur(blur_limit=(3, 9)),
            A.Defocus(radius=(2, 5)),
        ], p=0.35 * s),
        A.Downscale(scale_range=(0.4, 0.9), p=0.20 * s),
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.18)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.30 * s),
        # asymmetric ranges: real failures (SoF benchmark) are severe
        # UNDERexposure with hard shadows, not mild jitter
        A.RandomBrightnessContrast(brightness_limit=(-0.55, 0.3),
                                   contrast_limit=(-0.5, 0.3), p=0.7),
        A.RandomGamma((55, 140), p=0.4),
        A.CLAHE(p=0.1),
        A.HueSaturationValue(10, 15, 10, p=0.2),
        A.OneOf([A.RandomSunFlare(src_radius=60), A.RandomShadow()],
                p=0.15 * s),
        A.ToGray(p=0.15),
        A.ImageCompression(quality_range=(30, 90), p=0.40 * s),
        # constrained: a single hole that can never cover both eyes
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(0.05, 0.18),
                        hole_width_range=(0.10, 0.35), p=0.20),
    ])


def _to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class ManifestDataset(Dataset):
    """Face crops via ROI-v1. Modes: 'train' (aug), 'eval' (clean deterministic),
    or a (condition, severity) tuple for frozen degraded eval."""

    def __init__(self, manifest: pd.DataFrame | str | Path, split: str | None = None,
                 mode: str | tuple = "eval", aug_severity: float = 1.0,
                 p_clean: float = 0.15, seed: int = 42):
        df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
        if split is not None:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.p_clean = p_clean
        self.seed = seed
        self.aug = build_train_aug(aug_severity) if mode == "train" else None
        self._kp_cols = [f"kp{i}{ax}" for i in range(5) for ax in "xy"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        bgr = cv2.imread(row["path"])
        if bgr is None:
            raise FileNotFoundError(row["path"])
        kps = np.array(row[self._kp_cols], dtype=np.float64).reshape(5, 2)

        if self.mode == "train":
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                bgr = bgr[:, ::-1].copy()
                kps = flip_landmarks(kps, bgr.shape[1])
            kps = jitter_landmarks(kps, rng)
            crop = roi_crop(bgr, kps)
            if rng.random() > self.p_clean:
                crop = self.aug(image=crop)["image"]
        else:
            crop = roi_crop(bgr, kps)
            if isinstance(self.mode, tuple):
                condition, severity = self.mode
                crop = degrade.apply(crop, condition, severity, index=idx)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return _to_tensor(rgb), int(row["label_id"]), idx
