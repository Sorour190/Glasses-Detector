"""ROI-v1: landmark-based eye-region crop, shared by data prep, training, serving.

Given the 5 SCRFD landmarks, we de-rotate the face so the eye line is
horizontal, center the crop slightly below the eye midpoint, and take a
region of 3.0d x 2.0d (d = inter-ocular distance) warped to IMAGE_SIZE^2.
Width 3.0d deliberately includes the temples/arms of the frame — a strong,
tint-independent glasses cue.

Training-time landmark jitter simulates SCRFD localization error (the actual
production noise source) and MUST be applied to the landmarks BEFORE calling
roi_crop, never to the pixels after.
"""

from __future__ import annotations

import numpy as np
import cv2

PREPROCESS_VERSION = "roi-v1"
IMAGE_SIZE = 160

# Crop geometry, in units of inter-ocular distance d.
_HALF_W = 1.5
_HALF_H = 1.0
_CENTER_DOWN = 0.10


def roi_crop(bgr: np.ndarray, kps: np.ndarray, out_size: int = IMAGE_SIZE) -> np.ndarray:
    """Warp the eye-region ROI to (out_size, out_size). kps is (5,2): le, re, ...."""
    le, re = kps[0].astype(np.float64), kps[1].astype(np.float64)
    d = float(np.linalg.norm(re - le))
    if d < 1e-3:
        raise ValueError("degenerate landmarks: zero inter-ocular distance")
    c = (le + re) / 2.0
    ex = (re - le) / d                       # unit vector along the eye line
    ey = np.array([-ex[1], ex[0]])           # perpendicular, pointing image-down
    cc = c + _CENTER_DOWN * d * ey           # crop center

    src = np.float32([
        cc - _HALF_W * d * ex - _HALF_H * d * ey,   # top-left
        cc + _HALF_W * d * ex - _HALF_H * d * ey,   # top-right
        cc - _HALF_W * d * ex + _HALF_H * d * ey,   # bottom-left
    ])
    dst = np.float32([[0, 0], [out_size, 0], [0, out_size]])
    m = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(bgr, m, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def jitter_landmarks(kps: np.ndarray, rng: np.random.Generator,
                     sigma_frac: float = 0.03, scale_lo: float = 0.92,
                     scale_hi: float = 1.10, angle_deg: float = 3.0) -> np.ndarray:
    """Simulate SCRFD landmark noise: gaussian jitter on the eye points plus a
    small global scale/rotation of the eye pair about its midpoint."""
    kps = kps.astype(np.float64).copy()
    le, re = kps[0], kps[1]
    d = float(np.linalg.norm(re - le))
    c = (le + re) / 2.0

    # global scale + rotation of the eye pair
    s = rng.uniform(scale_lo, scale_hi)
    a = np.deg2rad(rng.uniform(-angle_deg, angle_deg))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    for i in (0, 1):
        kps[i] = c + s * rot @ (kps[i] - c)

    # independent per-point gaussian noise
    kps[0] += rng.normal(0.0, sigma_frac * d, size=2)
    kps[1] += rng.normal(0.0, sigma_frac * d, size=2)
    return kps


def flip_landmarks(kps: np.ndarray, image_width: int) -> np.ndarray:
    """Horizontally flip landmarks (and swap left/right pairs)."""
    kps = kps.copy()
    kps[:, 0] = image_width - 1 - kps[:, 0]
    kps[[0, 1]] = kps[[1, 0]]                # swap eyes
    kps[[3, 4]] = kps[[4, 3]]                # swap mouth corners
    return kps
