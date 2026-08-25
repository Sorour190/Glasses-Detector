"""Fast classical restoration for degraded-camera crops (RESTORE_V1).

Counters the invertible parts of the bad-camera profiles (degrade.py):
Gaussian sensor noise, global darkening, and some of the motion blur.
Applied to the 160px eye ROI only — cheap enough (~10-40 ms CPU) for the
live 5 fps path. The downscale and JPEG information loss are not
recoverable here; deep super-resolution is a documented follow-up.

Pure and deterministic: same crop in, same crop out. Quality metrics are
always measured on the unrestored crop (predict.measure_quality), so the
min_blur / brightness / contrast gates keep their calibrated meaning.
"""

from __future__ import annotations

import cv2
import numpy as np

RESTORE_VERSION = "restore-v1"

_TARGET_BRIGHTNESS = 110.0   # mean gray the gain step aims for
_MAX_GAIN = 2.2              # profiles darken by 0.50-0.82 -> gain <= 2 needed
_NLM_H = 7                   # fastNlMeans strength; profile noise sigma is 5-15
_UNSHARP_SIGMA = 1.5
_UNSHARP_AMOUNT = 0.7


def restore_crop(bgr: np.ndarray) -> np.ndarray:
    """Brighten, denoise, and re-sharpen a degraded BGR crop.

    Order matters: gain first (so denoising sees the amplified noise it must
    remove), then non-local-means denoise, CLAHE on luminance, and finally an
    unsharp mask against the motion blur.
    """
    img = bgr
    mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    if 0.0 < mean < _TARGET_BRIGHTNESS:
        gain = min(_TARGET_BRIGHTNESS / mean, _MAX_GAIN)
        img = cv2.convertScaleAbs(img, alpha=gain, beta=0.0)

    img = cv2.fastNlMeansDenoisingColored(img, None, _NLM_H, _NLM_H, 7, 21)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    soft = cv2.GaussianBlur(img, (0, 0), _UNSHARP_SIGMA)
    return cv2.addWeighted(img, 1.0 + _UNSHARP_AMOUNT, soft, -_UNSHARP_AMOUNT, 0)
