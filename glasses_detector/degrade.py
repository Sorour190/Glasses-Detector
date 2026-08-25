"""Frozen evaluation degradations (DEGRADE_V1).

Six conditions x three severities, implemented directly in cv2/numpy —
deliberately DISJOINT from the albumentations training pipeline (different
code, different parameter ranges) so per-condition eval measures robustness,
not memorization of the training augmentation. Deterministic per image index.

Do not edit parameters after test sets are frozen; bump the version instead.
"""

from __future__ import annotations

import hashlib

import numpy as np
import cv2

DEGRADE_VERSION = "degrade-v1"

# severity index: 1, 2, 3
_PARAMS = {
    "motion_blur": {1: 7, 2: 13, 3: 21},                 # kernel length
    "gauss_noise": {1: 12.0, 2: 25.0, 3: 45.0},          # sigma (uint8 scale)
    "low_light": {1: 0.55, 2: 0.40, 3: 0.28},            # brightness factor
    "pose": {1: 12.0, 2: 22.0, 3: 32.0},                 # rotation deg + perspective
    "jpeg": {1: 45, 2: 25, 3: 12},                       # quality
    "downscale": {1: 0.55, 2: 0.40, 3: 0.28},            # scale factor
}

CONDITIONS = tuple(_PARAMS)
SEVERITIES = (1, 2, 3)

# Composite, full-frame profiles used by the live test page. Unlike the
# condition-isolated evaluation above, these deliberately stack the defects a
# weak phone camera produces before face detection and ROI extraction.
BAD_CAMERA_PROFILES = {
    "clean": None,
    "mild": {"scale": 0.75, "motion_k": 5, "brightness": 0.82,
             "noise_sigma": 5.0, "jpeg_quality": 55},
    "bad_phone": {"scale": 0.50, "motion_k": 9, "brightness": 0.58,
                  "noise_sigma": 12.0, "jpeg_quality": 25},
    "extreme": {"scale": 0.40, "motion_k": 11, "brightness": 0.50,
                "noise_sigma": 15.0, "jpeg_quality": 20},
}


def _rng_for(index: int, condition: str, severity: int) -> np.random.Generator:
    seed = hash((index, condition, severity, DEGRADE_VERSION)) & 0x7FFFFFFF
    return np.random.default_rng(seed)


def _profile_rng(index: int, profile: str) -> np.random.Generator:
    payload = f"{DEGRADE_VERSION}:{profile}:{index}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def apply_bad_camera(bgr: np.ndarray, profile: str, index: int = 0) -> np.ndarray:
    """Apply a deterministic composite degradation to a full submitted frame.

    Shape and uint8 encoding are preserved so the returned image can be sent
    through the exact production detector/crop/model path.
    """
    if profile not in BAD_CAMERA_PROFILES:
        raise ValueError(f"unknown bad-camera profile: {profile}")
    params = BAD_CAMERA_PROFILES[profile]
    if params is None:
        return bgr.copy()

    rng = _profile_rng(index, profile)
    h, w = bgr.shape[:2]
    sw = max(1, round(w * params["scale"]))
    sh = max(1, round(h * params["scale"]))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    k = int(params["motion_k"])
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.line(kernel, (0, k // 2), (k - 1, k // 2), 1.0, 1)
    angle = float(rng.uniform(0, 180))
    matrix = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (k, k))
    kernel /= max(float(kernel.sum()), 1e-6)
    img = cv2.filter2D(img, -1, kernel)

    img = img.astype(np.float32) * params["brightness"]
    img += rng.normal(0.0, params["noise_sigma"], img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(params["jpeg_quality"])]
    )
    if not ok:
        raise RuntimeError("failed to encode simulated bad-camera frame")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply(bgr: np.ndarray, condition: str, severity: int, index: int = 0) -> np.ndarray:
    """Apply one frozen degradation. `index` keeps randomness deterministic per image."""
    p = _PARAMS[condition][severity]
    rng = _rng_for(index, condition, severity)
    img = bgr

    if condition == "motion_blur":
        k = int(p)
        kernel = np.zeros((k, k), dtype=np.float32)
        angle = rng.uniform(0, 180)
        cv2.line(kernel, (0, k // 2), (k - 1, k // 2), 1.0, 1)
        m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, m, (k, k))
        kernel /= max(kernel.sum(), 1e-6)
        return cv2.filter2D(img, -1, kernel)

    if condition == "gauss_noise":
        noise = rng.normal(0.0, p, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if condition == "low_light":
        dark = img.astype(np.float32) * p
        # low light comes with sensor noise
        dark += rng.normal(0.0, 8.0, img.shape)
        return np.clip(dark, 0, 255).astype(np.uint8)

    if condition == "pose":
        h, w = img.shape[:2]
        angle = rng.choice([-1, 1]) * p
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # mild perspective as a yaw/pitch proxy, scaled with severity
        f = p / 300.0
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = src + np.float32(rng.uniform(-f * w, f * w, (4, 2)))
        pm = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(rotated, pm, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if condition == "jpeg":
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(p)])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if condition == "downscale":
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(8, int(w * p)), max(8, int(h * p))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    raise ValueError(f"unknown condition: {condition}")
