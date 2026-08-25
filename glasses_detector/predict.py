"""Inference API for the onboarding validation step.

Full production path: SCRFD face/landmark detection -> ROI-v1 eye-region crop
-> 3-class model -> probability = P(eyeglasses) (legacy/calibrated) and
eyewear_prob = P(eyeglasses)+P(sunglasses), which the verification gate decides on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from .dataset import MEAN, STD
from .model import load_checkpoint, load_hf_checkpoint
from .preprocess import IMAGE_SIZE, roi_crop
from .restore import restore_crop
from .scrfd import SCRFD

ImageInput = Union[str, Path, np.ndarray]
QUALITY_VERSION = "quality-v2"


@dataclass
class GlassesResult:
    wearing_glasses: bool
    confidence: float      # confidence in the returned label, in [0.5, 1.0]
    probability: float     # calibrated-ish P(eyeglasses), in [0, 1]
    uncertain: bool        # inside the uncertainty band -> retry the capture
    face_found: bool
    det_score: float
    class_probs: tuple     # (none, eyeglasses, sunglasses) — telemetry only
    eyewear_prob: float = 0.0   # P(eyeglasses) + P(sunglasses) = 1 - P(none): "any glasses"
    blur_score: float = 0.0     # denoised Laplacian variance of the crop (higher = sharper)
    eye_dist_px: float = 0.0    # inter-ocular distance in the submitted frame (face size proxy)
    brightness: float = 128.0   # mean grayscale value on the unmodified crop, [0, 255]
    contrast: float = 100.0     # grayscale p95 - p5 dynamic range, [0, 255]
    restored: bool = False      # crop was restoration-enhanced before the model saw it
    crop: Optional[np.ndarray] = field(default=None, repr=False, compare=False)  # BGR crop, logging only


@dataclass(frozen=True)
class FrameQuality:
    blur_score: float
    brightness: float
    contrast: float


def measure_quality(crop_bgr: np.ndarray) -> FrameQuality:
    """Measure capture quality without changing the pixels sent to the model.

    A small Gaussian blur is used only on the measurement branch so sensor
    noise cannot masquerade as focus in the Laplacian score.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (11, 11), 2.5)
    blur = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
    brightness = float(gray.mean())
    p5, p95 = np.percentile(gray, (5, 95))
    return FrameQuality(blur, brightness, float(p95 - p5))


def blur_score(crop_bgr: np.ndarray) -> float:
    """Backward-compatible access to the quality-v2 blur measurement."""
    return measure_quality(crop_bgr).blur_score


class GlassesDetector:
    """Detects whether the person in an image is wearing clear eyeglasses."""

    def __init__(self, checkpoint: Union[str, Path],
                 scrfd_model: Union[str, Path] = "models/det_500m.onnx",
                 device: str | None = None, threshold: float = 0.5,
                 uncertainty_band: float = 0.15,
                 threshold_file: Union[str, Path, None] = "models/threshold.json"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = str(checkpoint)
        if Path(checkpoint).is_dir():
            # HuggingFace checkpoint directory (e.g. a binary ConvNeXt classifier).
            self.backend = "hf"
            self.model = load_hf_checkpoint(self.checkpoint, self.device)
            self.model_input_size = getattr(self.model.config, "image_size", 224) or 224
            self.hf_input = os.environ.get("GLASSES_HF_INPUT", "face")  # face | roi | full
            threshold_file = None  # the v1 temperature/band calibration does not transfer
        else:
            self.backend = "torch"
            self.model = load_checkpoint(self.checkpoint, self.device)
        self.detector = SCRFD(scrfd_model)
        self.threshold = threshold
        self.uncertainty_band = uncertainty_band
        # calibrated operating point (temperature + abstain band), if present
        self.temperature, self.band = 1.0, None
        if threshold_file and Path(threshold_file).exists():
            import json
            cfg = json.loads(Path(threshold_file).read_text())
            self.temperature = cfg.get("T", 1.0)
            self.band = (cfg["t_low"], cfg["t_high"])
            self.threshold = cfg["t_high"]

    def _load_bgr(self, image: ImageInput) -> np.ndarray:
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(image)
            return bgr
        return image

    def _model_input(self, bgr: np.ndarray, bbox: np.ndarray,
                     kps: np.ndarray) -> np.ndarray:
        """Crop fed to an HF checkpoint (GLASSES_HF_INPUT: face | roi | full)."""
        size = self.model_input_size
        if self.hf_input == "roi":
            return roi_crop(bgr, kps, out_size=size)
        if self.hf_input == "full":
            h, w = bgr.shape[:2]
            s = size / min(h, w)
            resized = cv2.resize(bgr, (max(size, round(w * s)), max(size, round(h * s))))
            y0 = (resized.shape[0] - size) // 2
            x0 = (resized.shape[1] - size) // 2
            return resized[y0:y0 + size, x0:x0 + size]
        # "face": square SCRFD box with a 30% margin
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * 0.65
        h, w = bgr.shape[:2]
        x1i, y1i = max(0, int(round(cx - half))), max(0, int(round(cy - half)))
        x2i, y2i = min(w, int(round(cx + half))), min(h, int(round(cy + half)))
        if x2i <= x1i or y2i <= y1i:
            return roi_crop(bgr, kps, out_size=size)
        return cv2.resize(bgr[y1i:y2i, x1i:x2i], (size, size))

    @torch.no_grad()
    def predict(self, image: ImageInput, restore: bool = False) -> GlassesResult:
        """`restore=True` enhances the crop fed to the model (degraded-camera
        profiles); quality metrics are always measured on the raw crop."""
        bgr = self._load_bgr(image)
        det = self.detector.detect(bgr)
        if det is None:
            return GlassesResult(False, 0.5, 0.0, True, False, 0.0,
                                 (1.0, 0.0, 0.0), brightness=0.0, contrast=0.0)
        bbox, det_score, kps = det

        crop = roi_crop(bgr, kps, out_size=IMAGE_SIZE)
        quality = measure_quality(crop)
        model_crop = crop if self.backend == "torch" else self._model_input(bgr, bbox, kps)
        if restore:
            model_crop = restore_crop(model_crop)
        rgb = cv2.cvtColor(model_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        if self.backend == "hf":
            logits = self.model(x).logits[0] / self.temperature
        else:
            logits = self.model(x)[0] / self.temperature
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        if probs.shape[0] == 2:
            # binary head (no_glasses, glasses) -> (none, eyeglasses, sunglasses);
            # sunglasses are folded into the glasses class by such a model
            probs = np.array([probs[0], probs[1], 0.0], dtype=np.float32)
        probability = float(probs[1])
        wearing = probability >= self.threshold
        confidence = probability if wearing else 1.0 - probability
        if self.band is not None:
            uncertain = self.band[0] <= probability < self.band[1]
        else:
            uncertain = abs(probability - self.threshold) < self.uncertainty_band
        return GlassesResult(wearing, confidence, probability, uncertain,
                             True, det_score, tuple(round(float(p), 4) for p in probs),
                             eyewear_prob=float(1.0 - probs[0]),
                             blur_score=quality.blur_score,
                             eye_dist_px=float(np.linalg.norm(kps[1] - kps[0])),
                             brightness=quality.brightness,
                             contrast=quality.contrast,
                             restored=restore,
                             crop=crop)
