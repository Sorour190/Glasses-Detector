"""Inference API for the onboarding validation step.

Full production path: SCRFD face/landmark detection -> ROI-v1 eye-region crop
-> 3-class model -> p_glasses = P(eyeglasses). Sunglasses count as negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from .dataset import MEAN, STD
from .model import load_checkpoint
from .preprocess import IMAGE_SIZE, roi_crop
from .scrfd import SCRFD

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class GlassesResult:
    wearing_glasses: bool
    confidence: float      # confidence in the returned label, in [0.5, 1.0]
    probability: float     # calibrated-ish P(eyeglasses), in [0, 1]
    uncertain: bool        # inside the uncertainty band -> retry the capture
    face_found: bool
    det_score: float
    class_probs: tuple     # (none, eyeglasses, sunglasses) — telemetry only


class GlassesDetector:
    """Detects whether the person in an image is wearing clear eyeglasses."""

    def __init__(self, checkpoint: Union[str, Path],
                 scrfd_model: Union[str, Path] = "models/det_500m.onnx",
                 device: str | None = None, threshold: float = 0.5,
                 uncertainty_band: float = 0.15):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_checkpoint(str(checkpoint), self.device)
        self.detector = SCRFD(scrfd_model)
        self.threshold = threshold
        self.uncertainty_band = uncertainty_band

    def _load_bgr(self, image: ImageInput) -> np.ndarray:
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(image)
            return bgr
        return image

    @torch.no_grad()
    def predict(self, image: ImageInput) -> GlassesResult:
        bgr = self._load_bgr(image)
        det = self.detector.detect(bgr)
        if det is None:
            return GlassesResult(False, 0.5, 0.0, True, False, 0.0, (1.0, 0.0, 0.0))
        _, det_score, kps = det

        crop = roi_crop(bgr, kps, out_size=IMAGE_SIZE)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        probability = float(probs[1])
        wearing = probability >= self.threshold
        confidence = probability if wearing else 1.0 - probability
        uncertain = abs(probability - self.threshold) < self.uncertainty_band
        return GlassesResult(wearing, confidence, probability, uncertain,
                             True, det_score, tuple(round(float(p), 4) for p in probs))
