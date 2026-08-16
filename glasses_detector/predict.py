"""Inference API for the onboarding validation step."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from .dataset import eval_transforms
from .model import load_checkpoint

ImageInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class GlassesResult:
    wearing_glasses: bool
    confidence: float  # confidence in the returned label, in [0.5, 1.0]
    probability: float  # raw P(wearing glasses), in [0, 1]
    uncertain: bool  # True when probability falls inside the uncertainty band


class GlassesDetector:
    """Detects whether the person in a face crop is wearing glasses.

    The image should be the same face crop the identity model receives.
    Results inside the uncertainty band should be treated as "retry the
    capture" rather than a hard accept/reject.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: str = None,
        threshold: float = 0.5,
        uncertainty_band: float = 0.15,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_checkpoint(str(checkpoint), self.device)
        self.transform = eval_transforms()
        self.threshold = threshold
        self.uncertainty_band = uncertainty_band

    def _to_tensor(self, image: ImageInput) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: ImageInput) -> GlassesResult:
        tensor = self._to_tensor(image)
        probability = torch.sigmoid(self.model(tensor)).item()
        wearing = probability >= self.threshold
        confidence = probability if wearing else 1.0 - probability
        uncertain = abs(probability - self.threshold) < self.uncertainty_band
        return GlassesResult(
            wearing_glasses=wearing,
            confidence=confidence,
            probability=probability,
            uncertain=uncertain,
        )
