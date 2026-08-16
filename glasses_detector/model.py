"""Binary glasses/no-glasses classifier.

MobileNetV3-Small backbone (ImageNet pretrained) with a single-logit head.
The input is expected to be a face crop — the upstream identity model already
localizes the face, so no face detection happens here.
"""

import torch
import torch.nn as nn
from torchvision import models

IMAGE_SIZE = 224
# ImageNet normalization — must match training and inference.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the 1000-class head with a single logit: P(wearing glasses).
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    )
    return model


def unfreeze_backbone(model: nn.Module, last_n_blocks: int = 4) -> None:
    """Unfreeze the last N feature blocks for fine-tuning after head warmup."""
    for block in model.features[-last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


def load_checkpoint(path: str, device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False, freeze_backbone=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    return model
