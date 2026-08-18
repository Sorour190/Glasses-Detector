"""Glasses classifier: 3-class training head, binary production output.

MobileNetV3-Small backbone (ImageNet pretrained), 160x160 landmark-ROI input.
Classes: 0=none, 1=eyeglasses, 2=sunglasses. Production consumes only
p_glasses = softmax(logits)[1]; the full vector is kept for telemetry and so
the sunglasses hard-negative boundary gets its own gradient path.
"""

import torch
import torch.nn as nn
from torchvision import models

from .preprocess import IMAGE_SIZE  # noqa: F401  (single source of truth: 160)

NUM_CLASSES = 3
# ImageNet normalization — must match dataset._to_tensor and the ONNX graph.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def unfreeze_backbone(model: nn.Module, last_n_blocks: int = 4) -> None:
    """Stage 2: unfreeze the last N feature blocks."""
    for block in model.features[-last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Stage 3: full fine-tune."""
    for param in model.parameters():
        param.requires_grad = True


def backbone_head_param_groups(model: nn.Module, backbone_lr: float, head_lr: float):
    return [
        {"params": [p for p in model.features.parameters() if p.requires_grad],
         "lr": backbone_lr},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad],
         "lr": head_lr},
    ]


def load_checkpoint(path: str, device: str = "cpu") -> nn.Module:
    model = build_model(pretrained=False, freeze_backbone=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    return model
