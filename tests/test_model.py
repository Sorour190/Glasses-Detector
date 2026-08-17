import torch

from glasses_detector.model import IMAGE_SIZE, build_model, unfreeze_backbone


def test_forward_shape():
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (2, 1)


def test_backbone_frozen_by_default():
    model = build_model(pretrained=False, freeze_backbone=True)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_unfreeze_backbone():
    model = build_model(pretrained=False, freeze_backbone=True)
    unfreeze_backbone(model, last_n_blocks=2)
    assert any(p.requires_grad for p in model.features[-1].parameters())
    assert not any(p.requires_grad for p in model.features[0].parameters())
