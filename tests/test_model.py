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


def test_rebuild_sequential_head_matches_checkpoint_layout():
    from glasses_detector.model import _rebuild_sequential_head
    state = {
        "classifier.0.weight": torch.zeros(256, 768), "classifier.0.bias": torch.zeros(256),
        "classifier.2.weight": torch.zeros(64, 256), "classifier.2.bias": torch.zeros(64),
        "classifier.4.weight": torch.zeros(2, 64), "classifier.4.bias": torch.zeros(2),
    }
    head = _rebuild_sequential_head(state)
    head.load_state_dict({k.removeprefix("classifier."): v for k, v in state.items()})
    with torch.no_grad():
        out = head(torch.randn(3, 768))
    assert out.shape == (3, 2)
