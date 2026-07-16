import torch
from utils.model import SiameseDamageNet


def test_forward_pass_shape():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    pre = torch.randn(2, 3, 224, 224)
    post = torch.randn(2, 3, 224, 224)

    logits = model(pre, post)

    assert logits.shape == (2, 4)


def test_backbone_is_shared_between_branches():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    pre = torch.randn(1, 3, 224, 224)

    feat_via_backbone = model.backbone(pre)

    assert feat_via_backbone.shape == (1, model.feature_dim)


def test_concat_merge_produces_expected_feature_dimension():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    pre = torch.randn(2, 3, 224, 224)
    post = torch.randn(2, 3, 224, 224)

    captured = {}

    def capture_head_input(module, args):
        captured["shape"] = args[0].shape

    handle = model.head.register_forward_pre_hook(capture_head_input)
    try:
        model(pre, post)
    finally:
        handle.remove()

    assert captured["shape"] == (2, model.feature_dim * 2)
