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
