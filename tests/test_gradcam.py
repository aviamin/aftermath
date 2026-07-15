import numpy as np
import torch

from utils.model import SiameseDamageNet
from utils.gradcam import GradCAM, resize_cam


def test_gradcam_generate_returns_normalized_heatmap():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    target_layer = model.backbone.layer4[-1]
    cam_tool = GradCAM(model, target_layer)

    pre = torch.randn(1, 3, 224, 224)
    post = torch.randn(1, 3, 224, 224)

    cam = cam_tool.generate(pre, post, target_class=0, branch="post")

    assert cam.ndim == 2
    assert cam.shape == (7, 7)  # resnet18 layer4 output spatial size for 224x224 input
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_gradcam_works_for_pre_branch_too():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    target_layer = model.backbone.layer4[-1]
    cam_tool = GradCAM(model, target_layer)

    pre = torch.randn(1, 3, 224, 224)
    post = torch.randn(1, 3, 224, 224)

    cam = cam_tool.generate(pre, post, target_class=1, branch="pre")

    assert cam.shape == (7, 7)


def test_resize_cam_upsamples_to_target_size():
    cam = np.random.rand(7, 7).astype(np.float32)

    resized = resize_cam(cam, size=224)

    assert resized.shape == (224, 224)


def test_gradcam_activation_order_matches_pre_then_post():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    model.eval()
    target_layer = model.backbone.layer4[-1]

    # Independent probe hook (separate from GradCAM's own hook) to capture
    # ground-truth layer4 outputs for each input processed in isolation.
    probe_activations = []

    def probe_hook(module, input, output):
        probe_activations.append(output.detach().clone())

    handle = target_layer.register_forward_hook(probe_hook)
    with torch.no_grad():
        model.backbone(torch.zeros(1, 3, 224, 224))
        model.backbone(torch.ones(1, 3, 224, 224))
    handle.remove()
    expected_pre_activation, expected_post_activation = probe_activations

    cam_tool = GradCAM(model, target_layer)
    pre = torch.zeros(1, 3, 224, 224)
    post = torch.ones(1, 3, 224, 224)
    cam_tool.generate(pre, post, target_class=0, branch="post")

    assert torch.allclose(cam_tool.activations[0].detach(), expected_pre_activation, atol=1e-5)
    assert torch.allclose(cam_tool.activations[1].detach(), expected_post_activation, atol=1e-5)
    assert not torch.allclose(expected_pre_activation, expected_post_activation, atol=1e-4)
