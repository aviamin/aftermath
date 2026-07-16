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


def test_gradcam_branch_selection_uses_correctly_paired_activation():
    # Seeded: whether post's ReLU-clipped CAM is non-zero depends on the
    # random init, so an unseeded run is flaky (~1-in-6 failure rate
    # observed across seeds 0-29). Seed 42 deterministically produces a
    # non-zero post_cam, which is what this test needs to be meaningful.
    torch.manual_seed(42)
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    model.eval()
    target_layer = model.backbone.layer4[-1]
    cam_tool = GradCAM(model, target_layer)

    pre = torch.zeros(1, 3, 224, 224)
    post = torch.ones(1, 3, 224, 224)

    pre_cam = cam_tool.generate(pre, post, target_class=0, branch="pre")
    post_cam = cam_tool.generate(pre, post, target_class=0, branch="post")

    # With model.eval() and a freshly-constructed (never-trained) resnet18,
    # an all-zeros input propagates as exactly zero through every bias-free
    # conv + BatchNorm (default running_mean=0/running_var=1/affine identity)
    # + ReLU + residual stage, so layer4's output for `pre` is an all-zero
    # activation tensor. The einsum of ANY gradient against an all-zero
    # activation is zero, so `branch="pre"` must produce an exactly all-zero
    # CAM if (and only if) it is correctly paired with `pre`'s activation.
    # If `act_idx` were swapped, branch="pre" would incorrectly read post's
    # non-zero activation and produce a non-zero CAM instead -- this test
    # would then fail.
    assert np.allclose(pre_cam, 0.0)
    assert not np.allclose(post_cam, 0.0)
