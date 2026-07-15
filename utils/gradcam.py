import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = []
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        output.retain_grad()
        self.activations.append(output)

    def generate(self, pre_img: torch.Tensor, post_img: torch.Tensor, target_class: int, branch: str = "post") -> np.ndarray:
        self.activations.clear()
        self.model.zero_grad()

        logits = self.model(pre_img, post_img)
        score = logits[0, target_class]

        # The shared backbone is called twice in one forward pass (pre, then
        # post), so the forward hook fires twice. Forward-call order IS
        # deterministic (SiameseDamageNet.forward calls pre before post,
        # sequentially) so index 0 is always pre's activation, index 1 always
        # post's. We deliberately do NOT use a backward hook to pair up
        # gradients: backward-hook firing order for two independent branches
        # that merge at a later `cat()` is not guaranteed to mirror forward
        # order, which risks silently swapping pre/post attribution. Instead,
        # `retain_grad()` lets us read `.grad` directly off the exact
        # activation tensor we already identified by forward order, which is
        # unambiguous regardless of backward traversal order.
        act_idx = 0 if branch == "pre" else 1
        activation = self.activations[act_idx]

        score.backward()
        gradient = activation.grad[0]
        activation_values = activation.detach()[0]

        weights = gradient.mean(dim=(1, 2))
        cam = torch.einsum("c,chw->hw", weights, activation_values)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def resize_cam(cam: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(cam, (size, size))
