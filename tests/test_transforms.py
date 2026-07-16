import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from utils.transforms import (
    RandomDiscreteRotation,
    build_eval_transform,
    build_train_transform,
)


def _make_test_image():
    # Asymmetric pattern so rotation/flip actually changes pixel values.
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[0, 0] = [255, 0, 0]
    return Image.fromarray(arr, mode="RGB")


def test_eval_transform_is_deterministic():
    img = _make_test_image()
    transform = build_eval_transform()

    out1 = transform(img)
    out2 = transform(img)

    assert torch.equal(out1, out2)


def test_eval_transform_applies_imagenet_normalization():
    img = _make_test_image()
    transform = build_eval_transform()

    out = transform(img)

    # A pure ToTensor() output would be in [0, 1]; normalization shifts and
    # scales it, so some values should fall outside that range.
    assert out.min() < 0.0 or out.max() > 1.0


def test_train_transform_output_shape_and_normalization():
    img = _make_test_image()
    transform = build_train_transform()

    out = transform(img)

    assert out.shape == (3, 8, 8)
    assert out.min() < 0.0 or out.max() > 1.0


def test_random_discrete_rotation_only_produces_90_degree_multiples():
    img = _make_test_image()
    rotate = RandomDiscreteRotation()
    reference = {angle: np.array(TF.rotate(img, angle)) for angle in (0, 90, 180, 270)}

    seen_angles = set()
    for _ in range(30):
        rotated = np.array(rotate(img))
        matches = [angle for angle, ref in reference.items() if np.array_equal(rotated, ref)]
        assert matches, "rotation output did not match any 0/90/180/270 reference"
        seen_angles.add(matches[0])

    # With 30 draws from 4 equally-likely angles, we should see more than
    # just one -- this guards against a no-op rotation implementation.
    assert len(seen_angles) > 1
