import numpy as np
from utils.cropping import crop_and_resize


def test_crop_and_resize_returns_correct_shape():
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    polygon = [(10.0, 10.0), (10.0, 50.0), (50.0, 50.0), (50.0, 10.0), (10.0, 10.0)]

    crop = crop_and_resize(image, polygon, size=224)

    assert crop.shape == (224, 224, 3)
    assert crop.dtype == np.uint8


def test_crop_and_resize_clips_polygon_to_image_bounds():
    image = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)
    polygon = [(-10.0, -10.0), (-10.0, 200.0), (200.0, 200.0), (200.0, -10.0)]

    crop = crop_and_resize(image, polygon, size=224)

    assert crop.shape == (224, 224, 3)
