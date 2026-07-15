import cv2
import numpy as np


def crop_and_resize(image: np.ndarray, polygon: list, size: int = 224) -> np.ndarray:
    height, width = image.shape[:2]

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min = min(max(int(min(xs)), 0), width - 1)
    x_max = min(max(int(max(xs)), x_min + 1), width)
    y_min = min(max(int(min(ys)), 0), height - 1)
    y_max = min(max(int(max(ys)), y_min + 1), height)

    crop = image[y_min:y_max, x_min:x_max]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized
