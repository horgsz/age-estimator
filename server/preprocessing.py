"""Face cropping + normalisation.

THIS MODULE IS THE SINGLE SOURCE OF TRUTH FOR HOW A FACE IS CROPPED.

!!! KEEP IN SYNC WITH TRAINING !!!
The crop produced by :func:`crop_face` must match the crop distribution the
model was trained on (UTKFace aligned+cropped images are fairly tight around the
face). If the training pipeline in ``ml/`` changes its cropping geometry --
margin, squareness, or the resize/normalisation constants -- this module has to
change with it, otherwise inference silently sees out-of-distribution inputs and
the predicted ages drift. The knobs live in ``server/config.py``
(``CROP_MARGIN``, ``INPUT_SIZE``, ``IMAGENET_MEAN``, ``IMAGENET_STD``).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config

BBox = tuple[int, int, int, int]  # (x, y, w, h) in original-image pixel coords


def compute_crop_box(bbox: BBox, image_shape: tuple[int, int], margin: float | None = None) -> BBox:
    """Expand a detector bbox to a square and apply ``margin``.

    The square is centred on the detection, its side is
    ``max(w, h) * (1 + 2 * margin)``, and it is then nudged/clamped to sit
    inside the image. If the image is too small to contain the full square the
    returned box is smaller than requested; :func:`crop_face` restores
    squareness with edge padding so the aspect ratio is never distorted.

    Returns integer ``(x, y, w, h)`` in the ORIGINAL image's coordinate space.
    """
    if margin is None:
        margin = config.CROP_MARGIN

    img_h, img_w = image_shape[:2]
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    side = max(float(w), float(h)) * (1.0 + 2.0 * float(margin))
    side = max(side, 1.0)

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    side_i = max(1, int(round(side)))

    # Shift the square back inside the frame before clamping, so a face near an
    # edge keeps its full context on the other side instead of being cropped.
    x0 = min(max(x0, 0), max(0, img_w - side_i))
    y0 = min(max(y0, 0), max(0, img_h - side_i))

    x1 = min(img_w, x0 + side_i)
    y1 = min(img_h, y0 + side_i)
    x0 = max(0, x0)
    y0 = max(0, y0)

    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def crop_face(image_bgr: np.ndarray, bbox: BBox, margin: float | None = None) -> np.ndarray:
    """Return the square BGR face crop for ``bbox`` (not yet resized)."""
    x, y, w, h = compute_crop_box(bbox, image_bgr.shape[:2], margin)
    crop = image_bgr[y : y + h, x : x + w]

    # Restore squareness if the image was too small to hold the full box.
    side = max(crop.shape[0], crop.shape[1])
    if crop.shape[0] != side or crop.shape[1] != side:
        pad_y = side - crop.shape[0]
        pad_x = side - crop.shape[1]
        crop = cv2.copyMakeBorder(
            crop,
            pad_y // 2,
            pad_y - pad_y // 2,
            pad_x // 2,
            pad_x - pad_x // 2,
            cv2.BORDER_REPLICATE,
        )
    return crop


def resize_crop(crop_bgr: np.ndarray, size: int | None = None) -> np.ndarray:
    """Resize a face crop to the model's square input resolution."""
    if size is None:
        size = config.INPUT_SIZE
    interp = cv2.INTER_AREA if crop_bgr.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(crop_bgr, (size, size), interpolation=interp)


def normalize(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC crop -> ImageNet-normalised float32 CHW (RGB) array."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(config.IMAGENET_STD, dtype=np.float32)
    rgb = (rgb - mean) / std
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def preprocess_face(
    image_bgr: np.ndarray, bbox: BBox, margin: float | None = None, size: int | None = None
) -> np.ndarray:
    """Full detector-bbox -> model-ready float32 CHW tensor pipeline."""
    return normalize(resize_crop(crop_face(image_bgr, bbox, margin), size))
