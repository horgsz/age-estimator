"""Shared paths, constants and crop geometry for the ``datasets/`` corpus build.

Crop geometry
-------------
``server/preprocessing.py`` is the single source of truth for how a face is
cropped. This module imports it when the server package is importable and falls
back to a byte-faithful mirror otherwise, so the corpus build works in a
worktree that only contains ``datasets/``. :func:`crop_backend` reports which
path was taken, and ``verify_crop.py`` asserts the two agree.

Do not "improve" the fallback. If ``server/preprocessing.py`` changes, this
mirror has to change with it, and the crops in ``datasets/crops/`` have to be
regenerated, or training silently sees a different framing than serving.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

DATASETS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATASETS_DIR.parent

RAW_DIR = DATASETS_DIR / "raw"
CROPS_DIR = DATASETS_DIR / "crops"
REPORTS_DIR = DATASETS_DIR / "reports"
MANIFEST_PATH = DATASETS_DIR / "manifest.csv"

# Mirrors ml/data.py so a row that survives here would have survived there.
SEED = 42
INPUT_SIZE = 224
NUM_BINS = 101  # classifier heads cover ages 0..100
MIN_AGE = 0
MAX_AGE = 100

# server/config.py DEFAULT_CROP_MARGIN. Established by two independent measured
# sweeps that agreed within 0.07 years and both minimised at 0.0. Do not change
# without re-running those sweeps.
CROP_MARGIN = 0.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BBox = tuple[int, int, int, int]


def _import_server_preprocessing():
    """Return ``server.preprocessing`` if this checkout has it, else ``None``."""
    if not (REPO_ROOT / "server" / "preprocessing.py").is_file():
        return None
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from server import preprocessing  # type: ignore[import-not-found]
    except Exception:
        return None
    return preprocessing


_SERVER_PREPROCESSING = _import_server_preprocessing()


def crop_backend() -> str:
    """``"server"`` when server/preprocessing.py drove the crop, else ``"mirror"``."""
    return "server" if _SERVER_PREPROCESSING is not None else "mirror"


def _mirror_crop_face(image_bgr: np.ndarray, bbox: BBox, margin: float) -> np.ndarray:
    """Mirror of ``server.preprocessing.crop_face``; see the module docstring."""
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    side = max(1, int(round(max(float(w), float(h)) * (1.0 + 2.0 * float(margin)))))

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))

    # Slide back inside the frame only when the square actually fits; otherwise
    # stay centred on the face and pad the deficit with replicated edge pixels.
    if side <= img_w:
        x0 = min(max(x0, 0), img_w - side)
    if side <= img_h:
        y0 = min(max(y0, 0), img_h - side)

    ax0 = max(0, x0)
    ay0 = max(0, y0)
    ax1 = min(img_w, x0 + side)
    ay1 = min(img_h, y0 + side)

    if ax1 <= ax0:
        ax0 = min(max(ax0, 0), max(0, img_w - 1))
        ax1 = min(img_w, ax0 + 1)
    if ay1 <= ay0:
        ay0 = min(max(ay0, 0), max(0, img_h - 1))
        ay1 = min(img_h, ay0 + 1)

    crop = image_bgr[ay0:ay1, ax0:ax1]
    pad_left, pad_top = ax0 - x0, ay0 - y0
    pad_right, pad_bottom = (x0 + side) - ax1, (y0 + side) - ay1
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    return crop


def crop_face(image_bgr: np.ndarray, bbox: BBox, margin: float | None = None) -> np.ndarray:
    """Square BGR face crop for ``bbox``, at the source image's pixel scale."""
    if margin is None:
        margin = CROP_MARGIN
    if _SERVER_PREPROCESSING is not None:
        return _SERVER_PREPROCESSING.crop_face(image_bgr, bbox, margin)
    return _mirror_crop_face(image_bgr, bbox, margin)


def resize_crop(crop_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize a face crop to the model's square input resolution."""
    if _SERVER_PREPROCESSING is not None:
        return _SERVER_PREPROCESSING.resize_crop(crop_bgr, size)
    interp = cv2.INTER_AREA if crop_bgr.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(crop_bgr, (size, size), interpolation=interp)
