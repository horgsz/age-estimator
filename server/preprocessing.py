"""Face cropping + normalisation.

THIS MODULE IS THE SINGLE SOURCE OF TRUTH FOR HOW A FACE IS CROPPED.

!!! KEEP IN SYNC WITH TRAINING !!!
The crop produced by :func:`crop_face` must match the crop distribution the
model was trained on. If the training pipeline in ``ml/`` changes its cropping
geometry -- margin, squareness, or the resize/normalisation constants -- this
module has to change with it, otherwise inference silently sees
out-of-distribution inputs and the predicted ages drift with no visible error.
The knobs live in ``server/config.py``.

Why ``CROP_MARGIN`` defaults to 0.0
-----------------------------------
Measured end to end, not guessed. Two independent sweeps over the 1,185-image
UTKFace test split -- ``ml/``'s, which re-frames ground-truth crops, and this
server's ``tools/eval_end_to_end.py``, which runs real YuNet detection and then
this crop -- both put the minimum at 0.0 (5.495 and 5.477 MAE respectively), and
agree within 0.07 years at every margin from -0.05 to 0.2. That agreement is the
useful part: it says the detector path does not shift framing, so a constant
measured offline transfers to the deployed path.

UTKFace aligned+cropped is essentially the raw detector box -- far tighter than
the 0.4 we first assumed, which put the face at ~31% of the frame by area
against ~94% in training and cost ~3 years of MAE invisibly.

Erring **wide is the dangerous direction**: the curve is strongly asymmetric.
Cropping tighter than training costs almost nothing (-0.05 is +0.11 years),
cropping wider degrades steeply (0.2 is +0.71, 0.4 is +2.97). Anything in
[-0.05, +0.05] sits within ~0.11 years, which absorbs detector jitter.

Geometry
--------
At a margin this tight the square is exactly the detector box, so the
clamp-to-image-bounds path is hit far more often than it was at 0.4 -- any face
near a frame edge reaches it. Two rules keep that honest:

* If the square **fits** in the image, it is slid back inside rather than
  shrunk. All real pixels, exact scale, at most a pixel or two off-centre.
* If the square **does not fit** (a face filling the frame, which is exactly
  what a 200x200 UTKFace image looks like), it stays centred on the face and the
  deficit is filled with replicated edge pixels.

Either way the result is exactly ``side x side`` at the original pixel scale, so
it is never squashed and the face never changes apparent size. That last part
matters: padding to the *available* region instead of the *requested* square
would quietly zoom the face in whenever the crop ran off an edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config

BBox = tuple[int, int, int, int]  # (x, y, w, h) in original-image pixel coords


@dataclass(frozen=True)
class CropGeometry:
    """Where the square crop lands, split into in-image region + padding.

    ``x, y, w, h`` is the part of the square that actually exists in the image
    (original-image pixel coords). The ``pad_*`` fields are the deficit on each
    side, filled by edge replication. By construction::

        pad_left + w + pad_right == side == pad_top + h + pad_bottom
    """

    x: int
    y: int
    w: int
    h: int
    side: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @property
    def needs_padding(self) -> bool:
        return bool(self.pad_left or self.pad_top or self.pad_right or self.pad_bottom)

    @property
    def box(self) -> BBox:
        return (self.x, self.y, self.w, self.h)


def compute_crop_geometry(
    bbox: BBox, image_shape: tuple[int, int], margin: float | None = None
) -> CropGeometry:
    """Resolve the square crop for ``bbox``; see the module docstring."""
    if margin is None:
        margin = config.CROP_MARGIN

    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    side = max(1, int(round(max(float(w), float(h)) * (1.0 + 2.0 * float(margin)))))

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))

    # Slide back inside the frame only when the square actually fits; otherwise
    # stay centred on the face and let the padding below make up the difference.
    if side <= img_w:
        x0 = min(max(x0, 0), img_w - side)
    if side <= img_h:
        y0 = min(max(y0, 0), img_h - side)

    ax0 = max(0, x0)
    ay0 = max(0, y0)
    ax1 = min(img_w, x0 + side)
    ay1 = min(img_h, y0 + side)

    # Degenerate guard: a box whose centre sits outside the image would give an
    # empty intersection. Detector boxes are clamped in-frame so this should not
    # happen, but an empty slice would crash rather than degrade.
    if ax1 <= ax0:
        ax0 = min(max(ax0, 0), max(0, img_w - 1))
        ax1 = min(img_w, ax0 + 1)
    if ay1 <= ay0:
        ay0 = min(max(ay0, 0), max(0, img_h - 1))
        ay1 = min(img_h, ay0 + 1)

    return CropGeometry(
        x=ax0,
        y=ay0,
        w=ax1 - ax0,
        h=ay1 - ay0,
        side=side,
        pad_left=ax0 - x0,
        pad_top=ay0 - y0,
        pad_right=(x0 + side) - ax1,
        pad_bottom=(y0 + side) - ay1,
    )


def compute_crop_box(bbox: BBox, image_shape: tuple[int, int], margin: float | None = None) -> BBox:
    """The in-image part of the square crop, as ``(x, y, w, h)``.

    This is the region genuinely present in the source image; when the square
    runs off an edge it is smaller than ``side`` and :func:`crop_face` pads the
    remainder. Use :func:`compute_crop_geometry` if you need the padding too.
    """
    return compute_crop_geometry(bbox, image_shape, margin).box


def crop_face(image_bgr: np.ndarray, bbox: BBox, margin: float | None = None) -> np.ndarray:
    """Return the square BGR face crop for ``bbox`` (not yet resized).

    Always exactly ``side x side``, at the source image's pixel scale.
    """
    geom = compute_crop_geometry(bbox, image_bgr.shape[:2], margin)
    crop = image_bgr[geom.y : geom.y + geom.h, geom.x : geom.x + geom.w]

    if geom.needs_padding:
        crop = cv2.copyMakeBorder(
            crop,
            geom.pad_top,
            geom.pad_bottom,
            geom.pad_left,
            geom.pad_right,
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
