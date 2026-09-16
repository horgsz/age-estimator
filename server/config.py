"""Runtime configuration, all overridable via environment variables.

Every value below is a plain module attribute that callers read *at call time*
(e.g. ``preprocessing.compute_crop_box`` reads ``config.CROP_MARGIN`` on each
call, not as a default argument evaluated at import). That matters for
``CROP_MARGIN`` in particular: it must be tunable without editing code, because
the right value is whatever the training pipeline actually used, and we expect
to A/B a couple of candidates once real weights exist.

:func:`reload_from_env` re-reads the environment and reassigns those attributes.
The app calls it during startup, so the process environment -- not whatever was
set when this module happened to be first imported -- decides the settings.

    CROP_MARGIN=0.25 make api

Note the ordering constraint: ``TorchPredictor`` overwrites ``INPUT_SIZE``,
``IMAGENET_MEAN`` and ``IMAGENET_STD`` from the checkpoint's ``meta`` so
normalisation can never drift from training. :func:`reload_from_env` must
therefore run *before* the predictor is constructed, or it would clobber the
checkpoint's own values with the defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent

DEFAULT_MODEL_PATH = str(REPO_ROOT / "checkpoints" / "age_model.pt")

# The margin between the YuNet detection box and the square we feed the model.
#
# Measured, not guessed: YuNet over 300 random UTKFace images (300/300 detected)
# solving m = (200 / max(w_det, h_det) - 1) / 2 for UTKFace's own native framing
# gives a median of 0.0135 (IQR [0.0012, 0.0268]). UTKFace aligned+cropped is
# essentially the raw detector box plus ~1%.
#
# This was 0.4 before it was measured, which framed the face at ~31% of the crop
# area against ~94% in training. Erring wide is the dangerous direction: the
# training augmentation (RandomResizedCrop scale=(0.8, 1.0)) only ever crops in,
# so the model tolerates tighter framing than UTKFace but not wider.
#
# See preprocessing.py -- this is the number to keep in sync with training.
DEFAULT_CROP_MARGIN = 0.0135

# Bounds for a per-request `crop_margin` override on POST /estimate. Negative
# margins crop inside the detector box; the UTKFace measurement had a p05 of
# -0.0149 and a minimum of -0.0368, so mildly negative values are legitimate.
MIN_CROP_MARGIN = -0.5
MAX_CROP_MARGIN = 2.0


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def reload_from_env() -> None:
    """Re-read every setting from the environment.

    Called at application startup. Safe to call again (the tests and the offline
    eval harness do), but see the ordering note in the module docstring.
    """
    global AGE_MODEL_PATH, MODELS_DIR, CROP_MARGIN, INPUT_SIZE
    global IMAGENET_MEAN, IMAGENET_STD, NUM_BINS
    global MAX_UPLOAD_BYTES, ALLOWED_CONTENT_TYPES
    global DETECT_SCORE_THRESHOLD, DETECT_NMS_THRESHOLD, DETECT_TOP_K
    global CORS_ORIGINS

    # Path to the trained checkpoint produced by the `ml/` side of the project.
    # When missing, the server falls back to the deterministic StubPredictor.
    AGE_MODEL_PATH = _env_str("AGE_MODEL_PATH", DEFAULT_MODEL_PATH)

    # Where the YuNet ONNX weights are cached (downloaded on first run).
    MODELS_DIR = Path(_env_str("SERVER_MODELS_DIR", str(SERVER_DIR / "models")))

    # Face crop margin, as a fraction of the square face box side length.
    # MUST match the value used at training time (see preprocessing.crop_face).
    CROP_MARGIN = _env_float("CROP_MARGIN", DEFAULT_CROP_MARGIN)

    # Model input resolution and ImageNet normalisation constants.
    INPUT_SIZE = _env_int("INPUT_SIZE", 224)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # Number of age bins in the DEX-style head (ages 0..100 inclusive).
    NUM_BINS = _env_int("NUM_BINS", 101)

    # Upload guard rails.
    MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)  # 10 MiB
    ALLOWED_CONTENT_TYPES = frozenset(
        {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
    )

    # YuNet detector thresholds.
    DETECT_SCORE_THRESHOLD = _env_float("DETECT_SCORE_THRESHOLD", 0.7)
    DETECT_NMS_THRESHOLD = _env_float("DETECT_NMS_THRESHOLD", 0.3)
    DETECT_TOP_K = _env_int("DETECT_TOP_K", 50)

    # CORS origins for the Vite dev server.
    _default_origins = "http://localhost:5173,http://127.0.0.1:5173"
    CORS_ORIGINS = [
        o.strip() for o in _env_str("CORS_ORIGINS", _default_origins).split(",") if o.strip()
    ]


def describe() -> dict[str, object]:
    """The settings worth echoing in logs, so the active crop is never a guess."""
    return {
        "age_model_path": AGE_MODEL_PATH,
        "crop_margin": CROP_MARGIN,
        "input_size": INPUT_SIZE,
        "num_bins": NUM_BINS,
        "detect_score_threshold": DETECT_SCORE_THRESHOLD,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


# Type annotations for the module-level settings, populated by reload_from_env().
AGE_MODEL_PATH: str
MODELS_DIR: Path
CROP_MARGIN: float
INPUT_SIZE: int
IMAGENET_MEAN: tuple[float, float, float]
IMAGENET_STD: tuple[float, float, float]
NUM_BINS: int
MAX_UPLOAD_BYTES: int
ALLOWED_CONTENT_TYPES: frozenset[str]
DETECT_SCORE_THRESHOLD: float
DETECT_NMS_THRESHOLD: float
DETECT_TOP_K: int
CORS_ORIGINS: list[str]

reload_from_env()
