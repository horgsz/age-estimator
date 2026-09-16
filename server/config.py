"""Runtime configuration, all overridable via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent


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


# Path to the trained checkpoint produced by the `ml/` side of the project.
# When missing, the server falls back to the deterministic StubPredictor.
AGE_MODEL_PATH: str = os.environ.get(
    "AGE_MODEL_PATH", str(REPO_ROOT / "checkpoints" / "age_model.pt")
)

# Where the YuNet ONNX weights are cached (downloaded on first run).
MODELS_DIR: Path = Path(os.environ.get("SERVER_MODELS_DIR", str(SERVER_DIR / "models")))

# Face crop margin, as a fraction of the square face box side length.
# MUST match the value used at training time (see preprocessing.crop_face).
CROP_MARGIN: float = _env_float("CROP_MARGIN", 0.4)

# Model input resolution and ImageNet normalisation constants.
INPUT_SIZE: int = _env_int("INPUT_SIZE", 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Number of age bins in the DEX-style head (ages 0..100 inclusive).
NUM_BINS: int = _env_int("NUM_BINS", 101)

# Upload guard rails.
MAX_UPLOAD_BYTES: int = _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)  # 10 MiB
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    }
)

# YuNet detector thresholds.
DETECT_SCORE_THRESHOLD: float = _env_float("DETECT_SCORE_THRESHOLD", 0.7)
DETECT_NMS_THRESHOLD: float = _env_float("DETECT_NMS_THRESHOLD", 0.3)
DETECT_TOP_K: int = _env_int("DETECT_TOP_K", 50)

# CORS origins for the Vite dev server.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()
]
