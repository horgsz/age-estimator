"""YuNet face detection (``cv2.FaceDetectorYN``).

The ONNX weights are small (~230 KB) and are downloaded into ``server/models/``
on first use. They are gitignored.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger(__name__)

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

# opencv_zoo stores the ONNX files in git-lfs, so the plain raw.githubusercontent
# URL returns a 131-byte pointer file. The media.githubusercontent.com host
# serves the real LFS object.
_ZOO_DIR = "opencv/opencv_zoo/main/models/face_detection_yunet/"
YUNET_URLS = (
    f"https://media.githubusercontent.com/media/{_ZOO_DIR}{YUNET_FILENAME}",
    f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_FILENAME}",
)

# sha256 of the git-lfs object referenced by opencv_zoo@main.
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# Detection runs on a downscaled copy of large uploads for speed; the resulting
# boxes are scaled back into the ORIGINAL image's coordinate space.
MAX_DETECT_SIDE = 1024

BBox = tuple[int, int, int, int]


def _looks_like_onnx(path: Path) -> bool:
    try:
        if path.stat().st_size < 50_000:
            return False
        with path.open("rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    # Guard against git-lfs pointer files or HTML error pages.
    return not (head.startswith(b"version https://git-lfs") or head.lstrip().startswith(b"<"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``.

    Tries urllib first, then falls back to ``curl``. The fallback matters on
    machines behind a TLS-intercepting proxy, where the corporate root CA lives
    in the system keychain that curl consults but Python's OpenSSL bundle
    does not.
    """
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        return
    except Exception as exc:
        log.debug("urllib download failed for %s (%s); trying curl", url, exc)
        dest.unlink(missing_ok=True)

    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError(f"urllib download failed for {url} and curl is unavailable")
    subprocess.run(
        [curl, "-fsSL", "--retry", "2", "--max-time", "120", "-o", str(dest), url],
        check=True,
        capture_output=True,
    )


def ensure_yunet_weights(models_dir: Path | None = None) -> Path:
    """Return the path to the YuNet ONNX file, downloading it if missing."""
    models_dir = Path(models_dir or config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / YUNET_FILENAME

    if _looks_like_onnx(target):
        return target

    last_error: Exception | None = None
    for url in YUNET_URLS:
        tmp = target.with_suffix(".onnx.part")
        try:
            log.info("Downloading YuNet face detector weights from %s", url)
            _fetch(url, tmp)
            if not _looks_like_onnx(tmp):
                raise RuntimeError(f"downloaded file from {url} does not look like ONNX")
            actual = _sha256(tmp)
            if actual != YUNET_SHA256:
                raise RuntimeError(
                    f"checksum mismatch for {url}: expected {YUNET_SHA256}, got {actual}"
                )
            tmp.replace(target)
            log.info("YuNet weights saved to %s", target)
            return target
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            log.warning("YuNet download failed (%s): %s", url, exc)
            tmp.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not obtain YuNet weights ({YUNET_FILENAME}). Place the file in "
        f"{models_dir} manually. Last error: {last_error}"
    )


class FaceDetector:
    """Thin wrapper around ``cv2.FaceDetectorYN`` with coordinate rescaling."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else ensure_yunet_weights()
        self._net = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=config.DETECT_SCORE_THRESHOLD,
            nms_threshold=config.DETECT_NMS_THRESHOLD,
            top_k=config.DETECT_TOP_K,
        )

    def detect(self, image_bgr: np.ndarray) -> list[BBox]:
        """Detect faces, returning integer bboxes in ORIGINAL image coords."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        img_h, img_w = image_bgr.shape[:2]
        scale = min(1.0, MAX_DETECT_SIDE / float(max(img_h, img_w)))
        if scale < 1.0:
            det_w = max(1, int(round(img_w * scale)))
            det_h = max(1, int(round(img_h * scale)))
            det_img = cv2.resize(image_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_w, det_h = img_w, img_h
            det_img = image_bgr

        self._net.setInputSize((det_w, det_h))
        _, faces = self._net.detect(det_img)
        if faces is None:
            return []

        # Map detection-space boxes back to the original image's pixel grid.
        inv_x = img_w / float(det_w)
        inv_y = img_h / float(det_h)

        boxes: list[BBox] = []
        for face in faces:
            x, y, w, h = (float(v) for v in face[:4])
            bx = int(round(x * inv_x))
            by = int(round(y * inv_y))
            bw = int(round(w * inv_x))
            bh = int(round(h * inv_y))

            # Clamp into the frame; drop degenerate boxes.
            bx = min(max(bx, 0), max(0, img_w - 1))
            by = min(max(by, 0), max(0, img_h - 1))
            bw = min(bw, img_w - bx)
            bh = min(bh, img_h - by)
            if bw >= 2 and bh >= 2:
                boxes.append((bx, by, bw, bh))

        return boxes
