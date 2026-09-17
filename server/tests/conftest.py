"""Shared pytest fixtures.

The whole suite runs against :class:`StubPredictor`, so no trained checkpoint is
needed. Tests that require real face detection depend on the YuNet fixture,
which skips (rather than fails) if the weights cannot be fetched offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make sure the stub path is taken even if someone has a checkpoint lying around.
os.environ["AGE_MODEL_PATH"] = "/nonexistent/age_model.pt"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402

from server import app as app_module  # noqa: E402
from server.detector import FaceDetector, ensure_yunet_weights  # noqa: E402
from server.predictor import StubPredictor  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
FACE_IMAGE = ASSETS / "face.jpg"


@pytest.fixture(scope="session")
def yunet_detector() -> FaceDetector:
    try:
        ensure_yunet_weights()
    except Exception as exc:  # pragma: no cover - offline environments
        pytest.skip(f"YuNet weights unavailable: {exc}")
    return FaceDetector()


@pytest.fixture
def face_bgr() -> np.ndarray:
    img = cv2.imread(str(FACE_IMAGE), cv2.IMREAD_COLOR)
    assert img is not None, f"missing test asset {FACE_IMAGE}"
    return img


@pytest.fixture
def face_bytes() -> bytes:
    return FACE_IMAGE.read_bytes()


class FakeDetector(FaceDetector):
    """Detector returning a fixed set of boxes, with no model and no network."""

    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect(self, image_bgr):  # noqa: D102
        return list(self.boxes)


@pytest.fixture
def make_client():
    """Build a TestClient wired to a StubPredictor with a chosen detector."""
    created = []

    def _make(detector=None):
        app_module.set_predictor(StubPredictor(detector=detector))
        client = TestClient(app_module.app)
        created.append(client)
        return client

    yield _make

    for client in created:
        client.close()
    app_module.set_predictor(None)


@pytest.fixture
def client(make_client, yunet_detector):
    """TestClient using the stub predictor and the real YuNet detector."""
    return make_client(yunet_detector)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()
