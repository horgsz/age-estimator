"""API-level tests: stub predictor, no-face, malformed uploads, bbox coords."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import FakeDetector, encode_jpeg

from server import config


def post_image(client, data: bytes, filename="frame.jpg", content_type="image/jpeg"):
    return client.post("/estimate", files={"image": (filename, data, content_type)})


# --------------------------------------------------------------------------
# health / stub predictor
# --------------------------------------------------------------------------


def test_health_reports_stub(make_client):
    client = make_client(FakeDetector([]))
    body = client.get("/health").json()
    assert body == {"status": "ok", "model": "stub", "stub": True}


def test_estimate_with_stub_predictor(client, face_bytes):
    resp = post_image(client, face_bytes)
    assert resp.status_code == 200

    faces = resp.json()["faces"]
    assert len(faces) == 1

    face = faces[0]
    assert set(face) == {"bbox", "age", "low", "high", "confidence"}
    assert len(face["bbox"]) == 4
    assert all(isinstance(v, int) for v in face["bbox"])
    assert 0.0 <= face["low"] <= face["age"] <= face["high"] <= 100.0
    assert 0.0 < face["confidence"] <= 1.0


def test_stub_is_deterministic(client, face_bytes):
    first = post_image(client, face_bytes).json()
    second = post_image(client, face_bytes).json()
    assert first == second


def test_stub_gives_different_ages_for_different_faces(make_client, face_bgr):
    # Two boxes over visibly different content -> different hashed ages.
    client = make_client(FakeDetector([(148, 36, 97, 134), (0, 0, 80, 80)]))
    faces = post_image(client, encode_jpeg(face_bgr)).json()["faces"]
    assert len(faces) == 2
    assert faces[0]["age"] != faces[1]["age"]


# --------------------------------------------------------------------------
# zero faces
# --------------------------------------------------------------------------


def test_image_with_no_face_returns_empty_list(client):
    blank = np.full((480, 640, 3), 200, np.uint8)
    cv2.rectangle(blank, (100, 100), (300, 300), (40, 90, 160), -1)

    resp = post_image(client, encode_jpeg(blank))
    assert resp.status_code == 200
    assert resp.json() == {"faces": []}


# --------------------------------------------------------------------------
# malformed uploads
# --------------------------------------------------------------------------


def test_undecodable_bytes_are_rejected(make_client):
    client = make_client(FakeDetector([]))
    resp = post_image(client, b"this is definitely not a JPEG")
    assert resp.status_code == 400
    assert "decode" in resp.json()["detail"].lower()


def test_truncated_jpeg_is_rejected(make_client, face_bytes):
    client = make_client(FakeDetector([]))
    resp = post_image(client, face_bytes[:24])
    assert resp.status_code == 400


def test_empty_upload_is_rejected(make_client):
    client = make_client(FakeDetector([]))
    resp = post_image(client, b"")
    assert resp.status_code == 400


def test_non_image_content_type_is_rejected(make_client, face_bytes):
    client = make_client(FakeDetector([]))
    resp = post_image(client, face_bytes, filename="x.pdf", content_type="application/pdf")
    assert resp.status_code == 415


def test_missing_file_field_is_rejected(make_client, face_bytes):
    client = make_client(FakeDetector([]))
    resp = client.post("/estimate", files={"photo": ("f.jpg", face_bytes, "image/jpeg")})
    assert resp.status_code == 422


def test_oversized_upload_is_rejected(make_client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)
    client = make_client(FakeDetector([]))
    resp = post_image(client, b"\xff\xd8\xff" + b"0" * 4096)
    assert resp.status_code == 413


# --------------------------------------------------------------------------
# bbox coordinate space
# --------------------------------------------------------------------------


def test_bbox_is_passed_through_in_original_coordinates(make_client, face_bgr):
    boxes = [(11, 23, 57, 91), (140, 30, 100, 130)]
    client = make_client(FakeDetector(boxes))

    faces = post_image(client, encode_jpeg(face_bgr)).json()["faces"]
    assert [tuple(f["bbox"]) for f in faces] == boxes


def test_bbox_scales_with_image_resolution(client, face_bgr):
    """The bbox must be in the ORIGINAL upload's pixel space, not a resized one.

    The same photo at 1x and 2x must yield boxes whose *relative* geometry
    matches, and whose absolute pixel values differ by the scale factor.
    """
    h, w = face_bgr.shape[:2]
    big = cv2.resize(face_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    small_faces = post_image(client, encode_jpeg(face_bgr)).json()["faces"]
    big_faces = post_image(client, encode_jpeg(big)).json()["faces"]
    assert len(small_faces) == 1 and len(big_faces) == 1

    sx, sy, sw, sh = small_faces[0]["bbox"]
    bx, by, bw, bh = big_faces[0]["bbox"]

    # Absolute coords roughly double.
    assert bx == pytest.approx(sx * 2, abs=0.06 * w)
    assert by == pytest.approx(sy * 2, abs=0.06 * h)
    assert bw == pytest.approx(sw * 2, abs=0.06 * w)
    assert bh == pytest.approx(sh * 2, abs=0.06 * h)

    # Normalised coords agree.
    assert bx / (2 * w) == pytest.approx(sx / w, abs=0.03)
    assert bw / (2 * w) == pytest.approx(sw / w, abs=0.03)


def test_bbox_stays_inside_the_uploaded_image(client, face_bgr):
    """Covers the detector's internal downscale path for large uploads."""
    big = cv2.resize(face_bgr, (2400, 3000), interpolation=cv2.INTER_CUBIC)
    faces = post_image(client, encode_jpeg(big)).json()["faces"]
    assert faces

    for face in faces:
        x, y, bw, bh = face["bbox"]
        assert 0 <= x < 2400
        assert 0 <= y < 3000
        assert 0 < bw <= 2400 - x
        assert 0 < bh <= 3000 - y

    # And the face should be found in roughly the same relative spot as at 1x.
    ref = post_image(client, encode_jpeg(face_bgr)).json()["faces"][0]["bbox"]
    ref_cx = (ref[0] + ref[2] / 2) / face_bgr.shape[1]
    bx, _, bw, _ = faces[0]["bbox"]
    assert (bx + bw / 2) / 2400 == pytest.approx(ref_cx, abs=0.05)


def test_png_upload_is_accepted(client, face_bgr):
    ok, buf = cv2.imencode(".png", face_bgr)
    assert ok
    resp = post_image(client, buf.tobytes(), filename="f.png", content_type="image/png")
    assert resp.status_code == 200
    assert len(resp.json()["faces"]) == 1
