"""Unit tests for the shared crop + normalisation pipeline and result decoding."""

from __future__ import annotations

import numpy as np
import pytest

from server import config, preprocessing
from server.predictor import (
    StubPredictor,
    build_result,
    confidence_from_std,
)


def test_crop_box_is_square_and_applies_margin():
    img = np.zeros((1000, 1000, 3), np.uint8)
    x, y, w, h = preprocessing.compute_crop_box((400, 400, 100, 200), img.shape[:2], margin=0.4)

    assert w == h, "crop box must be square"
    # side = max(100, 200) * (1 + 2*0.4) = 360
    assert w == pytest.approx(360, abs=1)
    # centred on the original box centre (450, 500)
    assert x + w / 2 == pytest.approx(450, abs=1)
    assert y + h / 2 == pytest.approx(500, abs=1)


def test_margin_zero_gives_the_tight_square():
    img = np.zeros((500, 500, 3), np.uint8)
    _, _, w, _ = preprocessing.compute_crop_box((100, 100, 60, 90), img.shape[:2], margin=0.0)
    assert w == pytest.approx(90, abs=1)


def test_larger_margin_gives_a_larger_box():
    img = np.zeros((2000, 2000, 3), np.uint8)
    small = preprocessing.compute_crop_box((800, 800, 100, 100), img.shape[:2], margin=0.1)
    large = preprocessing.compute_crop_box((800, 800, 100, 100), img.shape[:2], margin=0.8)
    assert large[2] > small[2]


def test_crop_box_is_clamped_to_image_bounds():
    img = np.zeros((200, 300, 3), np.uint8)
    x, y, w, h = preprocessing.compute_crop_box((0, 0, 150, 150), img.shape[:2], margin=0.4)

    assert x >= 0 and y >= 0
    assert x + w <= 300
    assert y + h <= 200


def test_crop_near_edge_is_shifted_inside_not_squashed():
    img = np.zeros((1000, 1000, 3), np.uint8)
    # Box hugging the left edge; the square should slide right rather than shrink.
    x, y, w, h = preprocessing.compute_crop_box((0, 400, 100, 100), img.shape[:2], margin=0.4)
    assert x == 0
    assert w == h == pytest.approx(180, abs=1)


def test_crop_face_always_returns_a_square_image():
    img = np.random.randint(0, 255, (120, 90, 3), dtype=np.uint8)
    # Box far larger than the frame forces the padding path.
    crop = preprocessing.crop_face(img, (0, 0, 90, 120), margin=1.0)
    assert crop.shape[0] == crop.shape[1]


def test_preprocess_face_shape_and_normalisation():
    img = np.full((400, 400, 3), 127, np.uint8)
    tensor = preprocessing.preprocess_face(img, (100, 100, 120, 120))

    assert tensor.shape == (3, config.INPUT_SIZE, config.INPUT_SIZE)
    assert tensor.dtype == np.float32

    # A flat mid-grey image maps to (0.498 - mean) / std per channel.
    for c in range(3):
        expected = (127 / 255.0 - config.IMAGENET_MEAN[c]) / config.IMAGENET_STD[c]
        assert tensor[c].mean() == pytest.approx(expected, abs=1e-3)


def test_normalize_produces_rgb_channel_order():
    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[..., 2] = 255  # pure red in BGR
    tensor = preprocessing.normalize(bgr)
    assert tensor[0].mean() > tensor[1].mean()
    assert tensor[0].mean() > tensor[2].mean()


# --------------------------------------------------------------------------
# result decoding
# --------------------------------------------------------------------------


def test_confidence_decreases_monotonically_with_std():
    values = [confidence_from_std(s) for s in (0, 1, 3, 6, 12, 40)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(1.0)
    assert 0.0 < values[-1] < 0.2


def test_build_result_derives_interval_from_std():
    r = build_result((1, 2, 3, 4), age=31.4, std=5.2)
    assert r.age == pytest.approx(31.4)
    assert r.low == pytest.approx(26.2)
    assert r.high == pytest.approx(36.6)


def test_build_result_clamps_the_interval():
    r = build_result((0, 0, 10, 10), age=3.0, std=9.0)
    assert r.low == 0.0
    r = build_result((0, 0, 10, 10), age=99.0, std=9.0)
    assert r.high == 100.0


def test_stub_predictor_is_stable_and_in_range():
    stub = StubPredictor()
    rng = np.random.default_rng(0)
    batch = rng.random((4, 3, 224, 224)).astype(np.float32)

    ages_a, stds_a = stub._estimate(batch)
    ages_b, stds_b = stub._estimate(batch)

    assert np.array_equal(ages_a, ages_b)
    assert np.array_equal(stds_a, stds_b)
    assert ((ages_a >= 0) & (ages_a <= 100)).all()
    assert (stds_a > 0).all()
    assert len(set(ages_a.tolist())) == 4


def test_soft_expectation_decoding_matches_the_contract():
    """Mirror of TorchPredictor._estimate, verified on a known distribution."""
    torch = pytest.importorskip("torch")

    logits = torch.full((1, 101), -20.0)
    logits[0, 30] = 0.0
    logits[0, 40] = 0.0  # 50/50 over ages 30 and 40

    probs = torch.softmax(logits, dim=1)
    bins = torch.arange(101, dtype=torch.float32)
    age = (probs * bins).sum(dim=1)
    std = torch.sqrt((probs * (bins.unsqueeze(0) - age.unsqueeze(1)) ** 2).sum(dim=1))

    assert age.item() == pytest.approx(35.0, abs=0.1)
    assert std.item() == pytest.approx(5.0, abs=0.1)

    result = build_result((0, 0, 1, 1), age.item(), std.item())
    assert result.low == pytest.approx(30.0, abs=0.1)
    assert result.high == pytest.approx(40.0, abs=0.1)
