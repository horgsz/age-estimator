"""Unit tests for the shared crop + normalisation pipeline and result decoding."""

from __future__ import annotations

import numpy as np
import pytest

from server import config, preprocessing
from server.predictor import (
    StubPredictor,
    build_result,
    confidence_from_std,
    confidence_from_width,
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


# --------------------------------------------------------------------------
# the default margin
# --------------------------------------------------------------------------


def test_default_crop_margin_matches_the_measured_optimum():
    """0.0 is the joint minimum of two independent end-to-end sweeps.

    Both ml/'s re-framed-crop sweep (5.495) and this server's YuNet -> crop ->
    model sweep (5.477) bottom out at 0.0 on the 1,185-image test split.

    This is deliberately a hard-coded expectation: if someone changes the
    default, it should be a conscious decision backed by a new measurement,
    not a drive-by edit.
    """
    assert config.DEFAULT_CROP_MARGIN == pytest.approx(0.0)
    assert config.CROP_MARGIN == pytest.approx(0.0)


def test_crop_margin_is_read_from_the_environment_at_startup(monkeypatch):
    """`CROP_MARGIN=... make api` must work with no code edit."""
    monkeypatch.setenv("CROP_MARGIN", "0.25")
    try:
        config.reload_from_env()
        assert config.CROP_MARGIN == pytest.approx(0.25)

        # ...and it must be honoured by callers, not captured at import time.
        img = np.zeros((1000, 1000, 3), np.uint8)
        _, _, w, _ = preprocessing.compute_crop_box((400, 400, 100, 100), img.shape[:2])
        assert w == pytest.approx(150, abs=1)  # 100 * (1 + 2*0.25)
    finally:
        monkeypatch.delenv("CROP_MARGIN", raising=False)
        config.reload_from_env()

    assert config.CROP_MARGIN == pytest.approx(config.DEFAULT_CROP_MARGIN)


def test_invalid_crop_margin_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("CROP_MARGIN", "not-a-number")
    try:
        config.reload_from_env()
        assert config.CROP_MARGIN == pytest.approx(config.DEFAULT_CROP_MARGIN)
    finally:
        monkeypatch.delenv("CROP_MARGIN", raising=False)
        config.reload_from_env()


def test_default_margin_crop_is_the_tight_detector_square():
    img = np.zeros((1000, 1000, 3), np.uint8)
    _, _, w, _ = preprocessing.compute_crop_box((400, 400, 200, 200), img.shape[:2])
    assert w == pytest.approx(200 * (1 + 2 * config.DEFAULT_CROP_MARGIN), abs=1)


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
# faces flush against the frame edge
#
# At the measured margin (0.0) the square is exactly the detector box, so any
# face near an edge lands on the clamp path -- far more often than it did at
# 0.4. These tests pin the two properties that matter there: the crop stays
# square at the requested size, and the face keeps its apparent scale.
# --------------------------------------------------------------------------

FACE_COLOR = (200, 180, 160)


def scene_with_face(img_w: int, img_h: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Dark frame with a uniquely-coloured rectangle standing in for a face."""
    img = np.full((img_h, img_w, 3), 30, np.uint8)
    x, y, w, h = box
    img[y : y + h, x : x + w] = FACE_COLOR
    return img


def count_face_pixels(image: np.ndarray) -> int:
    return int(np.all(image == np.asarray(FACE_COLOR, np.uint8), axis=-1).sum())


EDGE_CASES = {
    "flush-left": (0, 200, 120, 160),
    "flush-right": (640 - 120, 200, 120, 160),
    "flush-top": (260, 0, 120, 160),
    "flush-bottom": (260, 480 - 160, 120, 160),
    "corner-top-left": (0, 0, 120, 160),
    "corner-bottom-right": (640 - 120, 480 - 160, 120, 160),
    "centred": (260, 160, 120, 160),
}


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_edge_flush_face_crop_is_square_and_unscaled(name):
    box = EDGE_CASES[name]
    img = scene_with_face(640, 480, box)

    geom = preprocessing.compute_crop_geometry(box, img.shape[:2])
    crop = preprocessing.crop_face(img, box)

    expected_side = round(max(box[2], box[3]) * (1 + 2 * config.CROP_MARGIN))
    assert geom.side == pytest.approx(expected_side, abs=1)

    # Square, at exactly the requested size -- never squashed, never shrunk.
    assert crop.shape[0] == crop.shape[1] == geom.side

    # The square fits in a 640x480 frame, so no padding should be needed at all.
    assert not geom.needs_padding
    assert geom.w == geom.h == geom.side

    # Stays inside the image...
    assert 0 <= geom.x and geom.x + geom.w <= 640
    assert 0 <= geom.y and geom.y + geom.h <= 480

    # ...and still contains the whole detector box.
    assert geom.x <= box[0] and geom.x + geom.w >= box[0] + box[2]
    assert geom.y <= box[1] and geom.y + geom.h >= box[1] + box[3]

    # Scale is untouched: every face pixel survives, none are duplicated.
    assert count_face_pixels(crop) == box[2] * box[3]


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_edge_flush_face_survives_the_full_pipeline(name):
    box = EDGE_CASES[name]
    img = scene_with_face(640, 480, box)
    tensor = preprocessing.preprocess_face(img, box)
    assert tensor.shape == (3, config.INPUT_SIZE, config.INPUT_SIZE)
    assert np.isfinite(tensor).all()


def test_face_filling_the_whole_frame_is_padded_not_zoomed():
    """A 200x200 UTKFace-style image where the detection fills the frame.

    The margin is explicit rather than the default so this keeps exercising the
    pad path regardless of what the default becomes: at 0.1 the square is 240 px
    and cannot fit, so the deficit is edge-padded. The face must still end up at
    200/240 of the crop -- padding to the *available* region instead of the
    *requested* square would silently zoom the face to fill 100% of it, which is
    a framing change the model would see.
    """
    box = (0, 0, 200, 200)
    img = scene_with_face(200, 200, box)
    margin = 0.1

    geom = preprocessing.compute_crop_geometry(box, img.shape[:2], margin=margin)
    crop = preprocessing.crop_face(img, box, margin=margin)

    assert geom.needs_padding
    assert crop.shape[0] == crop.shape[1] == geom.side
    assert geom.side == pytest.approx(round(200 * (1 + 2 * margin)), abs=1)
    assert 200 / geom.side == pytest.approx(0.833, abs=0.01)

    # Padding is symmetric, so the face stays centred.
    assert abs(geom.pad_left - geom.pad_right) <= 1
    assert abs(geom.pad_top - geom.pad_bottom) <= 1


def test_default_margin_needs_no_padding_when_the_box_fills_the_frame():
    """At the default 0.0 the square *is* the detector box, so it always fits.

    Worth pinning: the whole reason the clamp/pad path got so much attention is
    that it was constantly live at a wide margin. At 0.0 a box that exactly
    fills the frame should take the plain path with no padding at all.
    """
    box = (0, 0, 200, 200)
    img = scene_with_face(200, 200, box)

    geom = preprocessing.compute_crop_geometry(box, img.shape[:2])
    assert geom.side == 200
    assert not geom.needs_padding


def test_oversized_square_keeps_scale_at_a_wide_margin():
    box = (0, 0, 200, 200)
    img = scene_with_face(200, 200, box)

    geom = preprocessing.compute_crop_geometry(box, img.shape[:2], margin=0.2)
    crop = preprocessing.crop_face(img, box, margin=0.2)

    assert geom.side == 280
    assert crop.shape[:2] == (280, 280)
    # The face occupies 200/280 of the crop, not all of it.
    assert 200 / crop.shape[0] == pytest.approx(0.714, abs=0.01)


def test_geometry_padding_always_sums_to_the_requested_side():
    rng = np.random.default_rng(1234)
    for _ in range(300):
        img_w = int(rng.integers(40, 800))
        img_h = int(rng.integers(40, 800))
        w = int(rng.integers(8, img_w + 1))
        h = int(rng.integers(8, img_h + 1))
        x = int(rng.integers(0, img_w - w + 1))
        y = int(rng.integers(0, img_h - h + 1))
        margin = float(rng.uniform(-0.2, 1.0))

        geom = preprocessing.compute_crop_geometry((x, y, w, h), (img_h, img_w), margin)
        assert geom.pad_left + geom.w + geom.pad_right == geom.side
        assert geom.pad_top + geom.h + geom.pad_bottom == geom.side
        assert geom.pad_left >= 0 and geom.pad_right >= 0
        assert geom.pad_top >= 0 and geom.pad_bottom >= 0
        assert 0 <= geom.x and geom.x + geom.w <= img_w
        assert 0 <= geom.y and geom.y + geom.h <= img_h

        crop = preprocessing.crop_face(np.zeros((img_h, img_w, 3), np.uint8), (x, y, w, h), margin)
        assert crop.shape[0] == crop.shape[1] == geom.side


def test_negative_margin_crops_inside_the_detector_box():
    """The UTKFace measurement had a p05 of -0.0149, so this must not explode."""
    img = np.zeros((500, 500, 3), np.uint8)
    _, _, w, _ = preprocessing.compute_crop_box((200, 200, 100, 100), img.shape[:2], margin=-0.1)
    assert w == pytest.approx(80, abs=1)


# --------------------------------------------------------------------------
# result decoding
# --------------------------------------------------------------------------


def test_confidence_decreases_monotonically_with_std():
    values = [confidence_from_std(s) for s in (0, 1, 3, 6, 12, 40)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(1.0)
    assert 0.0 < values[-1] < 0.2


def test_build_result_derives_interval_from_quantiles():
    r = build_result((1, 2, 3, 4), age=31.0, low=26.0, high=37.0)
    assert r.age == pytest.approx(31.0)
    assert r.low == pytest.approx(26.0)
    assert r.high == pytest.approx(37.0)


def test_build_result_allows_an_asymmetric_interval():
    """Quantiles need not straddle the median evenly -- that is the point."""
    r = build_result((0, 0, 1, 1), age=70.0, low=68.0, high=85.0)
    assert r.low == 68.0 and r.high == 85.0
    assert r.confidence == confidence_from_width(17.0)


def test_build_result_clamps_the_interval():
    r = build_result((0, 0, 10, 10), age=3.0, low=-6.0, high=12.0)
    assert r.low == 0.0
    r = build_result((0, 0, 10, 10), age=99.0, low=90.0, high=108.0)
    assert r.high == 100.0


def test_build_result_keeps_the_range_around_the_point_estimate():
    """A degenerate interval must never exclude the number being shown."""
    r = build_result((0, 0, 1, 1), age=40.0, low=45.0, high=38.0)
    assert r.low <= r.age <= r.high


def test_stub_predictor_is_stable_and_in_range():
    stub = StubPredictor()
    rng = np.random.default_rng(0)
    batch = rng.random((4, 3, 224, 224)).astype(np.float32)

    d_a = stub._estimate(batch)
    d_b = stub._estimate(batch)

    assert np.array_equal(d_a.age, d_b.age)
    assert np.array_equal(d_a.std, d_b.std)
    assert ((d_a.age >= 0) & (d_a.age <= 100)).all()
    assert (d_a.std > 0).all()
    assert (d_a.low <= d_a.age).all() and (d_a.age <= d_a.high).all()
    assert len(set(d_a.age.tolist())) == 4


def test_median_decode_ignores_a_label_smoothing_pedestal():
    """The reason the decode is the median and not the mean.

    Training used ``label_smoothing=0.1``, which trains a uniform pedestal
    across all 101 bins. That pedestal's own expectation is exactly 50, so an
    expectation decode returns ``0.9 * age + 5`` -- it drags every estimate
    toward the middle of the range, which is what made a real 85-year-old read
    as ~68 and an infant read as ~6. The median is robust to it.

    This is pinned as a test because it is the whole justification for the
    decode: if someone "simplifies" it back to an expectation, the child and
    elderly estimates silently regress by several years.
    """
    torch = pytest.importorskip("torch")
    from server.predictor import LOW_Q, HIGH_Q

    true_age = 8
    probs = torch.full((1, 101), 0.1 / 101)
    probs[0, true_age] += 0.9
    probs = probs / probs.sum()

    bins = torch.arange(101, dtype=torch.float32)
    expectation = (probs * bins).sum(dim=1).item()
    cdf = probs.cumsum(dim=1)
    median = int((cdf < 0.5).sum(dim=1).item())

    # The pedestal pulls the mean 4+ years off a confidently-predicted child.
    assert expectation == pytest.approx(0.9 * true_age + 5.0, abs=0.3)
    assert expectation - true_age > 4.0
    # The median lands on the actual mode.
    assert median == true_age


def test_quantile_decode_reads_straight_off_the_cdf():
    """Mirror of TorchPredictor._estimate on a known bimodal distribution."""
    torch = pytest.importorskip("torch")
    from server.predictor import LOW_Q, HIGH_Q

    logits = torch.full((1, 101), -30.0)
    logits[0, 30] = 0.0
    logits[0, 40] = 0.0  # 50/50 over ages 30 and 40

    probs = torch.softmax(logits, dim=1)
    cdf = probs.cumsum(dim=1)

    def q(level):
        return int((cdf < level).sum(dim=1).item())

    # cdf[30] == 0.5 exactly, so the first bin reaching 0.5 is 30.
    assert q(0.5) == 30
    assert q(LOW_Q) == 30
    assert q(HIGH_Q) == 40

    result = build_result((0, 0, 1, 1), q(0.5), q(LOW_Q), q(HIGH_Q))
    assert result.age == 30.0
    assert result.low == 30.0
    assert result.high == 40.0


def test_confidence_is_monotonic_in_interval_width():
    widths = [0, 2, 6, 12, 24, 80]
    values = [confidence_from_width(w) for w in widths]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(1.0)
    assert 0.0 < values[-1] < 0.2


def test_confidence_from_std_matches_the_equivalent_width():
    """The shim must preserve the old scale, so UI thresholds still hold."""
    assert confidence_from_std(5.0) == confidence_from_width(10.0)
