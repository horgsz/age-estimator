"""Tests for the offline end-to-end eval harness.

The harness is the check that actually matters once weights exist: ml/eval.py
measures the model on training-framed crops, this measures the deployed system
(YuNet detect -> server crop -> model). The gap between the two is the crop
mismatch, quantified. So the harness itself needs to be trustworthy.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pathlib import Path

from conftest import FakeDetector, encode_jpeg

from server import config
from server.predictor import StubPredictor
from server.tools import eval_end_to_end as harness


@pytest.fixture
def dataset(tmp_path, face_bgr):
    """A tiny on-disk dataset of the real test face under several ages."""
    images = tmp_path / "imgs"
    images.mkdir()
    ages = [25, 31, 44]
    for age in ages:
        (images / f"{age}.jpg").write_bytes(encode_jpeg(face_bgr))
    return tmp_path, images, ages


def write_csv(path, rows, header=None):
    lines = []
    if header:
        lines.append(",".join(header))
    lines.extend(f"{p},{a}" for p, a in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# sample loading
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        ("path", "age"),
        ("filepath", "true_age"),
        ("image", "label"),
        ("Path", "Age"),
        ("  path  ", "  AGE  "),
    ],
)
def test_csv_column_name_variants_are_accepted(dataset, header):
    """We don't control ml/splits/test.csv's exact header, so be tolerant."""
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages], header)
    samples = harness.load_samples_from_csv(csv_path)
    assert [s.age for s in samples] == [float(a) for a in ages]
    assert all(s.path.exists() for s in samples)


def test_headerless_csv_is_accepted(dataset):
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages])
    samples = harness.load_samples_from_csv(csv_path)
    assert len(samples) == len(ages)


def test_csv_paths_resolve_relative_to_the_csv(dataset):
    root, images, ages = dataset
    nested = root / "splits"
    nested.mkdir()
    csv_path = write_csv(nested / "test.csv", [(f"../imgs/{a}.jpg", a) for a in ages], ("path", "age"))
    assert len(harness.load_samples_from_csv(csv_path)) == len(ages)


def test_image_root_override_resolves_paths(dataset):
    root, images, ages = dataset
    elsewhere = root / "far" / "away"
    elsewhere.mkdir(parents=True)
    csv_path = write_csv(elsewhere / "test.csv", [(f"{a}.jpg", a) for a in ages], ("path", "age"))

    with pytest.raises(harness.DataError):
        harness.load_samples_from_csv(csv_path)

    samples = harness.load_samples_from_csv(csv_path, [images])
    assert len(samples) == len(ages)


def test_absolute_csv_paths_are_used_directly(dataset):
    root, images, ages = dataset
    csv_path = write_csv(
        root / "t.csv", [(str(images / f"{a}.jpg"), a) for a in ages], ("path", "age")
    )
    assert len(harness.load_samples_from_csv(csv_path)) == len(ages)


def test_unresolvable_and_malformed_rows_are_skipped_not_fatal(dataset):
    root, _, ages = dataset
    rows = [("imgs/25.jpg", 25), ("imgs/nope.jpg", 30), ("imgs/31.jpg", "not-a-number")]
    csv_path = write_csv(root / "t.csv", rows, ("path", "age"))
    samples = harness.load_samples_from_csv(csv_path)
    assert [s.age for s in samples] == [25.0]


def test_missing_csv_names_the_ml_split(tmp_path):
    with pytest.raises(harness.DataError, match="ml/splits/test.csv"):
        harness.load_samples_from_csv(tmp_path / "absent.csv")


def test_empty_csv_is_reported(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(harness.DataError, match="empty"):
        harness.load_samples_from_csv(path)


def test_csv_without_recognisable_columns_is_reported(dataset):
    root, _, _ = dataset
    csv_path = root / "t.csv"
    csv_path.write_text("alpha,beta\nimgs/25.jpg,25\nimgs/31.jpg,31\n", encoding="utf-8")
    with pytest.raises(harness.DataError, match="expected a path column"):
        harness.load_samples_from_csv(csv_path)


def test_all_rows_unresolvable_is_reported(tmp_path):
    csv_path = write_csv(
        tmp_path / "t.csv", [("nowhere/a.jpg", 20), ("nowhere/b.jpg", 30)], ("path", "age")
    )
    with pytest.raises(harness.DataError, match="No usable rows"):
        harness.load_samples_from_csv(csv_path)


def test_utkface_directory_ages_come_from_filenames(tmp_path, face_bgr):
    for name in ["25_0_2_20170116.jpg", "61_1_0_20170117.jpg", "notanage.jpg", "999_0_0_x.jpg"]:
        (tmp_path / name).write_bytes(encode_jpeg(face_bgr))
    samples = harness.load_samples_from_dir(tmp_path)
    assert sorted(s.age for s in samples) == [25.0, 61.0]


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(harness.DataError, match="not a directory"):
        harness.load_samples_from_dir(tmp_path / "absent")


# --------------------------------------------------------------------------
# box selection
# --------------------------------------------------------------------------


def test_primary_box_is_the_largest():
    boxes = [(0, 0, 10, 10), (5, 5, 40, 50), (1, 1, 30, 30)]
    assert harness.select_primary_box(boxes) == (5, 5, 40, 50)


def test_primary_box_of_nothing_is_none():
    assert harness.select_primary_box([]) is None


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def test_evaluate_uses_the_same_inference_path_as_the_server(dataset):
    """Not a lookalike: the harness must call predict_boxes, like /estimate does."""
    root, images, ages = dataset
    samples = [harness.Sample(images / f"{a}.jpg", float(a)) for a in ages]
    predictor = StubPredictor(detector=FakeDetector([(40, 30, 120, 150)]))

    seen = []
    original = predictor.predict_boxes

    def spy(image, boxes, margin=None):
        seen.append(margin)
        return original(image, boxes, margin)

    predictor.predict_boxes = spy
    report = harness.evaluate(samples, predictor, [0.0135, 0.4])

    # Detection happens once per image and the margins sweep inside it, so the
    # sweep compares crops of identical boxes rather than re-detecting.
    assert seen == [0.0135, 0.4] * len(ages)
    assert report.evaluated == len(ages)
    assert report.detection_rate == 1.0
    assert [row["margin"] for row in report.margins] == [0.0135, 0.4]
    assert all(row["n"] == len(ages) for row in report.margins)


def test_evaluate_counts_undetected_faces_without_scoring_them(tmp_path, face_bgr):
    blank = tmp_path / "blank.jpg"
    blank.write_bytes(encode_jpeg(np.full((240, 320, 3), 128, np.uint8)))
    good = tmp_path / "good.jpg"
    good.write_bytes(encode_jpeg(face_bgr))

    class SometimesDetector(FakeDetector):
        def detect(self, image_bgr):
            return [] if image_bgr.std() < 1.0 else [(40, 30, 120, 150)]

    samples = [harness.Sample(blank, 30.0), harness.Sample(good, 40.0)]
    report = harness.evaluate(samples, StubPredictor(detector=SometimesDetector([])), [0.0135])

    assert report.no_detection == 1
    assert report.evaluated == 1
    assert report.detection_rate == 0.5
    assert report.margins[0]["n"] == 1


def test_evaluate_counts_unreadable_files(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    report = harness.evaluate(
        [harness.Sample(broken, 30.0)],
        StubPredictor(detector=FakeDetector([(0, 0, 10, 10)])),
        [0.0135],
    )
    assert report.read_errors == 1
    assert report.evaluated == 0
    assert report.margins[0]["n"] == 0


def test_evaluate_flags_multi_face_images(tmp_path, face_bgr):
    path = tmp_path / "two.jpg"
    path.write_bytes(encode_jpeg(face_bgr))
    detector = FakeDetector([(10, 10, 40, 40), (100, 20, 80, 90)])
    report = harness.evaluate(
        [harness.Sample(path, 30.0)], StubPredictor(detector=detector), [0.0135]
    )
    assert report.multi_detection == 1
    assert report.evaluated == 1


def test_metrics_are_arithmetically_correct(tmp_path, face_bgr, monkeypatch):
    """Pin MAE/bias/RMSE against a predictor with known outputs."""
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        p.write_bytes(encode_jpeg(face_bgr))
        paths.append(p)
    samples = [harness.Sample(p, age) for p, age in zip(paths, [10.0, 20.0, 30.0])]

    predictor = StubPredictor(detector=FakeDetector([(40, 30, 120, 150)]))
    predicted = iter([12.0, 18.0, 40.0])  # errors +2, -2, +10

    def fake_predict_boxes(image, boxes, margin=None):
        age = next(predicted)
        return [harness_result(boxes[0], age)]

    predictor.predict_boxes = fake_predict_boxes
    report = harness.evaluate(samples, predictor, [0.0135])

    row = report.margins[0]
    assert row["n"] == 3
    assert row["mae"] == pytest.approx((2 + 2 + 10) / 3, abs=1e-3)
    assert row["median_ae"] == pytest.approx(2.0, abs=1e-3)
    assert row["bias"] == pytest.approx((2 - 2 + 10) / 3, abs=1e-3)
    assert row["rmse"] == pytest.approx(((4 + 4 + 100) / 3) ** 0.5, abs=1e-3)
    assert row["within_5"] == pytest.approx(2 / 3, abs=1e-3)
    assert row["within_10"] == pytest.approx(1.0, abs=1e-3)


def harness_result(box, age, std=3.0):
    from server.predictor import build_result

    return build_result(box, age, std)


def test_save_crops_writes_one_directory_per_margin(dataset, tmp_path):
    root, images, ages = dataset
    samples = [harness.Sample(images / f"{a}.jpg", float(a)) for a in ages]
    out = tmp_path / "crops"
    harness.evaluate(
        samples,
        StubPredictor(detector=FakeDetector([(40, 30, 120, 150)])),
        [0.0135, 0.4],
        save_crops=out,
    )

    dirs = sorted(p.name for p in out.iterdir())
    assert dirs == ["margin_0.0135", "margin_0.4000"]

    tight = sorted((out / "margin_0.0135").glob("*.jpg"))
    wide = sorted((out / "margin_0.4000").glob("*.jpg"))
    assert tight and len(tight) == len(wide)

    # Both are model-sized, and the two margins really do produce different
    # framing — this is the eyeball check the sweep relies on.
    a = cv2.imread(str(tight[0]))
    b = cv2.imread(str(wide[0]))
    assert a.shape == b.shape == (config.INPUT_SIZE, config.INPUT_SIZE, 3)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 5


# --------------------------------------------------------------------------
# reporting + cli
# --------------------------------------------------------------------------


def test_report_shows_full_margin_precision(dataset, capsys):
    """A margin like 0.0135 must not be rendered as '0.01' — that hides the point.

    The margin is passed explicitly rather than relying on the default, so this
    keeps testing precision even after the default changes.
    """
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages], ("path", "age"))

    assert harness.main(["--csv", str(csv_path), "--margins", "0.0135", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "0.0135" in out
    assert "STUB" in out, "a stub run must be loudly labelled as meaningless"


def test_cli_sweeps_margins_and_writes_json(dataset, tmp_path, capsys):
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages], ("path", "age"))
    out_json = tmp_path / "nested" / "report.json"

    code = harness.main(
        ["--csv", str(csv_path), "--margins", "0.0135", "0.4", "--json", str(out_json), "--quiet"]
    )
    assert code == 0

    import json

    data = json.loads(out_json.read_text())
    assert [row["margin"] for row in data["margins"]] == [0.0135, 0.4]
    assert data["evaluated"] == len(ages)
    assert data["stub"] is True

    out = capsys.readouterr().out
    assert "best margin by MAE" in out
    assert "spread across swept margins" in out


def test_cli_limit_caps_the_sample_count(dataset, tmp_path, capsys):
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages], ("path", "age"))
    out_json = tmp_path / "r.json"
    harness.main(["--csv", str(csv_path), "--limit", "2", "--json", str(out_json), "--quiet"])

    import json

    assert json.loads(out_json.read_text())["total_samples"] == 2


def test_cli_defaults_to_the_active_crop_margin(dataset, tmp_path, monkeypatch):
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages], ("path", "age"))
    out_json = tmp_path / "r.json"

    monkeypatch.setenv("CROP_MARGIN", "0.22")
    try:
        harness.main(["--csv", str(csv_path), "--json", str(out_json), "--quiet"])
        import json

        assert json.loads(out_json.read_text())["margins"][0]["margin"] == pytest.approx(0.22)
    finally:
        monkeypatch.delenv("CROP_MARGIN", raising=False)
        config.reload_from_env()


def test_cli_allows_mildly_negative_margins(dataset, tmp_path):
    """UTKFace's implied margins went negative at p05; don't refuse to measure it."""
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages], ("path", "age"))
    out_json = tmp_path / "r.json"
    assert (
        harness.main(
            ["--csv", str(csv_path), "--margins", "-0.03", "--json", str(out_json), "--quiet"]
        )
        == 0
    )
    import json

    assert json.loads(out_json.read_text())["margins"][0]["margin"] == pytest.approx(-0.03)


@pytest.mark.parametrize("margin", ["-0.9", "3.0"])
def test_cli_rejects_out_of_range_margins(dataset, margin, capsys):
    root, _, ages = dataset
    csv_path = write_csv(root / "t.csv", [(f"imgs/{a}.jpg", a) for a in ages], ("path", "age"))
    assert harness.main(["--csv", str(csv_path), "--margins", margin, "--quiet"]) == 2
    assert "margins must be within" in capsys.readouterr().err


def test_cli_reports_a_missing_dataset_without_traceback(tmp_path, capsys):
    assert harness.main(["--csv", str(tmp_path / "absent.csv"), "--quiet"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_requires_a_data_source():
    with pytest.raises(SystemExit):
        harness.main(["--quiet"])


# ---------------------------------------------------------------------------
# subsampling
# ---------------------------------------------------------------------------


def _samples(ages: list[int]) -> list[harness.Sample]:
    return [harness.Sample(path=Path(f"{a}.jpg"), age=float(a)) for a in ages]


def test_subsample_returns_everything_when_limit_is_not_binding():
    samples = _samples(list(range(10)))
    assert harness.subsample(samples, 10) == samples
    assert harness.subsample(samples, 99) == samples
    assert harness.subsample(samples, 0) == samples


def test_subsample_spreads_across_the_set_rather_than_taking_a_head():
    """A head slice of a path-sorted split is age-skewed; an even stride is not.

    ml/splits/test.csv is sorted by path and UTKFace paths start with the age,
    so the first rows are overwhelmingly young. Sampling must preserve the age
    distribution or a margin sweep measures child bias instead of framing.
    """
    samples = _samples(list(range(100)))
    picked = harness.subsample(samples, 10)

    assert len(picked) == 10
    assert picked[0].age == 0
    assert picked[-1].age > 80, "an even stride must reach the tail of the set"

    mean = sum(s.age for s in picked) / len(picked)
    assert abs(mean - 49.5) < 10, "subsample mean should track the full-set mean"


def test_subsample_head_is_available_but_opt_in():
    samples = _samples(list(range(100)))
    picked = harness.subsample(samples, 10, head=True)
    assert [s.age for s in picked] == [float(a) for a in range(10)]


def test_limit_flag_defaults_to_even_sampling():
    args = harness.build_parser().parse_args(["--dir", ".", "--limit", "5"])
    assert args.limit == 5
    assert args.limit_head is False
