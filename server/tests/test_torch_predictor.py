"""Verifies the pinned checkpoint contract loads and decodes correctly.

Uses a randomly-initialised network saved in exactly the format the ``ml/`` side
promises, so the swap from stub to real weights is exercised without needing a
trained model.
"""

from __future__ import annotations

import logging

import pytest

from conftest import FakeDetector

from server import config
from server.predictor import (
    MEASURED_DIGEST,
    StubPredictor,
    TorchPredictor,
    load_predictor,
)

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

BACKBONE = "mobilenetv3_small_100"


def _write_checkpoint(directory, *, decode=None, name="age_model.pt"):
    """Save a contract-shaped checkpoint, optionally declaring a decode."""
    directory.mkdir(parents=True, exist_ok=True)
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=config.NUM_BINS)
    meta = {
        "backbone": BACKBONE,
        "num_bins": 101,
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "test_mae": 5.43,
    }
    if decode is not None:
        meta["decode"] = decode
    path = directory / name
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    return path


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _write_checkpoint(tmp_path_factory.mktemp("ckpt"))


def test_torch_predictor_loads_the_contract_checkpoint(checkpoint, face_bgr):
    predictor = TorchPredictor(str(checkpoint), detector=FakeDetector([(148, 36, 97, 134)]))

    assert predictor.model_name == BACKBONE
    assert predictor.is_stub is False
    assert predictor.num_bins == 101
    assert predictor.meta["test_mae"] == pytest.approx(5.43)

    results = predictor.predict(face_bgr)
    assert len(results) == 1

    face = results[0]
    assert face.bbox == (148, 36, 97, 134)
    assert 0.0 <= face.low <= face.age <= face.high <= 100.0
    assert 0.0 < face.confidence <= 1.0


def test_load_predictor_prefers_a_real_checkpoint(checkpoint):
    assert isinstance(load_predictor(str(checkpoint)), TorchPredictor)


def test_load_predictor_falls_back_to_stub_when_missing(tmp_path):
    predictor = load_predictor(str(tmp_path / "does_not_exist.pt"))
    assert isinstance(predictor, StubPredictor)
    assert predictor.is_stub is True
    assert predictor.model_name == "stub"


def test_load_predictor_falls_back_to_stub_on_a_corrupt_checkpoint(tmp_path):
    bad = tmp_path / "age_model.pt"
    bad.write_bytes(b"definitely not a torch checkpoint")
    assert isinstance(load_predictor(str(bad)), StubPredictor)


def test_load_predictor_falls_back_on_a_wrong_shaped_checkpoint(tmp_path):
    bad = tmp_path / "age_model.pt"
    torch.save({"weights": {}}, bad)
    assert isinstance(load_predictor(str(bad)), StubPredictor)


def test_torch_predictor_accepts_a_wrapper_prefixed_state_dict(checkpoint, face_bgr, tmp_path):
    """The delivered checkpoint prefixes every key with ``backbone.``.

    Training wrapped the timm model (``self.backbone = timm.create_model(...)``)
    so the saved keys do not match a bare timm model, even though the tensors
    are identical. Loading must succeed rather than falling back to the stub.
    """
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    wrapped = tmp_path / "wrapped.pt"
    torch.save(
        {
            "state_dict": {f"backbone.{k}": v for k, v in ckpt["state_dict"].items()},
            "meta": ckpt["meta"],
        },
        wrapped,
    )

    predictor = TorchPredictor(str(wrapped), detector=FakeDetector([(148, 36, 97, 134)]))
    assert predictor.is_stub is False

    # Same weights under a different key spelling must give an identical answer.
    plain = TorchPredictor(str(checkpoint), detector=FakeDetector([(148, 36, 97, 134)]))
    assert predictor.predict(face_bgr)[0].age == pytest.approx(plain.predict(face_bgr)[0].age)


def test_wrapper_prefix_is_not_stripped_when_keys_would_not_line_up(tmp_path):
    """Only strip a prefix when it genuinely unwraps the model.

    Stripping unconditionally could load unrelated tensors that happen to share
    a shape, which is far worse than refusing — so a checkpoint whose stripped
    keys still do not match must fall back to the stub, not load silently.
    """
    bad = tmp_path / "age_model.pt"
    torch.save({"state_dict": {"backbone.not_a_real_layer.weight": torch.zeros(3)}}, bad)
    assert isinstance(load_predictor(str(bad)), StubPredictor)


def test_health_style_checkpoint_identity(checkpoint):
    """The artifact must be identifiable at a glance.

    It was republished to the same path mid-evaluation once, silently changing
    the model under an in-flight comparison, so a content hash and the claimed
    test MAE are surfaced rather than inferred from the numbers.
    """
    predictor = TorchPredictor(str(checkpoint), detector=FakeDetector([]))
    info = predictor.describe_checkpoint()

    assert info is not None
    assert info["recorded_test_mae"] == pytest.approx(5.43)
    assert len(info["sha256"]) == 12
    assert info["bytes"] > 0
    assert info["path"] == str(checkpoint)


def test_checkpoint_digest_tracks_file_contents(checkpoint, tmp_path):
    """Two different artifacts must not report the same identity."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    other = tmp_path / "other.pt"
    meta = dict(ckpt["meta"])
    meta["test_mae"] = 5.55
    torch.save({"state_dict": ckpt["state_dict"], "meta": meta}, other)

    a = TorchPredictor(str(checkpoint), detector=FakeDetector([])).describe_checkpoint()
    b = TorchPredictor(str(other), detector=FakeDetector([])).describe_checkpoint()

    assert a["sha256"] != b["sha256"]
    assert a["recorded_test_mae"] != b["recorded_test_mae"]


def test_stub_reports_no_checkpoint():
    assert StubPredictor().describe_checkpoint() is None


def test_wrapper_prefix_strip_is_logged_loudly(checkpoint, tmp_path, caplog):
    """A live contract deviation must be visible in the startup log."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    wrapped = tmp_path / "wrapped.pt"
    torch.save(
        {
            "state_dict": {f"backbone.{k}": v for k, v in ckpt["state_dict"].items()},
            "meta": ckpt["meta"],
        },
        wrapped,
    )

    with caplog.at_level("WARNING", logger="server.predictor"):
        TorchPredictor(str(wrapped), detector=FakeDetector([]))

    warnings = "\n".join(r.getMessage() for r in caplog.records)
    assert "CHECKPOINT CONTRACT DEVIATION" in warnings
    assert "backbone." in warnings


def test_no_deviation_warning_for_a_contract_shaped_checkpoint(checkpoint, caplog):
    """The banner must not cry wolf when the checkpoint is correct."""
    with caplog.at_level("WARNING", logger="server.predictor"):
        TorchPredictor(str(checkpoint), detector=FakeDetector([]))

    warnings = "\n".join(r.getMessage() for r in caplog.records)
    assert "CONTRACT DEVIATION" not in warnings


def test_health_does_not_present_in_corpus_mae_as_real_accuracy(checkpoint):
    """UTKFace labels are DEX estimates, so in-corpus MAE is not accuracy.

    Both figures are published so the distinction travels with the number. The
    real-age figure must be the larger of the two -- if a future change makes
    the in-corpus number look like the headline, a caller would understate the
    error a user actually experiences by ~4 years.
    """
    from server.predictor import MEASURED_ACCURACY

    assert MEASURED_ACCURACY["real_age_mae_appa_real"] == pytest.approx(8.52)
    assert MEASURED_ACCURACY["in_corpus_mae_utkface"] == pytest.approx(4.762)
    assert (
        MEASURED_ACCURACY["real_age_mae_appa_real"]
        > MEASURED_ACCURACY["in_corpus_mae_utkface"]
    )
    assert "DEX" in MEASURED_ACCURACY["accuracy_note"]

    # The recorded checkpoint figure is in-corpus too, and must never be the
    # only MAE on offer.
    info = TorchPredictor(str(checkpoint), detector=FakeDetector([])).describe_checkpoint()
    assert "recorded_test_mae" in info
    assert "accuracy_note" in info


def test_accuracy_figures_are_withheld_for_an_unmeasured_checkpoint(checkpoint):
    """Accuracy belongs to a set of weights, not to "the model".

    Our 4.762/8.52 figures were measured on one specific artifact. Sibling
    checkpoints exist (a real-ground-truth retrain, and no-smoothing variants),
    and pointing AGE_MODEL_PATH at one to evaluate it must not make /health
    report another model's accuracy as though it were measured. Fails closed.
    """
    predictor = TorchPredictor(str(checkpoint), detector=FakeDetector([]))
    info = predictor.describe_checkpoint()

    assert info["sha256"] != MEASURED_DIGEST
    assert info["in_corpus_mae_utkface"] is None
    assert info["real_age_mae_appa_real"] is None
    assert "Unmeasured" in info["accuracy_note"]
    # The identity of the artifact we *did* measure stays discoverable, so the
    # mismatch can be diagnosed rather than merely observed.
    assert MEASURED_DIGEST in info["accuracy_note"]


def test_declared_decode_is_honoured_not_silently_ignored(tmp_path, caplog):
    """A checkpoint's meta["decode"] selects the point estimate.

    The right decode is a property of training, not a fixed choice: label
    smoothing creates a uniform pedestal that drags an expectation decode toward
    the middle, so median wins by ~0.2-0.4 years. Trained without smoothing the
    sign flips and expectation wins by ~0.12. Hardcoding either one leaves
    accuracy on the table the next time training changes.
    """
    path = _write_checkpoint(tmp_path, decode="expectation")
    with caplog.at_level(logging.WARNING):
        predictor = TorchPredictor(str(path), detector=FakeDetector([]))

    assert predictor.decode == "expectation"
    assert predictor.describe_checkpoint()["serving_decode"] == "expectation"
    # Silently changing how every age is computed is exactly the kind of switch
    # that must appear in the log.
    assert any("decode" in r.getMessage() for r in caplog.records)


def test_absent_decode_key_defaults_to_median(checkpoint):
    """The shipped artifact predates meta["decode"]; it must keep its decode."""
    predictor = TorchPredictor(str(checkpoint), detector=FakeDetector([]))
    assert predictor.decode == "median"


def test_unsupported_decode_falls_back_loudly(tmp_path, caplog):
    """An unknown decode must not silently serve something else."""
    path = _write_checkpoint(tmp_path, decode="mode")
    with caplog.at_level(logging.WARNING):
        predictor = TorchPredictor(str(path), detector=FakeDetector([]))

    assert predictor.decode == "median"
    assert any("unsupported" in r.getMessage().lower() for r in caplog.records)


def test_expectation_decode_changes_the_reported_age(tmp_path):
    """The declared decode must actually reach the number a user sees.

    Guards the wiring, not the maths: a decode key that is parsed, logged and
    reported by /health but never consumed by ``_estimate`` would pass every
    other test here. Uses the pedestal shape where the two decodes provably
    disagree -- 0.9 mass on bin 8 plus 0.1 uniform gives median 8 but
    expectation ~12.2 -- so a decode that silently fell back to median could
    not satisfy both assertions.
    """
    import numpy as np

    median_p = TorchPredictor(
        str(_write_checkpoint(tmp_path / "a", decode="median")), detector=FakeDetector([])
    )
    expect_p = TorchPredictor(
        str(_write_checkpoint(tmp_path / "b", decode="expectation")), detector=FakeDetector([])
    )
    assert (median_p.decode, expect_p.decode) == ("median", "expectation")

    probs = np.full((1, 101), 0.1 / 101, dtype=np.float64)
    probs[0, 8] += 0.9
    logits = torch.from_numpy(np.log(probs)).float()

    class _Fixed:
        def __call__(self, _):
            return logits

        def eval(self):
            return self

    median_p.model = _Fixed()
    expect_p.model = _Fixed()
    batch = np.zeros((1, 3, 224, 224), dtype=np.float32)

    assert median_p._estimate(batch).age[0] == pytest.approx(8.0)
    assert expect_p._estimate(batch).age[0] == pytest.approx(12.2, abs=0.3)
