"""Verifies the pinned checkpoint contract loads and decodes correctly.

Uses a randomly-initialised network saved in exactly the format the ``ml/`` side
promises, so the swap from stub to real weights is exercised without needing a
trained model.
"""

from __future__ import annotations

import pytest

from conftest import FakeDetector

from server import config
from server.predictor import StubPredictor, TorchPredictor, load_predictor

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

BACKBONE = "mobilenetv3_small_100"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=config.NUM_BINS)
    path = tmp_path_factory.mktemp("ckpt") / "age_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "backbone": BACKBONE,
                "num_bins": 101,
                "input_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "test_mae": 5.43,
            },
        },
        path,
    )
    return path


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
    assert info["test_mae"] == pytest.approx(5.43)
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
    assert a["test_mae"] != b["test_mae"]


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
