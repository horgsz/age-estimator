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
