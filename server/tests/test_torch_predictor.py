"""Verifies the pinned checkpoint contract loads and decodes correctly.

Uses a randomly-initialised network saved in exactly the format the ``ml/`` side
promises, so the swap from stub to real weights is exercised without needing a
trained model.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from conftest import FakeDetector

from server import config
from server.predictor import (
    MEASURED_ACCURACY_BY_DIGEST,
    StubPredictor,
    TorchPredictor,
    load_predictor,
)

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

BACKBONE = "mobilenetv3_small_100"


def _write_checkpoint(directory, *, decode=None, name="age_model.pt", **extra_meta):
    """Save a contract-shaped checkpoint, optionally declaring extra metadata."""
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
    meta.update(extra_meta)
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


def test_health_does_not_present_in_corpus_mae_as_real_accuracy():
    """UTKFace labels are DEX estimates, so in-corpus MAE is not accuracy.

    Where an artifact has both figures they are published together, so the
    distinction travels with the number. The real-age figure must be the larger
    -- if a future change makes the in-corpus number look like the headline, a
    caller would understate the error a user actually experiences by ~4 years.
    """
    utk = MEASURED_ACCURACY_BY_DIGEST["56894c480044"]
    assert utk["real_age_mae"] == pytest.approx(8.52)
    assert utk["in_corpus_mae_utkface"] == pytest.approx(4.762)
    assert utk["real_age_mae"] > utk["in_corpus_mae_utkface"]
    assert "DEX" in utk["accuracy_note"]


def test_real_gt_model_publishes_no_invented_in_corpus_figure():
    """The real-GT model never saw UTKFace, so it has no in-corpus figure.

    Filling that field for it -- by carrying the other model's 4.762 across, or
    by relabelling its own real-age number -- would manufacture a result that
    was never measured. Absent means absent.
    """
    realgt = MEASURED_ACCURACY_BY_DIGEST["fb629f49987a"]
    assert realgt["in_corpus_mae_utkface"] is None
    assert realgt["real_age_mae"] == pytest.approx(6.34, abs=0.05)
    # The two models' real-age figures are NOT interchangeable.
    assert realgt["real_age_mae"] < MEASURED_ACCURACY_BY_DIGEST["56894c480044"]["real_age_mae"]


def test_accuracy_figures_are_withheld_for_an_unmeasured_checkpoint(checkpoint):
    """Accuracy belongs to a set of weights, not to "the model".

    Our figures were measured on specific artifacts. Sibling checkpoints exist
    (no-smoothing and CE variants), and pointing AGE_MODEL_PATH at one to
    evaluate it must not make /health report another model's accuracy as though
    it were measured. Fails closed.
    """
    predictor = TorchPredictor(str(checkpoint), detector=FakeDetector([]))
    info = predictor.describe_checkpoint()

    assert info["sha256"] not in MEASURED_ACCURACY_BY_DIGEST
    assert info["in_corpus_mae_utkface"] is None
    assert info["real_age_mae"] is None
    assert "Unmeasured" in info["accuracy_note"]
    # The artifacts we *did* measure stay discoverable, so the mismatch can be
    # diagnosed rather than merely observed.
    for digest in MEASURED_ACCURACY_BY_DIGEST:
        assert digest in info["accuracy_note"]


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


def test_crop_margin_mismatch_with_training_is_flagged_loudly(tmp_path, caplog, monkeypatch):
    """A serving crop that differs from the training crop must not be silent.

    The crop is the largest preprocessing lever we have, and a mismatch has no
    symptom: faces are simply framed differently from training and every
    prediction degrades. Checkpoints now record `crop_margin`, so this is
    checkable rather than a comment asking a future maintainer to remember.
    """
    monkeypatch.setattr(config, "CROP_MARGIN", 0.4)
    path = _write_checkpoint(tmp_path / "m", crop_margin=0.0)

    with caplog.at_level(logging.WARNING):
        predictor = TorchPredictor(str(path), detector=FakeDetector([]))

    info = predictor.describe_checkpoint()
    assert info["trained_crop_margin"] == pytest.approx(0.0)
    assert info["serving_crop_margin"] == pytest.approx(0.4)
    assert info["crop_margin_matches_training"] is False

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "CROP MARGIN MISMATCH" in messages
    # The warning has to name the value to set, or it just tells someone that
    # something is wrong without saying what to do about it.
    assert "0.0" in messages


def test_matching_crop_margin_is_not_warned_about(tmp_path, caplog, monkeypatch):
    """The check must stay quiet when things are correct, or it gets ignored."""
    monkeypatch.setattr(config, "CROP_MARGIN", 0.0)
    path = _write_checkpoint(tmp_path / "m", crop_margin=0.0)

    with caplog.at_level(logging.WARNING):
        predictor = TorchPredictor(str(path), detector=FakeDetector([]))

    assert predictor.describe_checkpoint()["crop_margin_matches_training"] is True
    assert "CROP MARGIN MISMATCH" not in " ".join(r.getMessage() for r in caplog.records)


def test_crop_margin_is_not_silently_overridden_by_the_checkpoint(tmp_path, monkeypatch):
    """We warn on drift, we do not override.

    CROP_MARGIN is deliberately tunable at runtime for A/B sweeps, so silently
    replacing an operator's explicit setting with the checkpoint's would break
    the margin harness and ignore a deliberate instruction.
    """
    monkeypatch.setattr(config, "CROP_MARGIN", 0.4)
    path = _write_checkpoint(tmp_path / "m", crop_margin=0.0)
    TorchPredictor(str(path), detector=FakeDetector([]))

    assert config.CROP_MARGIN == pytest.approx(0.4)


def test_undeclared_crop_margin_reports_unknown_not_a_false_match(tmp_path, monkeypatch):
    """The shipped artifact predates `crop_margin`; absence is not agreement."""
    monkeypatch.setattr(config, "CROP_MARGIN", 0.0)
    predictor = TorchPredictor(
        str(_write_checkpoint(tmp_path / "m")), detector=FakeDetector([])
    )
    info = predictor.describe_checkpoint()

    assert info["trained_crop_margin"] is None
    # Must be None, not True. Reporting a match we never verified would be a
    # false assurance, which is worse than admitting we cannot tell.
    assert info["crop_margin_matches_training"] is None


def test_malformed_crop_margin_does_not_break_startup(tmp_path, monkeypatch):
    """Metadata is written by another codebase; junk must degrade, not crash."""
    monkeypatch.setattr(config, "CROP_MARGIN", 0.0)
    predictor = TorchPredictor(
        str(_write_checkpoint(tmp_path / "m", crop_margin="not-a-number")),
        detector=FakeDetector([]),
    )
    assert predictor.describe_checkpoint()["trained_crop_margin"] is None


def test_experiment_intermediate_role_is_surfaced(tmp_path):
    """An artifact built to prove a point must not pass as the published model.

    Three of the real-GT checkpoints are label-smoothing study intermediates.
    A pasted /health payload should make that obvious.
    """
    path = _write_checkpoint(
        tmp_path / "r", role="EXPERIMENT INTERMEDIATE from the label-smoothing study"
    )
    info = TorchPredictor(str(path), detector=FakeDetector([])).describe_checkpoint()
    assert "EXPERIMENT INTERMEDIATE" in info["role"]


def test_provisional_val_derived_mae_announces_itself(tmp_path):
    """train.py stamps best *val* MAE into test_mae until eval.py overwrites it.

    Until then the field is a selection-set score flattering itself by up to
    ~0.3 years, under a name that reads as a held-out result. Surfacing the
    source is what lets a consumer tell the two apart.
    """
    path = _write_checkpoint(
        tmp_path / "p", test_mae=6.464, test_mae_source="val", val_mae=6.464
    )
    info = TorchPredictor(str(path), detector=FakeDetector([])).describe_checkpoint()

    assert info["recorded_test_mae_source"] == "val"
    assert info["recorded_val_mae"] == pytest.approx(6.464)


def test_corpus_and_label_semantics_travel_with_the_number(tmp_path):
    """An MAE is meaningless without knowing what it is an error against.

    5.5472 against DEX-estimated apparent age and 6.393 against real
    chronological age are not comparable, and the smaller is the weaker result.
    """
    path = _write_checkpoint(
        tmp_path / "c",
        corpus="real_ground_truth (AgeDB + APPA-REAL + FG-NET)",
        label_semantics="real chronological age",
    )
    info = TorchPredictor(str(path), detector=FakeDetector([])).describe_checkpoint()

    assert "AgeDB" in info["recorded_test_mae_corpus"]
    assert info["label_semantics"] == "real chronological age"


def test_explicit_model_path_wins_over_candidates(tmp_path, monkeypatch):
    """An operator's explicit AGE_MODEL_PATH must never be second-guessed."""
    target = tmp_path / "chosen.pt"
    target.write_bytes(b"x")
    monkeypatch.setenv("AGE_MODEL_PATH", str(target))
    config.reload_from_env()
    assert config.AGE_MODEL_PATH == str(target)


def test_candidates_prefer_the_real_gt_model_then_fall_back(tmp_path, monkeypatch):
    """Resolution order is real-GT first, then the older UTKFace model.

    These are different models trained on different corpora with different
    label semantics, not versions of one -- so the order is a deliberate choice
    about which to serve, and the fallback exists so the app still serves real
    weights rather than the stub if only the older artifact is present.
    """
    monkeypatch.delenv("AGE_MODEL_PATH", raising=False)
    realgt = tmp_path / "age_model_realgt.pt"
    older = tmp_path / "age_model.pt"
    monkeypatch.setattr(
        config, "MODEL_PATH_CANDIDATES", (str(realgt), str(older)), raising=False
    )

    older.write_bytes(b"x")
    config.reload_from_env()
    assert config.AGE_MODEL_PATH == str(older), "should fall back when real-GT absent"

    realgt.write_bytes(b"x")
    config.reload_from_env()
    assert config.AGE_MODEL_PATH == str(realgt), "should prefer real-GT when present"


# --------------------------------------------------------------------------
# model registry: per-request selection
# --------------------------------------------------------------------------


def _stub_entry(key: str, label: str, age: float, detector=None):
    """A registry entry backed by a predictor returning a fixed age."""
    from server import registry as registry_mod
    from server.predictor import Decoded, StubPredictor

    class Fixed(StubPredictor):
        is_stub = False
        model_name = f"fixed-{key}"

        def _estimate(self, batch):
            n = len(batch)
            return Decoded(
                age=np.full(n, age, dtype=np.float32),
                low=np.full(n, age - 2, dtype=np.float32),
                high=np.full(n, age + 2, dtype=np.float32),
                expectation=np.full(n, age, dtype=np.float32),
                std=np.full(n, 2.0, dtype=np.float32),
            )

    return registry_mod.ModelEntry(
        key=key,
        label=label,
        question=label,
        explanation="",
        predictor=Fixed(detector=detector),
        path=f"/tmp/{key}.pt",
    )


def test_each_model_key_routes_to_a_different_model(face_bytes, yunet_detector):
    """The whole point of the toggle: two keys, two different answers."""
    from fastapi.testclient import TestClient

    from server import app as app_mod
    from server.registry import ModelRegistry

    registry = ModelRegistry(
        [_stub_entry("real", "How old they are", 40.0, yunet_detector),
         _stub_entry("apparent", "How old they look", 25.0, yunet_detector)],
        default_key="real",
    )
    app_mod.set_predictor(None)
    app_mod.set_registry(registry)
    try:
        client = TestClient(app_mod.app)

        default = client.post("/estimate", files={"image": ("f.jpg", face_bytes, "image/jpeg")})
        assert default.status_code == 200
        assert default.headers["X-Model"] == "real"
        assert default.json()["faces"][0]["age"] == 40.0

        other = client.post(
            "/estimate",
            files={"image": ("f.jpg", face_bytes, "image/jpeg")},
            data={"model": "apparent"},
        )
        assert other.status_code == 200
        assert other.headers["X-Model"] == "apparent"
        assert other.json()["faces"][0]["age"] == 25.0

        unknown = client.post(
            "/estimate",
            files={"image": ("f.jpg", face_bytes, "image/jpeg")},
            data={"model": "nope"},
        )
        assert unknown.status_code == 422
    finally:
        app_mod.set_registry(None)
        app_mod.set_predictor(None)


def test_a_slot_refuses_the_other_models_weights(tmp_path, monkeypatch):
    """Serving apparent-age weights under 'how old they actually are' is a lie.

    Filenames have been reused for different weights twice in this project, so
    identity is content-addressed and a mismatched artifact is refused outright
    rather than served under the wrong label.
    """
    from server import config
    from server import registry as registry_mod

    spec = config.MODEL_CATALOG[0]
    assert spec.key == "real"

    ckpt = _write_checkpoint(tmp_path, name="age_model_realgt.pt")
    # The legacy single-model override applies to the default slot, and the
    # test suite sets it globally; clear it so MODEL_DIR is what resolves.
    monkeypatch.delenv("AGE_MODEL_PATH", raising=False)
    monkeypatch.setattr(config, "MODEL_DIR", str(tmp_path))
    # Claim this file's digest belongs to the *other* catalog entry.
    digest = TorchPredictor(str(ckpt)).describe_checkpoint()["sha256"]
    monkeypatch.setattr(config, "digest_owner", lambda d: "apparent" if d == digest else None)

    entry = registry_mod._load_entry(spec, detector=None)
    assert not entry.available
    assert "apparent" in (entry.unavailable_reason or "")


def test_user_facing_figures_are_keyed_per_checkpoint():
    """The >=40 caveat is true of one model and false of the other.

    This is the reason the caveat lives beside a digest instead of in the UI.
    Applying either model's band to the other would assert a measured
    limitation about predictions it was never measured on.
    """
    real = MEASURED_ACCURACY_BY_DIGEST["fb629f49987a"]["user_facing"]
    apparent = MEASURED_ACCURACY_BY_DIGEST["56894c480044"]["user_facing"]

    assert real["caveat"] is None
    assert apparent["caveat"]["min_age"] == 40
    assert real["gating_under18_shown_adult_pct"] != apparent["gating_under18_shown_adult_pct"]


def test_an_unmeasured_checkpoint_gets_no_caveat(tmp_path):
    """Fails closed: an unrecognised artifact asserts no directional claim."""
    ckpt = _write_checkpoint(tmp_path, name="mystery.pt")
    info = TorchPredictor(str(ckpt)).describe_checkpoint()

    assert info["sha256"] not in MEASURED_ACCURACY_BY_DIGEST
    assert info["user_facing"] is None
