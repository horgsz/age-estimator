"""The static registry must not drift from the server's own tables.

`web/public/models/models.json` is what the GitHub Pages build serves in place
of ``GET /health``: the model catalog, the per-checkpoint accuracy figures, the
age-gating percentage, the per-model caveat, and the crop margin the browser
port has to match.

It is generated from ``server/`` rather than transcribed, but it is *committed*,
because generating it requires torch and making a static-site deploy install a
deep-learning stack to re-derive a 5 KB file is a bad trade. A committed
generated file is a second copy, and a second copy drifts. This test is what
stops it.

If this fails, regenerate rather than editing the JSON::

    .venv/bin/python -m server.tools.export_static_registry \\
        --out web/public/models/models.json

Editing the JSON by hand would put a number in front of a user that no longer
corresponds to anything measured -- which is the whole failure the digest keying
in ``predictor.MEASURED_ACCURACY_BY_DIGEST`` exists to prevent, arriving by a
different route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server import config
from server.tools.export_static_registry import build_registry_json

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "web" / "public" / "models" / "models.json"
CHECKPOINT_DIR = Path(config.MODEL_DIR)


@pytest.fixture(scope="module")
def committed() -> dict:
    if not REGISTRY_PATH.exists():
        pytest.skip(f"{REGISTRY_PATH} is not present")
    return json.loads(REGISTRY_PATH.read_text())


def test_committed_registry_matches_the_server(committed: dict) -> None:
    """The committed file is byte-for-byte what the generator produces now."""
    onnx_files = list(CHECKPOINT_DIR.glob("*.onnx"))
    if not onnx_files:
        pytest.skip("no ONNX exports present to regenerate against")
    pytest.importorskip("onnxruntime")

    regenerated = build_registry_json(CHECKPOINT_DIR)
    assert regenerated == committed, (
        "web/public/models/models.json has drifted from server/. Regenerate it:\n"
        "  python -m server.tools.export_static_registry "
        "--out web/public/models/models.json"
    )


def test_crop_margin_matches_the_server(committed: dict) -> None:
    """The browser reimplements the crop; it must be told the same margin.

    This is the single number the 2.4x-accuracy-loss incident turned on, and it
    is now consumed by a second implementation that cannot read
    ``server/config.py``. Asserted separately from the whole-file comparison so
    a failure names the thing that matters instead of printing a JSON diff.
    """
    assert committed["preprocessing"]["crop_margin"] == config.CROP_MARGIN
    assert committed["preprocessing"]["input_size"] == config.INPUT_SIZE
    assert tuple(committed["preprocessing"]["mean"]) == tuple(config.IMAGENET_MEAN)
    assert tuple(committed["preprocessing"]["std"]) == tuple(config.IMAGENET_STD)


def test_detector_settings_match_the_server(committed: dict) -> None:
    """Detector thresholds change which faces are found, so they travel too."""
    detector = committed["detector"]
    assert detector["score_threshold"] == config.DETECT_SCORE_THRESHOLD
    assert detector["nms_threshold"] == config.DETECT_NMS_THRESHOLD
    assert detector["top_k"] == config.DETECT_TOP_K


def test_every_shipped_model_is_tied_to_a_verified_checkpoint(committed: dict) -> None:
    """No model ships accuracy figures it has not earned.

    The browser runs an ONNX export, not the ``.pt`` the figures were measured
    on. A model may legitimately ship with ``user_facing`` null -- that is the
    fail-closed path -- but it must never ship figures without
    ``export_verified``.
    """
    for model in committed["models"]:
        checkpoint = model.get("checkpoint")
        if checkpoint is None:
            continue
        has_figures = checkpoint.get("user_facing") is not None
        if has_figures:
            assert checkpoint["export_verified"] is True, (
                f"model {model['key']!r} carries accuracy figures but its ONNX export "
                "was not verified against the checkpoint they were measured on"
            )
            assert (
                checkpoint["exported_from_sha256"]
                == checkpoint["expected_checkpoint_sha256"]
            )


def test_the_age_verification_disclaimer_survived_the_port(committed: dict) -> None:
    """The highest-consequence text in the product is a number, and it is here.

    The header's warning is generated from
    ``user_facing.gating_under18_shown_adult_pct``. If that key ever went
    missing the copy would silently de-quantify to "a large share" -- still a
    warning, but a weaker one, and weakened by accident rather than by a
    decision. Both shipped models have a measured figure, so both must carry it.
    """
    figures = {
        model["key"]: (model.get("checkpoint") or {}).get("user_facing")
        for model in committed["models"]
        if model["available"]
    }
    assert figures, "no models are available in the static registry"
    for key, user_facing in figures.items():
        assert user_facing is not None, f"model {key!r} lost its user-facing figures"
        pct = user_facing["gating_under18_shown_adult_pct"]
        assert isinstance(pct, (int, float)) and 0 < pct < 100, (
            f"model {key!r} has an implausible under-18 gating percentage: {pct}"
        )
