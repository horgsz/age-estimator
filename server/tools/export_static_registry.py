"""Bake the served model metadata into a static JSON asset.

The FastAPI build answers ``GET /health`` from
:data:`server.predictor.MEASURED_ACCURACY_BY_DIGEST`, which keys every derived
figure -- typical error, the age-gating percentage, the per-model caveat -- to
the sha256 of the exact checkpoint it was measured on. There is no server in the
GitHub Pages build, so that table has to ship as a file. This tool writes it.

WHY THIS IS NOT A COPY-PASTE OF THE FIGURES
-------------------------------------------
It imports them. A second hand-maintained copy of the accuracy numbers is the
same class of bug the digest keying exists to prevent: someone updates one and
not the other, and the app confidently reports a number that was measured on
different weights. Everything below is read out of ``server/`` at generation
time, so there is exactly one place the figures live.

THE ONNX ARTIFACT IS A DIFFERENT FILE FROM THE CHECKPOINT
---------------------------------------------------------
This is the sharp edge. Every measured figure belongs to a ``.pt`` checkpoint
(``fb629f49987a`` / ``56894c480044``). The browser cannot run a ``.pt``; it runs
``checkpoints/*.onnx``, which are *different files with different digests*.
Attaching the checkpoint's figures to the ONNX file because "it is an export of
the same weights" would be an assertion, and assertions about which weights
produced which number are precisely what went wrong here before.

So this tool does not assert it, it checks it: for each pair it runs the ``.pt``
model and the ``.onnx`` model over identical fixed random inputs and compares
the logits. Only if they agree to ``LOGIT_TOLERANCE`` does the ONNX artifact
inherit the checkpoint's figures, and the recorded provenance says which
checkpoint digest it inherited them from and what the measured agreement was.
If the check fails, that model ships with no figures and a note saying why --
the same fail-closed behaviour ``/health`` already has for an unrecognised
artifact.

Usage::

    .venv/bin/python -m server.tools.export_static_registry \\
        --out web/public/models/models.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .. import config
from ..predictor import (
    INTERVAL_CALIBRATION,
    MEASURED_ACCURACY_BY_DIGEST,
    TorchPredictor,
)

# Max absolute logit difference we accept between a checkpoint and its ONNX
# export. Export is lossless in principle; the observed figure is ~2e-05, which
# is ordinary float32 kernel variation. Anything appreciably larger means the
# ONNX is not the artifact the figures were measured on.
LOGIT_TOLERANCE = 1e-3

# Which ONNX file serves which catalog slot. Deliberately NOT derived by string
# substitution on the checkpoint filename: the mapping is the thing being
# verified, so it is written down and then checked.
ONNX_FOR_KEY = {
    "real": "age_model_realgt.onnx",
    "apparent": "age_model.onnx",
}


def _sha256_prefix(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _compare_logits(pt_path: Path, onnx_path: Path, samples: int = 4) -> float:
    """Max abs logit difference between the checkpoint and its ONNX export."""
    import onnxruntime as rt
    import torch

    predictor = TorchPredictor(str(pt_path))
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((samples, 3, 224, 224), dtype=np.float32)

    with torch.inference_mode():
        torch_logits = predictor.model(torch.from_numpy(batch)).numpy()

    session = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {session.get_inputs()[0].name: batch})[0]

    return float(np.max(np.abs(torch_logits - onnx_logits)))


def build_registry_json(model_dir: Path, *, verify: bool = True) -> dict:
    """The payload the static build serves in place of ``GET /health``."""
    models = []

    for spec in config.MODEL_CATALOG:
        pt_path = model_dir / spec.filename
        onnx_name = ONNX_FOR_KEY.get(spec.key)
        onnx_path = model_dir / onnx_name if onnx_name else None

        entry: dict = {
            "key": spec.key,
            "label": spec.label,
            "question": spec.question,
            "explanation": spec.explanation,
            "available": False,
            "identity_verified": False,
            "asset": None,
            "bytes": None,
            "sha256": None,
            "checkpoint": None,
        }

        if onnx_path is None or not onnx_path.exists():
            entry["unavailable_reason"] = f"No ONNX export at {onnx_path}"
            models.append(entry)
            continue

        onnx_digest = _sha256_prefix(onnx_path)
        entry.update(
            {
                "available": True,
                "asset": onnx_name,
                "bytes": onnx_path.stat().st_size,
                "sha256": onnx_digest,
            }
        )

        # The figures belong to the .pt checkpoint's digest. The ONNX only
        # inherits them if it is demonstrably the same function.
        checkpoint_digest = _sha256_prefix(pt_path) if pt_path.exists() else None
        measured = (
            MEASURED_ACCURACY_BY_DIGEST.get(checkpoint_digest) if checkpoint_digest else None
        )

        max_logit_diff: float | None = None
        export_verified = False
        if verify and pt_path.exists():
            try:
                max_logit_diff = _compare_logits(pt_path, onnx_path)
                export_verified = max_logit_diff <= LOGIT_TOLERANCE
            except Exception as exc:  # pragma: no cover - environment dependent
                entry["export_check_error"] = str(exc)

        checkpoint: dict = {
            "onnx_sha256": onnx_digest,
            "exported_from_sha256": checkpoint_digest,
            "expected_checkpoint_sha256": spec.expected_digest,
            "export_verified": export_verified,
            "export_max_logit_diff": max_logit_diff,
            "export_logit_tolerance": LOGIT_TOLERANCE,
        }

        identity_ok = checkpoint_digest == spec.expected_digest

        if measured is not None and identity_ok and export_verified:
            checkpoint.update(measured)
            entry["identity_verified"] = True
        else:
            # Fails closed, exactly as ``TorchPredictor.describe_checkpoint``
            # does: no figures and NO CAVEAT for an artifact whose provenance we
            # have not established. A caveat is a directional claim about
            # specific weights.
            if not identity_ok:
                why = (
                    f"the checkpoint beside it (sha256 {checkpoint_digest}) is not the "
                    f"artifact this slot was measured on (expected {spec.expected_digest})"
                )
            elif not export_verified:
                why = (
                    "the ONNX export could not be shown to match the measured "
                    "checkpoint"
                    + (
                        f" (max logit difference {max_logit_diff:.3g} > {LOGIT_TOLERANCE:g})"
                        if max_logit_diff is not None
                        else " (the comparison did not run)"
                    )
                )
            else:
                why = "no end-to-end figures have been measured for this checkpoint"
            checkpoint.update(
                {
                    "real_age_mae": None,
                    "real_age_corpus": None,
                    "in_corpus_mae_utkface": None,
                    "user_facing": None,
                    "accuracy_note": (
                        f"No accuracy figures are shown for this model because {why}. "
                        "Re-run server/tools/eval_end_to_end.py before quoting any "
                        "number for it."
                    ),
                }
            )
            entry["label"] = f"{spec.label} (unverified build)"
            entry["explanation"] = (
                "This build could not tie the browser's copy of this model to the "
                "artifact its accuracy was measured on, so its error figures and "
                "known limitations are unknown here. Treat the numbers as "
                "unvalidated."
            )

        entry["checkpoint"] = checkpoint
        models.append(entry)

    available = [m["key"] for m in models if m["available"]]
    default = config.DEFAULT_MODEL_KEY
    if default not in available and available:
        default = available[0]

    return {
        "default": default,
        "available": available,
        "models": models,
        # A sibling of `models`, not a field inside each entry, and that
        # placement is the point: interval coverage is the one derived number
        # here that is NOT per-checkpoint.
        "interval_calibration": INTERVAL_CALIBRATION,
        # The browser reimplements the crop; the parity harness asserts these
        # match what it compiled in, so the two cannot drift apart silently.
        "preprocessing": {
            "crop_margin": config.CROP_MARGIN,
            "input_size": config.INPUT_SIZE,
            "mean": list(config.IMAGENET_MEAN),
            "std": list(config.IMAGENET_STD),
            "num_bins": config.NUM_BINS,
            "decode": "median",
        },
        "detector": {
            "name": "YuNet (face_detection_yunet_2023mar)",
            "score_threshold": config.DETECT_SCORE_THRESHOLD,
            "nms_threshold": config.DETECT_NMS_THRESHOLD,
            "top_k": config.DETECT_TOP_K,
            "max_detect_side": 1024,
        },
        "catalog": [asdict(spec) for spec in config.MODEL_CATALOG],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Path to write models.json")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(config.MODEL_DIR),
        help="Directory holding the .pt checkpoints and .onnx exports",
    )
    parser.add_argument(
        "--copy-models",
        action="store_true",
        help="Also copy the .onnx exports next to the JSON",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip the checkpoint/ONNX logit comparison. Every model then ships "
            "with no accuracy figures, which is the point of the flag: it is for "
            "environments without torch, not a way to make the check pass."
        ),
    )
    args = parser.parse_args()

    payload = build_registry_json(args.model_dir, verify=not args.no_verify)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    for model in payload["models"]:
        ck = model.get("checkpoint") or {}
        print(
            f"  {model['key']:9s} asset={model['asset']} "
            f"onnx={ck.get('onnx_sha256')} from={ck.get('exported_from_sha256')} "
            f"verified={ck.get('export_verified')} "
            f"max_logit_diff={ck.get('export_max_logit_diff')}"
        )

    if args.copy_models:
        for model in payload["models"]:
            if not model["asset"]:
                continue
            src = args.model_dir / model["asset"]
            dst = args.out.parent / model["asset"]
            shutil.copyfile(src, dst)
            print(f"  copied {src} -> {dst}")


if __name__ == "__main__":
    main()
