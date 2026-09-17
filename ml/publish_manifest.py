"""Publish a manifest describing the current model artifacts.

A consumer session benchmarking against ``checkpoints/age_model.pt`` has no way
to tell that the file changed underneath it -- the path is stable and the schema
is pinned, so a retrained model looks identical until the numbers disagree. This
writes a sidecar with content hashes and provenance so a swap is detectable.

The manifest lives beside the reports rather than inside the checkpoint, because
the artifact contract pins ``meta``'s keys and must not grow new ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from model import CHECKPOINT_PATH, REPO_ROOT

ONNX_PATH = REPO_ROOT / "checkpoints" / "age_model.onnx"
MANIFEST_PATH = Path(__file__).resolve().parent / "reports" / "artifact_manifest.json"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build(note: str, ckpt: Path = CHECKPOINT_PATH, onnx: Path = ONNX_PATH) -> dict:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    prefixed = sum(k.startswith("backbone.") for k in state)
    return {
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "note": note,
        "checkpoint": {
            "path": f"checkpoints/{ckpt.name}",
            "sha256": sha256(ckpt),
            "top_level_keys": sorted(payload.keys()),
            "num_tensors": len(state),
            "state_dict_layout": "bare_timm" if prefixed == 0 else "backbone_prefixed",
            "loads_with": (
                'timm.create_model(meta["backbone"], '
                'num_classes=meta["num_bins"]).load_state_dict(ckpt["state_dict"])'
            ),
            "meta": payload["meta"],
        },
        "onnx": {"path": f"checkpoints/{onnx.name}", "sha256": sha256(onnx)},
        "recommended_decode": payload["meta"].get("decode", "median"),
        "recommended_crop_margin": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish artifact manifest")
    parser.add_argument("--note", default="", help="what changed in this publish")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH)
    parser.add_argument(
        "--out", type=Path, default=MANIFEST_PATH,
        help="manifest path; use a distinct file per artifact so publishing one "
             "never invalidates another's recorded hash",
    )
    args = parser.parse_args()

    manifest = build(args.note, args.checkpoint, args.onnx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
