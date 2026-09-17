"""The Python half of the parity harness.

Runs the SERVER's inference path -- `server.detector`, `server.preprocessing`,
`server.predictor` -- over a fixed set of images and dumps every intermediate as
JSON. `run_browser.mjs` produces the same shape from the browser build, and
`compare.py` diffs them.

Nothing here reimplements anything: it imports the production modules, so if the
server's crop or decode changes, this side changes with it automatically and the
browser side is the one that has to keep up. That asymmetry is the point -- the
Python is the reference, the TypeScript is the port.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server import config, preprocessing  # noqa: E402
from server.detector import FaceDetector  # noqa: E402
from server.predictor import LOW_Q, HIGH_Q, TorchPredictor, build_result  # noqa: E402


def _quantile(cdf: np.ndarray, level: float) -> int:
    """`(cdf < level).sum()`, clamped -- the server's quantile decode."""
    return int(np.clip(int((cdf < level).sum()), 0, cdf.shape[0] - 1))


def run_case(
    predictor: TorchPredictor,
    detector: FaceDetector,
    image_path: Path,
    margin: float,
    boxes: list[list[int]] | None,
) -> dict:
    raw = np.fromfile(image_path, dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not decode {image_path}")

    t0 = time.perf_counter()
    detected = [list(map(int, b)) for b in detector.detect(frame)] if boxes is None else boxes
    t1 = time.perf_counter()

    if not detected:
        return {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "crop_margin": margin,
            "boxes": [],
            "faces": [],
            "tensors_base64": "",
            "timings": {"detect_ms": (t1 - t0) * 1000, "preprocess_ms": 0.0, "infer_ms": 0.0},
        }

    batch = np.stack(
        [preprocessing.preprocess_face(frame, tuple(b), margin) for b in detected]
    ).astype(np.float32)
    t2 = time.perf_counter()

    import torch

    with torch.inference_mode():
        logits = predictor.model(torch.from_numpy(batch)).float().numpy()
    t3 = time.perf_counter()

    faces = []
    for i, box in enumerate(detected):
        row = logits[i]
        exps = np.exp(row - row.max())
        probs = exps / exps.sum()
        cdf = np.cumsum(probs)
        median = _quantile(cdf, 0.5)
        low = _quantile(cdf, LOW_Q)
        high = _quantile(cdf, HIGH_Q)
        expectation = float((probs * np.arange(probs.shape[0])).sum())
        std = float(
            np.sqrt(max((probs * (np.arange(probs.shape[0]) - expectation) ** 2).sum(), 0.0))
        )
        result = build_result(tuple(box), median, low, high)
        tensor = np.ascontiguousarray(batch[i])
        faces.append(
            {
                "bbox": [int(v) for v in box],
                "age": result.age,
                "low": result.low,
                "high": result.high,
                "confidence": result.confidence,
                "median": float(median),
                "expectation": expectation,
                "std": std,
                "logits": [float(v) for v in row],
                "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
            }
        )

    return {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "crop_margin": margin,
        "boxes": detected,
        "faces": faces,
        "tensors_base64": base64.b64encode(np.ascontiguousarray(batch).tobytes()).decode(),
        "timings": {
            "detect_ms": (t1 - t0) * 1000,
            "preprocess_ms": (t2 - t1) * 1000,
            "infer_ms": (t3 - t2) * 1000,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="cases.json")
    parser.add_argument("--out", required=True, type=Path, help="where to write results")
    args = parser.parse_args()

    spec = json.loads(args.cases.read_text())
    fixtures_dir = args.cases.parent

    detector = FaceDetector()
    predictors: dict[str, TorchPredictor] = {}
    results = []

    for case in spec["cases"]:
        model_key = case["model"]
        if model_key not in predictors:
            filename = next(s.filename for s in config.MODEL_CATALOG if s.key == model_key)
            predictors[model_key] = TorchPredictor(
                str(Path(config.MODEL_DIR) / filename), detector=detector
            )
        margin = case.get("crop_margin", config.CROP_MARGIN)
        results.append(
            {
                "id": case["id"],
                "model": model_key,
                **run_case(
                    predictors[model_key],
                    detector,
                    fixtures_dir / case["image"],
                    margin,
                    case.get("boxes"),
                ),
            }
        )
        print(f"  {case['id']}: {len(results[-1]['faces'])} face(s)")

    args.out.write_text(
        json.dumps(
            {
                "source": "python",
                "crop_margin_default": config.CROP_MARGIN,
                "input_size": config.INPUT_SIZE,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
