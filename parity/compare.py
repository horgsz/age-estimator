"""Compare the Python and browser runs, element by element, and say by how much.

This is the deliverable, not the ceremony around it. Two independent
implementations of the same preprocessing exist in this repository, and a silent
mismatch between them degrades every prediction while leaving every test green
-- which has already happened here once, with a crop margin that would have
shipped a 2.4x accuracy loss past a healthy-looking evaluation.

So the comparison is deliberately layered, because "the ages match" is a weak
statement on its own: two pipelines can agree on a rounded integer while
disagreeing about the pixels underneath, and the disagreement only shows up
later, on faces nobody tested.

  1. boxes      the detectors must agree exactly. Both run the same YuNet
                weights, so anything other than 0 means the port's
                post-processing is wrong, not that detectors differ.
  2. tensor     the 224x224x3 float32 input, compared element-wise. This is the
                layer that catches crop geometry, channel order, normalisation
                and resampling. It is reported as max and mean absolute
                difference, and as the count of differing elements.
  3. logits     the model outputs. Isolates ONNX-vs-PyTorch numerics from
                everything upstream.
  4. age        what the user reads.

Thresholds are asserted, not just printed. `--strict` makes any breach a
non-zero exit so CI fails on divergence instead of logging it. If a threshold is
breached the right response is to find out why -- the likely causes are listed
in `parity/README.md` -- not to raise the threshold.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

# Tolerances.
#
# The tensor one is tight on purpose. Both sides run the same crop arithmetic
# and a port of OpenCV's own resize kernels, so the expected difference is zero,
# not "small". Allowing 1e-3 would silently accommodate a whole least-
# significant bit of pixel disagreement (1/255/0.225 = 1.7e-2 after
# normalisation is a full LSB; 1e-3 is well under it) -- so 1e-3 is a genuine
# guard rather than a rubber stamp, and the measured value should be 0.
TENSOR_MAX_ABS = 1e-3
# ONNX export vs PyTorch, measured at ~1e-5 on these weights by
# server/tools/export_static_registry.py. This bounds it at run time too.
LOGIT_MAX_ABS = 1e-2
# The point estimate is an integer bin index, so this should be exactly 0.
# A fractional allowance exists only so a genuine near-tie reports as a small
# number rather than as a pass/fail with no magnitude.
AGE_MAX_ABS = 0.1


def _tensors(payload: dict) -> np.ndarray:
    raw = payload.get("tensors_base64") or payload.get("tensorsBase64") or ""
    if not raw:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(base64.b64decode(raw), dtype="<f4")


def compare(python_path: Path, browser_path: Path, strict: bool) -> int:
    py = json.loads(python_path.read_text())
    br = json.loads(browser_path.read_text())

    by_id = {r["id"]: r for r in br["results"]}
    failures: list[str] = []

    print()
    print(f"{'case':<18} {'faces':>5}  {'box Δ':>6}  {'tensor max Δ':>12}  "
          f"{'tensor mean Δ':>13}  {'elems Δ':>9}  {'logit max Δ':>11}  {'age Δ':>6}")
    print("-" * 104)

    for result in py["results"]:
        case_id = result["id"]
        other = by_id.get(case_id)
        if other is None:
            failures.append(f"{case_id}: missing from the browser run")
            continue

        py_boxes = [list(map(int, b)) for b in result["boxes"]]
        br_boxes = [list(map(int, b)) for b in other["boxes"]]

        if len(py_boxes) != len(br_boxes):
            failures.append(
                f"{case_id}: face count differs -- python {len(py_boxes)}, browser {len(br_boxes)}"
            )
            print(f"{case_id:<18} {len(py_boxes)}/{len(br_boxes):<3}  MISMATCHED FACE COUNT")
            continue

        box_delta = (
            max(
                (abs(a - b) for pa, pb in zip(py_boxes, br_boxes) for a, b in zip(pa, pb)),
                default=0,
            )
            if py_boxes
            else 0
        )

        py_tensor = _tensors(result)
        br_tensor = _tensors(other)
        if py_tensor.shape != br_tensor.shape:
            failures.append(f"{case_id}: tensor shape differs")
            continue

        if py_tensor.size:
            diff = np.abs(py_tensor.astype(np.float64) - br_tensor.astype(np.float64))
            tensor_max = float(diff.max())
            tensor_mean = float(diff.mean())
            tensor_elems = int((diff > 0).sum())
        else:
            tensor_max = tensor_mean = 0.0
            tensor_elems = 0

        logit_max = 0.0
        age_max = 0.0
        for pf, bf in zip(result["faces"], other["faces"]):
            logit_max = max(
                logit_max,
                float(np.max(np.abs(np.array(pf["logits"]) - np.array(bf["logits"])))),
            )
            age_max = max(age_max, abs(float(pf["age"]) - float(bf["age"])))

        print(
            f"{case_id:<18} {len(py_boxes):>5}  {box_delta:>6}  {tensor_max:>12.3e}  "
            f"{tensor_mean:>13.3e}  {tensor_elems:>9}  {logit_max:>11.3e}  {age_max:>6.2f}"
        )

        if box_delta != 0:
            failures.append(f"{case_id}: detector boxes differ by up to {box_delta} px")
        if tensor_max > TENSOR_MAX_ABS:
            failures.append(
                f"{case_id}: preprocessed tensor differs by {tensor_max:.3e} "
                f"(limit {TENSOR_MAX_ABS:g})"
            )
        if logit_max > LOGIT_MAX_ABS:
            failures.append(
                f"{case_id}: logits differ by {logit_max:.3e} (limit {LOGIT_MAX_ABS:g})"
            )
        if age_max > AGE_MAX_ABS:
            failures.append(
                f"{case_id}: predicted age differs by {age_max:.2f} years "
                f"(limit {AGE_MAX_ABS:g})"
            )

    print()
    print("Per-face detail (python -> browser):")
    for result in py["results"]:
        other = by_id.get(result["id"])
        if other is None:
            continue
        for i, (pf, bf) in enumerate(zip(result["faces"], other["faces"])):
            same_tensor = pf["tensor_sha256"] == bf["tensorSha256"]
            print(
                f"  {result['id']:<18} face {i}: "
                f"age {pf['age']:.1f} -> {bf['age']:.1f}  "
                f"range {pf['low']:.1f}-{pf['high']:.1f} -> {bf['low']:.1f}-{bf['high']:.1f}  "
                f"conf {pf['confidence']:.4f} -> {bf['confidence']:.4f}  "
                f"bbox {pf['bbox']} -> {bf['bbox']}  "
                f"tensor {'identical' if same_tensor else 'DIFFERENT'}"
            )

    env = br.get("environment", {})
    cold = br.get("cold", {})
    print()
    print("Browser environment:")
    print(f"  crossOriginIsolated : {env.get('crossOriginIsolated')}")
    print(f"  SharedArrayBuffer   : {env.get('hasSharedArrayBuffer')}")
    if cold:
        print("Cold load (empty cache):")
        print(f"  navigation          : {cold.get('navigationMs')} ms")
        print(f"  first result        : {cold.get('firstResultMs')} ms")
        print(f"  transferred         : {cold.get('transferredBytes', 0) / 1048576:.2f} MB")

    print()
    if failures:
        print(f"FAIL: {len(failures)} discrepancy(ies)")
        for line in failures:
            print(f"  - {line}")
        print()
        print(
            "Do not widen a tolerance to make this pass. See parity/README.md for "
            "the usual causes, in order of likelihood."
        )
        return 1 if strict else 0

    print("PASS: the browser path and the server path agree within tolerance.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--browser", required=True, type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any discrepancy (use in CI).",
    )
    args = parser.parse_args()
    sys.exit(compare(args.python, args.browser, args.strict))


if __name__ == "__main__":
    main()
