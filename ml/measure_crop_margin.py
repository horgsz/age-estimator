"""Measure the crop margin that makes inference framing match UTKFace framing.

The serving path detects a face with YuNet, expands the bbox to a square of
``side = max(w, h) * (1 + 2 * CROP_MARGIN)``, then resizes to 224. UTKFace
aligned+cropped images are already tightly framed, and the model is trained on
that native framing. So the correct ``CROP_MARGIN`` is the one that reproduces
the UTKFace framing from a YuNet detection:

    full_side = max(w_det, h_det) * (1 + 2m)   =>   m = (full_side / max(w, h) - 1) / 2

Run this to report the median/IQR of ``m`` and the detection failure rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data import REPO_ROOT, build_manifest

YUNET_PATH = REPO_ROOT / "checkpoints" / "pretrained" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_yunet(path: Path = YUNET_PATH) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet -> {path}")
    subprocess.run(["curl", "-sL", "--fail", "-o", str(path), YUNET_URL], check=True)
    return path


def measure(
    sample: pd.DataFrame, score_threshold: float = 0.6
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    model_path = ensure_yunet()
    detector = cv2.FaceDetectorYN.create(
        str(model_path), "", (200, 200), score_threshold, 0.3, 5000
    )

    margins: list[float] = []
    failures = 0
    sizes: list[tuple[int, int]] = []

    for rel_path in sample["path"]:
        image = cv2.imread(str(REPO_ROOT / rel_path))
        if image is None:
            failures += 1
            continue
        h, w = image.shape[:2]
        sizes.append((w, h))
        detector.setInputSize((w, h))
        _, faces = detector.detect(image)
        if faces is None or len(faces) == 0:
            failures += 1
            continue

        # Keep the highest-confidence detection (column 14 is the score).
        face = faces[np.argmax(faces[:, 14])]
        det_w, det_h = float(face[2]), float(face[3])
        longest = max(det_w, det_h)
        if longest <= 0:
            failures += 1
            continue

        # The UTKFace image *is* the crop, so its own side is the target side.
        full_side = float(max(w, h))
        margins.append((full_side / longest - 1.0) / 2.0)

    return np.asarray(margins), failures, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the YuNet crop margin")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    args = parser.parse_args()

    manifest = build_manifest(verbose=False)
    sample = manifest.sample(
        n=min(args.sample, len(manifest)), random_state=args.seed
    ).reset_index(drop=True)
    print(f"Sampling {len(sample)} of {len(manifest)} UTKFace images")

    margins, failures, sizes = measure(sample, args.score_threshold)
    detected = len(margins)
    total = len(sample)

    unique_sizes = sorted(set(sizes))
    print(f"Image sizes present: {unique_sizes[:5]}{' ...' if len(unique_sizes) > 5 else ''}")
    print(f"\nDetections: {detected}/{total}")
    print(f"Detection failure rate: {100.0 * failures / total:.2f}% ({failures} images)")

    if detected == 0:
        raise SystemExit("No detections; cannot estimate a margin.")

    q1, med, q3 = np.percentile(margins, [25, 50, 75])
    print("\n=== Implied CROP_MARGIN (m) ===")
    print(f"median : {med:.4f}")
    print(f"IQR    : [{q1:.4f}, {q3:.4f}]  (width {q3 - q1:.4f})")
    print(f"mean   : {margins.mean():.4f}   std: {margins.std():.4f}")
    print(f"p05/p95: {np.percentile(margins, 5):.4f} / {np.percentile(margins, 95):.4f}")
    print(f"min/max: {margins.min():.4f} / {margins.max():.4f}")

    # Translate the margin spread into the RandomResizedCrop scale range that
    # covers it, so train-time augmentation spans the framings seen at serve time.
    print("\n=== Implied framing scale relative to the median ===")
    for label, value in [
        ("p05", np.percentile(margins, 5)),
        ("q1", q1),
        ("median", med),
        ("q3", q3),
        ("p95", np.percentile(margins, 95)),
    ]:
        # A crop taken with margin `value` covers this fraction of the median
        # framing's area.
        ratio = ((1 + 2 * value) / (1 + 2 * med)) ** 2
        print(f"{label:>7}: m={value:.4f}  area vs median = {ratio:.3f}")


if __name__ == "__main__":
    main()
