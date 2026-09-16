"""Measure the crop margin that makes inference framing match UTKFace framing.

The serving path detects a face with YuNet, expands the bbox to a square of
``side = max(w, h) * (1 + 2 * CROP_MARGIN)``, then resizes to 224. UTKFace
aligned+cropped images are already tightly framed, and the model is trained on
that native framing. So the correct ``CROP_MARGIN`` is the one that reproduces
the UTKFace framing from a YuNet detection:

    full_side = max(w_det, h_det) * (1 + 2m)   =>   m = (full_side / max(w, h) - 1) / 2

Three things are reported, because the naive version of this measurement rests
on assumptions worth testing:

1. The median over all detections.
2. The median excluding *truncated* samples, where the YuNet box touches or
   exceeds the frame edge. Some UTKFace crops clip the face, which biases the
   box on exactly those images.
3. A scale/context invariance check. The measurement runs YuNet on tight 200x200
   crops, but at inference YuNet sees a full webcam frame. This pastes each crop
   into a larger padded canvas (optionally downscaled, so the face occupies a
   realistic fraction of the frame) and re-measures, which directly tests the
   invariance assumption the whole constant rests on.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data import REPO_ROOT, build_manifest

YUNET_PATH = (
    REPO_ROOT / "checkpoints" / "pretrained" / "face_detection_yunet_2023mar.onnx"
)
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
EDGE_TOL = 1.0  # px; a box within this of the border counts as truncated


def ensure_yunet(path: Path = YUNET_PATH) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet -> {path}")
    subprocess.run(["curl", "-sL", "--fail", "-o", str(path), YUNET_URL], check=True)
    return path


def make_detector(score_threshold: float) -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        str(ensure_yunet()), "", (200, 200), score_threshold, 0.3, 5000
    )


def best_face(detector: cv2.FaceDetectorYN, image: np.ndarray):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    return faces[np.argmax(faces[:, 14])]


def implied_margin(det_side: float, full_side: float) -> float:
    return (full_side / det_side - 1.0) / 2.0


def make_canvas(
    image: np.ndarray,
    pad_factor: float,
    render: int | None,
    pad_mode: str = "replicate",
) -> np.ndarray:
    """Edge-pad a tight crop into a larger frame, optionally rescaling it."""
    height, width = image.shape[:2]
    pad_x = round(width * (pad_factor - 1.0) / 2.0)
    pad_y = round(height * (pad_factor - 1.0) / 2.0)
    if pad_mode == "replicate":
        canvas = cv2.copyMakeBorder(
            image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
        )
    elif pad_mode == "gray":
        canvas = cv2.copyMakeBorder(
            image,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=(128, 128, 128),
        )
    elif pad_mode == "blur":
        # Blurred, upscaled copy of the crop as a stand-in for real background.
        canvas = cv2.resize(
            image,
            (width + 2 * pad_x, height + 2 * pad_y),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=max(3.0, width / 20.0))
        canvas[pad_y : pad_y + height, pad_x : pad_x + width] = image
    else:
        raise ValueError(f"unknown pad_mode {pad_mode!r}")
    if render is not None and render != canvas.shape[0]:
        canvas = cv2.resize(canvas, (render, render), interpolation=cv2.INTER_AREA)
    return canvas


def describe(name: str, values: np.ndarray) -> dict[str, float]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    print(
        f"{name:<40} n={len(values):>4}  median={med:+.4f}  "
        f"IQR=[{q1:+.4f}, {q3:+.4f}]  mean={values.mean():+.4f}"
    )
    return {"n": float(len(values)), "median": float(med), "q1": float(q1), "q3": float(q3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the YuNet crop margin")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument(
        "--pad-factor",
        type=float,
        default=3.0,
        help="canvas size relative to the crop for the invariance check",
    )
    parser.add_argument(
        "--pad-mode",
        choices=["replicate", "gray", "blur"],
        default="replicate",
        help="how to synthesise background for the full-frame check",
    )
    parser.add_argument(
        "--render",
        type=int,
        default=320,
        help="resize the canvas to this many px, shrinking the face to a "
        "realistic webcam size",
    )
    args = parser.parse_args()

    manifest = build_manifest(verbose=False)
    sample = manifest.sample(
        n=min(args.sample, len(manifest)), random_state=args.seed
    ).reset_index(drop=True)
    print(f"Sampling {len(sample)} of {len(manifest)} UTKFace images\n")

    detector = make_detector(args.score_threshold)

    rows: list[dict[str, float | bool]] = []
    failures = {"tight": 0, "context": 0, "scaled": 0}

    for rel_path in sample["path"]:
        image = cv2.imread(str(REPO_ROOT / rel_path))
        if image is None:
            failures["tight"] += 1
            continue
        height, width = image.shape[:2]
        full_side = float(max(width, height))

        face = best_face(detector, image)
        if face is None:
            failures["tight"] += 1
            continue
        x, y, box_w, box_h = (float(v) for v in face[:4])
        side = max(box_w, box_h)
        if side <= 0:
            failures["tight"] += 1
            continue

        truncated = (
            x <= EDGE_TOL
            or y <= EDGE_TOL
            or (x + box_w) >= (width - EDGE_TOL)
            or (y + box_h) >= (height - EDGE_TOL)
        )
        # A box that merely *touches* the border is expected here: the crop is
        # tight, so almost every box nearly fills the frame. Only a box that
        # runs past the border implies the crop actually clipped the face.
        clipped = (
            x < -EDGE_TOL
            or y < -EDGE_TOL
            or (x + box_w) > (width + EDGE_TOL)
            or (y + box_h) > (height + EDGE_TOL)
        )

        row: dict[str, float | bool] = {
            "m_tight": implied_margin(side, full_side),
            "truncated": bool(truncated),
            "clipped": bool(clipped),
        }

        # Same pixel scale, but now surrounded by background context.
        context = make_canvas(image, args.pad_factor, None, args.pad_mode)
        ctx_face = best_face(detector, context)
        if ctx_face is None:
            failures["context"] += 1
        else:
            ctx_side = max(float(ctx_face[2]), float(ctx_face[3]))
            if ctx_side > 0:
                row["m_context"] = implied_margin(ctx_side, full_side)

        # Background context *and* a smaller face, as on a real webcam frame.
        scaled = make_canvas(image, args.pad_factor, args.render, args.pad_mode)
        scale_back = (width * args.pad_factor) / scaled.shape[1]
        sc_face = best_face(detector, scaled)
        if sc_face is None:
            failures["scaled"] += 1
        else:
            sc_side = max(float(sc_face[2]), float(sc_face[3])) * scale_back
            if sc_side > 0:
                row["m_scaled"] = implied_margin(sc_side, full_side)

        rows.append(row)

    frame = pd.DataFrame(rows)
    total = len(sample)
    print(
        f"Tight-crop detections: {len(frame)}/{total} "
        f"(failure {100.0 * failures['tight'] / total:.2f}%)"
    )
    print(
        f"Canvas detections    : context {len(frame) - failures['context']}, "
        f"scaled {len(frame) - failures['scaled']}"
    )

    truncated = frame["truncated"]
    clipped = frame["clipped"]
    print(
        f"\nBox touches frame edge: {int(truncated.sum())}/{len(frame)} "
        f"({100.0 * truncated.mean():.1f}%)  <- expected, crops are tight"
    )
    print(
        f"Box runs past frame edge (face actually clipped): {int(clipped.sum())}"
        f"/{len(frame)} ({100.0 * clipped.mean():.1f}%)"
    )

    print("\n=== Implied CROP_MARGIN (tight-crop detection) ===")
    all_stats = describe("all detections", frame["m_tight"].to_numpy())
    describe("excluding edge-touching", frame.loc[~truncated, "m_tight"].to_numpy())
    clean_stats = describe("excluding clipped", frame.loc[~clipped, "m_tight"].to_numpy())
    if clipped.any():
        describe("clipped only", frame.loc[clipped, "m_tight"].to_numpy())
    print(
        f"\nMedian shift from dropping clipped samples: "
        f"{clean_stats['median'] - all_stats['median']:+.4f}"
    )
    print(
        "Note: 'excluding edge-touching' selects loosely-cropped images, so it "
        "is a biased subset, not a correction."
    )

    print(f"\n=== Scale / context invariance (pad x{args.pad_factor:g}) ===")
    print("This is the condition that matters: at inference YuNet sees a full frame.")
    for column, label in [
        ("m_context", "full frame, face at native px"),
        (
            "m_scaled",
            f"full frame, face ~{args.render / args.pad_factor:.0f}px (webcam-like)",
        ),
    ]:
        if column not in frame:
            print(f"{label}: no detections")
            continue
        subset = frame.loc[frame[column].notna() & ~clipped]
        if subset.empty:
            print(f"{label}: no usable detections")
            continue
        describe(label, subset[column].to_numpy())
        delta = (subset[column] - subset["m_tight"]).to_numpy()
        print(
            f"{'  per-image delta vs tight':<40} "
            f"median={np.median(delta):+.4f}  "
            f"IQR=[{np.percentile(delta, 25):+.4f}, {np.percentile(delta, 75):+.4f}]"
        )

    if "m_scaled" in frame:
        usable = frame.loc[frame["m_scaled"].notna() & ~clipped, "m_scaled"]
        print(f"\nRecommended CROP_MARGIN (full-frame condition): {usable.median():.4f}")
    else:
        print(f"\nRecommended CROP_MARGIN: {clean_stats['median']:.4f}")


if __name__ == "__main__":
    main()
