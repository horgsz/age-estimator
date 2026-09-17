"""Detect faces and write model-ready square crops for every manifest row.

The crop geometry here is *not* a reimplementation: it comes from
``server/preprocessing.py`` when that package is importable, and from a
faithful mirror in ``datasets/common.py`` otherwise. Margin is fixed at
``CROP_MARGIN = 0.0``, the value two independent measured sweeps agreed on.
Training on any other framing than serving uses is a silent accuracy leak, not
a visible error, which is why this script refuses to invent its own geometry.

Multiple detections
-------------------
APPA-REAL and FG-NET originals are full scenes and often contain more than one
face; the label only describes one of them. The largest detection is taken,
which is the subject in the overwhelming majority of portrait-style photos. Rows
where detection fails keep ``face_detected=False`` and get no crop, and the
per-source failure rate is reported so a bad source is visible rather than
quietly degrading the corpus.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import (
    CROP_MARGIN,
    CROPS_DIR,
    INPUT_SIZE,
    MANIFEST_PATH,
    REPO_ROOT,
    crop_backend,
    crop_face,
    resize_crop,
)

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    f"face_detection_yunet/{YUNET_FILENAME}",
    f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_FILENAME}",
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# server/config.py defaults.
DETECT_SCORE_THRESHOLD = 0.6
DETECT_NMS_THRESHOLD = 0.3
DETECT_TOP_K = 5000
MAX_DETECT_SIDE = 1024

# Scanned film, heavy grain and B&W all depress YuNet's confidence. Rather than
# discard those images -- FG-NET is the corpus's main source of under-12 faces,
# so losing it defeats the point -- retry at a lower threshold and record which
# pass succeeded, so the weaker detections stay auditable.
RESCUE_SCORE_THRESHOLD = 0.25

CROP_JPEG_QUALITY = 95


def _find_yunet() -> Path:
    """Locate the YuNet ONNX weights, reusing the server's copy when present."""
    candidates = [
        REPO_ROOT / "server" / "models" / YUNET_FILENAME,
        CROPS_DIR.parent / "models" / YUNET_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 50_000:
            return candidate

    target = candidates[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    import hashlib
    import subprocess

    for url in YUNET_URLS:
        try:
            # curl rather than urllib: on a machine with a TLS-intercepting
            # proxy the system keychain has the root CA and Python's bundle
            # does not. server/detector.py hits the same wall.
            subprocess.run(
                ["curl", "-fsSL", "--retry", "2", "--max-time", "120", "-o", str(target), url],
                check=True,
                capture_output=True,
            )
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != YUNET_SHA256:
                raise RuntimeError(f"checksum mismatch: expected {YUNET_SHA256}, got {digest}")
            return target
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  YuNet download failed ({url}): {exc}")
            target.unlink(missing_ok=True)

    raise RuntimeError(f"Could not obtain {YUNET_FILENAME}; place it in {target.parent}")


class Detector:
    """YuNet wrapper: detect on a downscaled copy, return original-image boxes."""

    def __init__(self, model_path: Path, score_threshold: float) -> None:
        self.score_threshold = score_threshold
        self._net = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=DETECT_NMS_THRESHOLD,
            top_k=DETECT_TOP_K,
        )

    def detect(self, image_bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        img_h, img_w = image_bgr.shape[:2]
        scale = min(1.0, MAX_DETECT_SIDE / float(max(img_h, img_w)))
        if scale < 1.0:
            det_w = max(1, int(round(img_w * scale)))
            det_h = max(1, int(round(img_h * scale)))
            det_img = cv2.resize(image_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_w, det_h = img_w, img_h
            det_img = image_bgr

        self._net.setInputSize((det_w, det_h))
        _, faces = self._net.detect(det_img)
        if faces is None:
            return []

        inv_x = img_w / float(det_w)
        inv_y = img_h / float(det_h)

        out = []
        for face in faces:
            x, y, w, h = (float(v) for v in face[:4])
            score = float(face[-1])
            bx = int(round(x * inv_x))
            by = int(round(y * inv_y))
            bw = int(round(w * inv_x))
            bh = int(round(h * inv_y))
            bx = min(max(bx, 0), max(0, img_w - 1))
            by = min(max(by, 0), max(0, img_h - 1))
            bw = min(bw, img_w - bx)
            bh = min(bh, img_h - by)
            if bw >= 2 and bh >= 2:
                out.append(((bx, by, bw, bh), score))
        return out


def process(
    manifest: pd.DataFrame,
    crops_dir: Path,
    margin: float,
    overwrite: bool,
    limit: int | None = None,
) -> pd.DataFrame:
    primary = Detector(_find_yunet(), DETECT_SCORE_THRESHOLD)
    rescue = Detector(_find_yunet(), RESCUE_SCORE_THRESHOLD)

    frame = manifest if limit is None else manifest.head(limit)
    total = len(frame)

    crop_paths: list[object] = []
    detected: list[bool] = []
    det_passes: list[str] = []
    stats: Counter[str] = Counter()

    for counter, record in enumerate(frame.itertuples(index=False), start=1):
        source = record.source
        src_path = REPO_ROOT / record.path
        out_path = crops_dir / source / (Path(record.path).stem + ".jpg")
        stats[f"{source}:total"] += 1

        if out_path.is_file() and not overwrite:
            crop_paths.append(out_path.relative_to(REPO_ROOT).as_posix())
            detected.append(True)
            det_passes.append("cached")
            stats[f"{source}:cached"] += 1
            continue

        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            crop_paths.append(pd.NA)
            detected.append(False)
            det_passes.append("unreadable")
            stats[f"{source}:unreadable"] += 1
            continue

        boxes = primary.detect(image)
        pass_name = "primary"
        if not boxes:
            boxes = rescue.detect(image)
            pass_name = "rescue"

        if not boxes:
            crop_paths.append(pd.NA)
            detected.append(False)
            det_passes.append("none")
            stats[f"{source}:failed"] += 1
            continue

        # Largest box: the labelled subject dominates the frame in portraits.
        bbox, _score = max(boxes, key=lambda item: item[0][2] * item[0][3])
        crop = resize_crop(crop_face(image, bbox, margin), INPUT_SIZE)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), CROP_JPEG_QUALITY])

        crop_paths.append(out_path.relative_to(REPO_ROOT).as_posix())
        detected.append(True)
        det_passes.append(pass_name)
        stats[f"{source}:{pass_name}"] += 1

        if counter % 500 == 0 or counter == total:
            print(f"  {counter}/{total}", flush=True)

    out = frame.copy()
    out["crop_path"] = crop_paths
    out["face_detected"] = detected
    out["detect_pass"] = det_passes
    _report(stats)
    return out


def _report(stats: Counter[str]) -> None:
    sources = sorted({key.split(":", 1)[0] for key in stats})
    print("\n=== Detection results ===")
    header = f"{'source':<12} {'images':>8} {'primary':>8} {'rescue':>8} {'failed':>8} {'fail %':>8}"
    print(header)
    print("-" * len(header))
    for source in sources:
        total = stats[f"{source}:total"]
        failed = stats[f"{source}:failed"] + stats[f"{source}:unreadable"]
        pct = 100.0 * failed / total if total else 0.0
        print(
            f"{source:<12} {total:>8} {stats[f'{source}:primary']:>8} "
            f"{stats[f'{source}:rescue']:>8} {failed:>8} {pct:>7.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--crops-dir", type=Path, default=CROPS_DIR)
    parser.add_argument("--sources", nargs="*", default=None, help="Restrict to these sources.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--margin",
        type=float,
        default=CROP_MARGIN,
        help="Override the crop margin. Do not, unless you have re-run the sweeps.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.sources:
        manifest = manifest[manifest["source"].isin(args.sources)].reset_index(drop=True)

    if args.margin != CROP_MARGIN:
        print(
            f"WARNING: margin {args.margin} differs from the measured optimum {CROP_MARGIN}. "
            "Serving crops at 0.0; training elsewhere costs accuracy invisibly.",
            file=sys.stderr,
        )

    print(f"Crop geometry backend: {crop_backend()} (margin={args.margin})")
    processed = process(manifest, args.crops_dir, args.margin, args.overwrite, args.limit)

    if args.sources or args.limit:
        # Partial run: merge back into the full manifest rather than truncating it.
        full = pd.read_csv(args.manifest)
        for column in ("crop_path", "face_detected", "detect_pass"):
            if column not in full.columns:
                full[column] = pd.NA
            full.loc[full["path"].isin(processed["path"]), column] = (
                full.loc[full["path"].isin(processed["path"]), "path"]
                .map(processed.set_index("path")[column])
                .values
            )
        processed = full

    processed.to_csv(args.manifest, index=False)
    print(f"\nUpdated {args.manifest}")


if __name__ == "__main__":
    main()
