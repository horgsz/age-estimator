"""Offline end-to-end evaluation of the *deployed* inference path.

This measures something different from ``ml/eval.py``, and the difference is the
whole point:

* ``ml/eval.py`` scores the model on **training-framed** crops. It answers
  "did the model learn?".
* This tool scores the model on **YuNet-detected, server-cropped** images --
  detect -> :mod:`server.preprocessing` crop -> model -- i.e. exactly what a
  user gets. It answers "does the deployed system work?".

The gap between the two numbers *is* the crop mismatch, quantified. If it is
large, fix ``CROP_MARGIN`` and re-measure rather than shipping it.

``--margins`` sweeps several crop margins in one run (detection happens once per
image and is reused), which is the cheap way to find the margin that best
matches training-time framing.

Usage::

    python -m server.tools.eval_end_to_end \\
        --csv ml/splits/test.csv \\
        --checkpoint checkpoints/age_model.pt \\
        --margins 0.2 0.3 0.4 0.5

    # UTKFace-style directory instead of a CSV (ages parsed from filenames)
    python -m server.tools.eval_end_to_end --dir data/UTKFace --checkpoint ...

This tool only *reads* from ``ml/``; it never writes there.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import config, preprocessing
from ..detector import FaceDetector
from ..predictor import AgePredictor, load_predictor

log = logging.getLogger("eval_end_to_end")

PATH_COLUMNS = ("path", "filepath", "file_path", "file", "image", "image_path", "filename")
AGE_COLUMNS = ("age", "true_age", "label", "target", "y")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BBox = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# sample loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    path: Path
    age: float


class DataError(RuntimeError):
    """Raised for user-facing problems with the requested dataset."""


def _resolve(raw: str, roots: list[Path]) -> Path | None:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for root in roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return None


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_samples_from_csv(csv_path: Path, extra_roots: list[Path] | None = None) -> list[Sample]:
    """Read ``path,age`` rows, tolerating column-name and header variations."""
    if not csv_path.exists():
        raise DataError(
            f"{csv_path} does not exist.\n"
            "The ml/ session writes ml/splits/test.csv; run this once that file "
            "is available, or use --dir to point at a UTKFace-style directory."
        )

    roots = [csv_path.resolve().parent, config.REPO_ROOT, Path.cwd()]
    roots.extend(extra_roots or [])

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        sample_text = fh.read(8192)
        fh.seek(0)
        if not sample_text.strip():
            raise DataError(f"{csv_path} is empty.")

        try:
            has_header = csv.Sniffer().has_header(sample_text)
        except csv.Error:
            has_header = True

        samples: list[Sample] = []
        skipped = 0

        if has_header:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            path_col = _pick_column(fieldnames, PATH_COLUMNS)
            age_col = _pick_column(fieldnames, AGE_COLUMNS)
            if path_col is None or age_col is None:
                raise DataError(
                    f"{csv_path} has columns {fieldnames}; expected a path column "
                    f"(one of {PATH_COLUMNS}) and an age column (one of {AGE_COLUMNS})."
                )
            rows = ((row.get(path_col), row.get(age_col)) for row in reader)
        else:
            plain = csv.reader(fh)
            rows = ((r[0], r[1]) if len(r) >= 2 else (None, None) for r in plain)

        for raw_path, raw_age in rows:
            if not raw_path or raw_age is None or str(raw_age).strip() == "":
                skipped += 1
                continue
            try:
                age = float(str(raw_age).strip())
            except ValueError:
                skipped += 1
                continue
            resolved = _resolve(str(raw_path).strip(), roots)
            if resolved is None:
                skipped += 1
                continue
            samples.append(Sample(resolved, age))

    if not samples:
        raise DataError(
            f"No usable rows in {csv_path} (skipped {skipped}). "
            "Check that the image paths in the CSV resolve from "
            f"{csv_path.parent} or the repo root."
        )
    if skipped:
        log.warning("Skipped %d unusable/unresolvable row(s) from %s", skipped, csv_path)
    return samples


def load_samples_from_dir(directory: Path) -> list[Sample]:
    """UTKFace filename convention: ``<age>_<gender>_<race>_<datetime>.jpg``."""
    if not directory.is_dir():
        raise DataError(f"{directory} is not a directory.")

    samples: list[Sample] = []
    skipped = 0
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        head = path.name.split("_", 1)[0]
        try:
            age = float(head)
        except ValueError:
            skipped += 1
            continue
        if 0 <= age <= 120:
            samples.append(Sample(path, age))
        else:
            skipped += 1

    if not samples:
        raise DataError(
            f"No UTKFace-style images under {directory} "
            "(expected filenames like 25_0_2_20170116174525125.jpg)."
        )
    if skipped:
        log.warning("Skipped %d file(s) with unparseable ages under %s", skipped, directory)
    return samples


def subsample(samples: list[Sample], limit: int, head: bool = False) -> list[Sample]:
    """Take at most ``limit`` samples, spread evenly across the set.

    This deliberately does *not* default to ``samples[:limit]``. Split CSVs are
    typically sorted by path, and UTKFace paths begin with the age, so they sort
    lexicographically as 1, 10, 100, 11, 12, ... A head slice of ml/splits/test.csv
    therefore contains almost nothing over 40, which silently turns any margin
    sweep into a measurement of the model's child bias rather than of framing.
    That produced a confidently wrong "negative margins are better" result once
    already; an even stride keeps the age distribution representative.
    """
    if limit <= 0 or limit >= len(samples):
        return samples
    if head:
        return samples[:limit]
    stride = len(samples) / limit
    return [samples[int(i * stride)] for i in range(limit)]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def select_primary_box(boxes: list[BBox]) -> BBox | None:
    """Pick the largest detection — the subject, when there are false positives."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


@dataclass
class MarginStats:
    """Accumulated errors for one crop margin."""

    margin: float
    errors: list[float] = field(default_factory=list)
    covered: int = 0  # true age fell inside the predicted [low, high]
    # (path, true, predicted, low, high, confidence) per sample. Kept so error
    # can be binned by *predicted* age, which is the only thing a UI threshold
    # can act on — the model compresses both tails, so a threshold that looks
    # right against true ages fires far too late against displayed ones.
    #
    # `path` is carried so a dump can be re-joined to labels it was not run
    # against — notably APPA-REAL's apparent ages, needed to judge a model that
    # claims to predict how old a face *looks* rather than how old it is.
    # Without it the only way to align two runs is to assume they dropped the
    # same undetected rows in the same order, which is true but implicit, and
    # the kind of assumption that has silently broken a comparison here before.
    samples: list[tuple[str, float, float, float, float, float]] = field(default_factory=list)

    def add(
        self,
        predicted: float,
        true_age: float,
        low: float,
        high: float,
        confidence: float = 0.0,
        path: str = "",
    ) -> None:
        self.errors.append(predicted - true_age)
        self.samples.append((path, true_age, predicted, low, high, confidence))
        if low <= true_age <= high:
            self.covered += 1

    def summary(self) -> dict[str, float | int | None]:
        n = len(self.errors)
        if n == 0:
            return {"margin": self.margin, "n": 0}
        absolute = [abs(e) for e in self.errors]
        return {
            "margin": round(self.margin, 4),
            "n": n,
            "mae": round(sum(absolute) / n, 3),
            "median_ae": round(statistics.median(absolute), 3),
            "rmse": round(math.sqrt(sum(e * e for e in self.errors) / n), 3),
            "bias": round(sum(self.errors) / n, 3),
            "within_5": round(sum(a <= 5 for a in absolute) / n, 4),
            "within_10": round(sum(a <= 10 for a in absolute) / n, 4),
            "interval_coverage": round(self.covered / n, 4),
        }


@dataclass
class RunReport:
    total: int = 0
    read_errors: int = 0
    no_detection: int = 0
    multi_detection: int = 0
    evaluated: int = 0
    margins: list[dict] = field(default_factory=list)
    # Per-margin raw accumulators, for --dump-predictions. Excluded from
    # to_dict() so the JSON report stays small.
    stats: list["MarginStats"] = field(default_factory=list, repr=False)
    model: str = "unknown"
    stub: bool = True
    # Which artifact produced these numbers. Recorded because the checkpoint was
    # once republished to the same path mid-evaluation, making two runs silently
    # incomparable.
    checkpoint: dict | None = None
    seconds: float = 0.0

    @property
    def detection_rate(self) -> float:
        readable = self.total - self.read_errors
        return self.evaluated / readable if readable else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "stub": self.stub,
            "checkpoint": self.checkpoint,
            "total_samples": self.total,
            "read_errors": self.read_errors,
            "no_detection": self.no_detection,
            "multi_detection": self.multi_detection,
            "evaluated": self.evaluated,
            "detection_rate": round(self.detection_rate, 4),
            "seconds": round(self.seconds, 1),
            "margins": self.margins,
        }


def evaluate(
    samples: list[Sample],
    predictor: AgePredictor,
    margins: list[float],
    detector: FaceDetector | None = None,
    save_crops: Path | None = None,
    save_crops_limit: int = 12,
    progress_every: int = 200,
) -> RunReport:
    """Run the full detect -> crop -> model path over ``samples``."""
    detector = detector or predictor.detector
    stats = {m: MarginStats(m) for m in margins}
    report = RunReport(
        total=len(samples),
        model=predictor.model_name,
        stub=predictor.is_stub,
        checkpoint=predictor.describe_checkpoint(),
    )
    started = time.monotonic()
    saved = 0

    for index, sample in enumerate(samples, start=1):
        frame = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if frame is None:
            report.read_errors += 1
            continue

        # Detect once; every margin reuses the same box, so the sweep isolates
        # the effect of the crop rather than mixing in detector jitter.
        boxes = detector.detect(frame)
        box = select_primary_box(boxes)
        if box is None:
            report.no_detection += 1
            continue
        if len(boxes) > 1:
            report.multi_detection += 1
        report.evaluated += 1

        for margin in margins:
            # predict_boxes is the same crop+model path the HTTP server uses.
            results = predictor.predict_boxes(frame, [box], margin)
            if not results:
                continue
            face = results[0]
            stats[margin].add(
                face.age, sample.age, face.low, face.high, face.confidence, str(sample.path)
            )

            if save_crops is not None and saved < save_crops_limit * len(margins):
                out_dir = save_crops / f"margin_{margin:.4f}"
                out_dir.mkdir(parents=True, exist_ok=True)
                crop = preprocessing.resize_crop(
                    preprocessing.crop_face(frame, box, margin)
                )
                cv2.imwrite(str(out_dir / f"{index:05d}_{sample.path.stem}.jpg"), crop)
                saved += 1

        if progress_every and index % progress_every == 0:
            log.info("… %d/%d images", index, len(samples))

    report.seconds = time.monotonic() - started
    report.margins = [stats[m].summary() for m in margins]
    report.stats = [stats[m] for m in margins]
    return report


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_report(report: RunReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("")
    add("End-to-end evaluation (YuNet detect -> server crop -> model)")
    add("=" * 78)
    add(f"model              : {report.model}{'  [STUB - ages are fake]' if report.stub else ''}")
    if report.checkpoint:
        ck = report.checkpoint
        add(
            f"checkpoint         : sha256:{ck.get('sha256')}  "
            f"recorded MAE {ck.get('recorded_test_mae')} "
            f"({ck.get('recorded_test_mae_decode')} decode)"
        )
        add(f"                     {ck.get('path')}")
        # A recorded MAE is meaningless without the corpus it was measured on:
        # 5.5472 against DEX-estimated apparent age and 6.393 against real
        # chronological age are not comparable, and the smaller is the weaker
        # result. Print them together or not at all.
        if ck.get("recorded_test_mae_corpus"):
            add(f"                     corpus: {ck['recorded_test_mae_corpus']}")
        if ck.get("label_semantics"):
            add(f"                     labels: {ck['label_semantics']}")
        if ck.get("role"):
            add(f"                     ROLE: {ck['role']}")
        if ck.get("crop_margin_matches_training") is False:
            add(
                f"                     WARNING: trained at crop_margin "
                f"{ck.get('trained_crop_margin')}, serving "
                f"{ck.get('serving_crop_margin')}"
            )
    add(f"samples            : {report.total}")
    add(f"unreadable files   : {report.read_errors}")
    add(f"no face detected   : {report.no_detection}")
    add(f"multi-face images  : {report.multi_detection} (largest box used)")
    add(f"evaluated          : {report.evaluated}  (detection rate {report.detection_rate:.1%})")
    add(f"elapsed            : {report.seconds:.1f}s")
    add("")

    header = (
        f"{'margin':>7}  {'n':>6}  {'MAE':>7}  {'medAE':>7}  {'RMSE':>7}  "
        f"{'bias':>7}  {'±5yr':>7}  {'±10yr':>7}  {'cover':>7}"
    )
    add(header)
    add("-" * len(header))
    for row in report.margins:
        if not row.get("n"):
            add(f"{row['margin']:>7.4f}  {'0':>6}  (nothing evaluated)")
            continue
        add(
            f"{row['margin']:>7.4f}  {row['n']:>6}  {row['mae']:>7.3f}  "
            f"{row['median_ae']:>7.3f}  {row['rmse']:>7.3f}  {row['bias']:>+7.3f}  "
            f"{row['within_5']:>6.1%}  {row['within_10']:>6.1%}  "
            f"{row['interval_coverage']:>6.1%}"
        )
    add("")

    scored = [r for r in report.margins if r.get("n")]
    if scored:
        best = min(scored, key=lambda r: r["mae"])
        add(f"best margin by MAE : {best['margin']:.4f}  (MAE {best['mae']:.3f} years)")
        if len(scored) > 1:
            worst = max(scored, key=lambda r: r["mae"])
            add(
                f"spread across swept margins: {worst['mae'] - best['mae']:.3f} years "
                "— if this is large, the crop really matters."
            )
        add("")
        add(
            "Compare the best MAE above with ml/eval.py's MAE on training-framed\n"
            "crops. A large gap is crop mismatch, not model quality: change\n"
            "CROP_MARGIN (env var, no code edit) and re-run."
        )
    if report.stub:
        add("")
        add("!! These numbers are meaningless: the stub predictor returns hashed")
        add("!! fake ages. Pass --checkpoint to evaluate a real model.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m server.tools.eval_end_to_end",
        description=(
            "Measure MAE of the deployed inference path (YuNet detect -> server "
            "crop -> model) and optionally sweep crop margins."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV of image paths and true ages")
    source.add_argument("--dir", type=Path, help="Directory of UTKFace-named images")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint (default: AGE_MODEL_PATH, else the stub)",
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=None,
        help=f"Crop margins to sweep (default: the active CROP_MARGIN, {config.CROP_MARGIN})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most N images, sampled evenly across the set (see --limit-head)",
    )
    parser.add_argument(
        "--limit-head",
        action="store_true",
        help="Make --limit take the first N rows instead of an even spread (rarely what you want)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        default=None,
        help="Extra root for resolving relative CSV paths (repeatable)",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the report as JSON")
    parser.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="Write per-sample predictions as CSV (margin, true, predicted, low, high, confidence)",
    )
    parser.add_argument(
        "--save-crops",
        type=Path,
        default=None,
        help="Write a few example crops per margin here, for eyeballing framing",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    # Same startup refresh the server does, so CROP_MARGIN et al. come from the
    # environment here too.
    config.reload_from_env()

    try:
        samples = (
            load_samples_from_csv(args.csv, args.image_root)
            if args.csv
            else load_samples_from_dir(args.dir)
        )
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.limit is not None:
        samples = subsample(samples, args.limit, head=args.limit_head)

    margins = args.margins if args.margins else [config.CROP_MARGIN]
    for margin in margins:
        if not config.MIN_CROP_MARGIN <= margin <= config.MAX_CROP_MARGIN:
            print(
                f"error: margins must be within "
                f"[{config.MIN_CROP_MARGIN}, {config.MAX_CROP_MARGIN}]",
                file=sys.stderr,
            )
            return 2

    checkpoint = str(args.checkpoint) if args.checkpoint else None
    predictor = load_predictor(checkpoint)

    log.info(
        "Evaluating %d image(s) with model=%s over margin(s) %s",
        len(samples),
        predictor.model_name,
        ", ".join(f"{m:.4f}" for m in margins),
    )

    report = evaluate(samples, predictor, margins, save_crops=args.save_crops)
    print(format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        log.info("Wrote %s", args.json)

    if args.dump_predictions:
        args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_predictions.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["margin", "path", "true_age", "predicted", "low", "high", "confidence"]
            )
            for st in report.stats:
                for path, true_age, predicted, low, high, conf in st.samples:
                    writer.writerow(
                        [
                            f"{st.margin:.4f}",
                            path,
                            f"{true_age:.2f}",
                            f"{predicted:.4f}",
                            f"{low:.4f}",
                            f"{high:.4f}",
                            f"{conf:.4f}",
                        ]
                    )
        log.info("Wrote %s", args.dump_predictions)

    return 0 if report.evaluated else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
