"""Error conditioned on *predicted* age, for UI thresholds.

Every per-decade table in this repo bins by **true** age. That is the correct
conditioning for evaluating a model: it answers "given a face of age A, how
wrong are we?".

It is the wrong conditioning for a user interface, which cannot observe the true
age. A UI rule can only ever threshold on the number the model *outputs*, so the
quantity it needs is ``E[|err| | predicted]``, not ``E[|err| | true]``.

These two disagree, and under a compressed regressor they can inverse the
ranking of which band is worst. On the previous shipped model the 80+ band had
bias -7.81 conditioned on true age but **+1.10** conditioned on displayed age --
because an 85-year-old was displayed as ~68, so no threshold on the displayed
value ever selected the population the caveat was written about. A caveat keyed
to the true-age number fired on the wrong people, and the evaluation was correct
the entire time. The bug was in which number got wired to product logic.

This script emits the predicted-conditioned view alongside the true-conditioned
one so the two can be compared directly, and dumps per-sample rows so a consumer
can re-bin against its own thresholds rather than trusting ours.

Read-only: loads checkpoints, writes only to ml/reports/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import load_checkpoint
from realgt_compare import DEFAULT_DATASETS_ROOT, REPORT_DIR, predict, score
from realgt_data import load_manifest


def bands_by(
    pred: np.ndarray, truth: np.ndarray, key: np.ndarray, edges: list[tuple]
) -> list[dict]:
    """Bin by ``key`` (either true or predicted age) and score within each bin."""
    rows = []
    for low, high, label in edges:
        mask = (key >= low) & (key < high)
        if mask.sum() == 0:
            continue
        error = pred[mask] - truth[mask]
        rows.append({
            "band": label,
            "n": int(mask.sum()),
            "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()),
            "cs5": float((np.abs(error) <= 5).mean() * 100),
        })
    return rows


def threshold_analysis(pred: np.ndarray, truth: np.ndarray) -> list[dict]:
    """Both error rates at a hard threshold, because they answer different questions.

    Neither band table above surfaces this, and the two rates diverge sharply
    because of base rates. For an age gate at T:

    * ``minors_shown_as_adult`` -- of people truly under T, what fraction does
      the model display at or above T? This is the *safety* question, and it is
      conditioned on true age.
    * ``shown_adult_actually_minor`` -- of people displayed at or above T, what
      fraction are truly under? This is what a UI can observe, and it is
      conditioned on the prediction.

    Adults vastly outnumber minors in this corpus, so a small rate on the second
    corresponds to a large rate on the first. Reading only the observable one
    makes a gate look far safer than it is.
    """
    rows = []
    for thr in (13, 16, 18, 21):
        minors = truth < thr
        shown = pred >= thr
        rows.append({
            "threshold": thr,
            "n_true_minors": int(minors.sum()),
            "minors_shown_as_adult_pct": (
                float((pred[minors] >= thr).mean() * 100) if minors.any() else None
            ),
            "n_shown_adult": int(shown.sum()),
            "shown_adult_actually_minor_pct": (
                float((truth[shown] < thr).mean() * 100) if shown.any() else None
            ),
        })
    return rows


DECADES = [
    (0, 10, "0-9"), (10, 20, "10-19"), (20, 30, "20-29"), (30, 40, "30-39"),
    (40, 50, "40-49"), (50, 60, "50-59"), (60, 70, "60-69"), (70, 80, "70-79"),
    (80, 200, "80+"),
]

# Thresholds a UI would plausibly key on, rather than tidy decades.
UI_BANDS = [
    (0, 13, "under 13"), (13, 18, "13-17"), (18, 25, "18-24"),
    (25, 40, "25-39"), (40, 55, "40-54"), (55, 65, "55-64"),
    (65, 200, "65+"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="name=path pairs")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--decode", default="median",
                        choices=["expectation", "median", "mode"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "predicted_age_bands.json")
    parser.add_argument("--dump-predictions", action="store_true",
                        help="also write per-sample CSVs for external re-binning")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    frame = load_manifest()
    frame = frame[frame["split"] == args.split].reset_index(drop=True)
    print(f"{args.split} split: {len(frame)} images  decode={args.decode}")

    report: dict = {
        "split": args.split, "n": int(len(frame)), "decode": args.decode,
        "conditioning_note": (
            "by_true_age is correct for model evaluation; by_predicted_age is "
            "what a UI can threshold on. They can rank bands differently."
        ),
        "models": {},
    }

    for spec in args.checkpoints:
        name, _, path = spec.partition("=")
        model, meta = load_checkpoint(Path(path))
        model.to(device).eval()
        decoded, truth = predict(
            model, frame, args.datasets_root, device, args.batch_size, args.workers
        )
        pred = decoded[args.decode]

        entry = {
            "path": path,
            "overall": score(pred, truth),
            "by_true_age": bands_by(pred, truth, truth, DECADES),
            "by_predicted_age": bands_by(pred, truth, pred, DECADES),
            "ui_bands_by_predicted_age": bands_by(pred, truth, pred, UI_BANDS),
            "thresholds": threshold_analysis(pred, truth),
        }
        report["models"][name] = entry

        print(f"\n{'='*66}\n{name}  (overall MAE {entry['overall']['mae']:.3f})")
        print(f"{'band':10s} {'n(true)':>8s} {'MAE':>7s} {'bias':>7s} | "
              f"{'n(pred)':>8s} {'MAE':>7s} {'bias':>7s}")
        by_t = {r["band"]: r for r in entry["by_true_age"]}
        by_p = {r["band"]: r for r in entry["by_predicted_age"]}
        for _, _, label in DECADES:
            t, p = by_t.get(label), by_p.get(label)
            tpart = (f"{t['n']:8d} {t['mae']:7.2f} {t['bias']:+7.2f}" if t
                     else f"{'-':>8s} {'-':>7s} {'-':>7s}")
            ppart = (f"{p['n']:8d} {p['mae']:7.2f} {p['bias']:+7.2f}" if p
                     else f"{'-':>8s} {'-':>7s} {'-':>7s}")
            print(f"{label:10s} {tpart} | {ppart}")

        print(f"\n  UI bands, conditioned on DISPLAYED age:")
        for r in entry["ui_bands_by_predicted_age"]:
            print(f"    {r['band']:10s} n={r['n']:5d}  MAE {r['mae']:6.2f}  "
                  f"bias {r['bias']:+6.2f}  CS@5 {r['cs5']:5.1f}%")

        print("\n  hard thresholds (both directions):")
        for r in entry["thresholds"]:
            print(f"    >={r['threshold']:3d}  of true minors shown as adult: "
                  f"{r['minors_shown_as_adult_pct']:5.1f}%  |  of shown-adult "
                  f"actually minor: {r['shown_adult_actually_minor_pct']:5.1f}%")

        if args.dump_predictions:
            csv_path = args.out.parent / f"preds_{name}_{args.split}_{args.decode}.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["true_age", "predicted", "error"])
                for tr, pr in zip(truth, pred):
                    writer.writerow([int(tr), float(pr), float(pr - tr)])
            print(f"  per-sample -> {csv_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
