"""Compare checkpoints on the real-ground-truth test split.

**The trap this script exists to avoid.** The shipped checkpoint's 8.52 MAE was
measured over all 7,534 detected APPA-REAL images, none of which it had ever
seen. The real-GT model trains on APPA-REAL's *train* split. Comparing those two
numbers would pit a fully-held-out score against a partially-seen one and would
flatter the new model for free.

So every number here is computed on the **test split only**, for both models,
through identical code. Per-source breakdowns are reported because the sources
differ in difficulty and in provenance quality, and a single pooled number would
let a gain on one hide a regression on another.

Also reported, and arguably the most diagnostic single number: the **regression
slope** of prediction against true age. The shipped model scores 0.817 against
real age versus 0.935 against apparent age -- it is a well-calibrated predictor
of how old a face *looks* and a compressed predictor of how old someone *is*. If
training on chronological labels fixes that compression, the slope moves toward
1.0. MAE alone would not distinguish "less compressed" from "less noisy".

Decode is re-measured for every checkpoint rather than inherited. The median
decision was made on a label-smoothed UTKFace model; carrying it across a
retrain without re-measuring is exactly the mistake this codebase has already
made once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import REPO_ROOT
from model import load_checkpoint
from realgt_data import (
    DEFAULT_DATASETS_ROOT,
    RealGTDataset,
    assert_no_subject_leakage,
    load_manifest,
    split_frame,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
DECADES = [(0, 10, "0-9"), (10, 20, "10-19"), (20, 30, "20-29"), (30, 40, "30-39"),
           (40, 50, "40-49"), (50, 60, "50-59"), (60, 70, "60-69"),
           (70, 80, "70-79"), (80, 200, "80+")]


def decode_all(logits: torch.Tensor) -> dict[str, np.ndarray]:
    """Expectation, median and mode from one set of logits."""
    probs = torch.softmax(logits.float(), dim=1)
    centers = torch.arange(probs.shape[1], dtype=probs.dtype, device=probs.device)
    cdf = probs.cumsum(dim=1)
    return {
        "expectation": (probs * centers).sum(dim=1).cpu().numpy(),
        "median": (cdf < 0.5).sum(dim=1).float().cpu().numpy(),
        "mode": probs.argmax(dim=1).float().cpu().numpy(),
    }


@torch.no_grad()
def predict(model, frame, datasets_root, device, batch_size, workers):
    loader = DataLoader(
        RealGTDataset(frame, train=False, datasets_root=datasets_root),
        batch_size=batch_size, shuffle=False, num_workers=workers,
    )
    chunks, ages = [], []
    for images, age in loader:
        chunks.append(model(images.to(device)).float().cpu())
        ages.append(age)
    logits = torch.cat(chunks)
    return decode_all(logits), torch.cat(ages).numpy().astype(float)


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    error = pred - truth
    slope, intercept = np.polyfit(truth, pred, 1)
    return {
        "n": int(len(truth)),
        "mae": float(np.abs(error).mean()),
        "cs5": float((np.abs(error) <= 5).mean() * 100),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "bias": float(error.mean()),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def band_table(pred: np.ndarray, truth: np.ndarray) -> list[dict]:
    rows = []
    for low, high, label in DECADES:
        mask = (truth >= low) & (truth < high)
        if not mask.any():
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare checkpoints on real GT")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="label=path pairs, e.g. shipped=checkpoints/age_model.pt")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    frame = load_manifest()
    assert_no_subject_leakage(frame)
    test = split_frame(frame, args.split)
    print(f"\n{args.split} split: {len(test)} images, "
          f"{test['subject_id'].nunique()} subjects")
    print("  by source: " + ", ".join(
        f"{s} {len(g)}" for s, g in test.groupby("source")
    ))

    results: dict = {"split": args.split, "n": int(len(test)), "models": {}}

    for spec in args.checkpoints:
        label, _, path = spec.partition("=")
        path = Path(path or label)
        model, meta = load_checkpoint(path, map_location=device)
        model.to(device).eval()

        print(f"\n{'=' * 66}\n{label}  ({path.name})")
        print(f"  meta.test_mae {meta.get('test_mae')}  "
              f"decode {meta.get('decode', 'unspecified')}")

        decoded, truth = predict(
            model, test, args.datasets_root, device, args.batch_size, args.workers
        )
        entry = {"path": str(path), "meta": {k: v for k, v in meta.items()
                                            if k != "state_dict"}}

        print(f"\n  {'decode':<12}{'MAE':>8}{'CS@5':>8}{'bias':>9}{'slope':>8}")
        for name, pred in decoded.items():
            stats = score(pred, truth)
            entry[name] = stats
            print(f"  {name:<12}{stats['mae']:>8.3f}{stats['cs5']:>7.1f}%"
                  f"{stats['bias']:>+9.3f}{stats['slope']:>8.3f}")

        # Per-source, under every decode, so a pooled gain cannot hide a
        # per-source regression.
        entry["by_source"] = {}
        print(f"\n  {'source':<12}{'n':>6}" + "".join(
            f"{d[:4]:>9}" for d in decoded))
        for source, group in test.groupby("source"):
            mask = (test["source"] == source).to_numpy()
            cells, per = "", {}
            for name, pred in decoded.items():
                s = score(pred[mask], truth[mask])
                per[name] = s
                cells += f"{s['mae']:>9.3f}"
            entry["by_source"][source] = per
            print(f"  {source:<12}{int(mask.sum()):>6}{cells}")

        best = min(decoded, key=lambda d: score(decoded[d], truth)["mae"])
        entry["best_decode"] = best
        entry["by_decade"] = {d: band_table(p, truth) for d, p in decoded.items()}
        print(f"\n  best decode on this split: {best}")

        print(f"\n  per-decade ({best} decode)")
        print(f"  {'band':<8}{'n':>7}{'MAE':>9}{'bias':>9}{'CS@5':>8}")
        for row in entry["by_decade"][best]:
            print(f"  {row['band']:<8}{row['n']:>7}{row['mae']:>9.2f}"
                  f"{row['bias']:>+9.2f}{row['cs5']:>7.1f}%")

        results["models"][label] = entry

    out = (args.out or REPORT_DIR / f"realgt_comparison_{args.split}.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"\nWrote {shown}")


if __name__ == "__main__":
    main()
