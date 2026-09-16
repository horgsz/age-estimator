"""Evaluate the trained age estimator on the held-out UTKFace test split.

Reports overall MAE / CS@5, a per-decade MAE breakdown with support counts, and
writes a predicted-vs-true scatter plot. Also stamps the real test MAE into the
checkpoint's ``meta`` block so the published artifact matches the contract.

Two things to know before reusing these numbers downstream:

* The per-decade table conditions on **true** age, which is correct for judging
  a model but is *not* actionable by a UI, which can only threshold on the value
  it displays. The two views disagree sharply at the tails -- under the median
  decode the 80+ bias is -7.81 by true age but +1.10 by displayed age. See
  "The tail bias does not survive re-conditioning" in ``ml/README.md``.

  The general hazard, since this is where someone will be standing when they
  are tempted: a product rule conditioned on a quantity the *model outputs* can
  silently invert when the model changes, even though the evaluation below
  stayed correct throughout. Both -7.81 and +1.10 were always right; the bug
  would be wiring the wrong one to a UI threshold. Re-derive such thresholds
  from predicted-age conditioning after any decode or weight change.
* This script decodes with the soft expectation for continuity with the original
  spec. The shipped serving decode is the **median** (``AgeEstimator.median``),
  which scores better on every axis; see ``decode_compare.py``. So the figures
  here, and ``meta["test_mae"]``, are expectation-decode numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import UTKFaceDataset, load_split
from model import CHECKPOINT_PATH, load_checkpoint

REPORT_DIR = Path(__file__).resolve().parent / "reports"
SCATTER_PATH = REPORT_DIR / "scatter.png"
METRICS_PATH = REPORT_DIR / "metrics.json"

DECADE_LABELS = [
    "0-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80+",
]


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_predictions(
    model, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, stds, trues = [], [], []
    for images, ages in loader:
        images = images.to(device, non_blocking=True)
        pred, std = model.expectation(model(images))
        preds.append(pred.cpu().numpy())
        stds.append(std.cpu().numpy())
        trues.append(ages.numpy())
    return (
        np.concatenate(preds),
        np.concatenate(trues).astype(np.float64),
        np.concatenate(stds),
    )


def per_decade_table(true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    decade_idx = np.clip((true // 10).astype(int), 0, 8)
    abs_err = np.abs(pred - true)

    rows = []
    for idx, label in enumerate(DECADE_LABELS):
        mask = decade_idx == idx
        support = int(mask.sum())
        if support == 0:
            rows.append(
                {
                    "decade": label,
                    "support": 0,
                    "mae": float("nan"),
                    "cs5": float("nan"),
                    "mean_bias": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "decade": label,
                "support": support,
                "mae": float(abs_err[mask].mean()),
                "cs5": float(100.0 * (abs_err[mask] <= 5).mean()),
                "mean_bias": float((pred[mask] - true[mask]).mean()),
            }
        )
    return pd.DataFrame(rows)


def save_scatter(true: np.ndarray, pred: np.ndarray, path: Path, mae: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true, pred, s=8, alpha=0.35, edgecolors="none", color="#1f77b4")
    lo, hi = 0, 105
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="perfect prediction")
    ax.fill_between(
        [lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5], color="red", alpha=0.08,
        label="+/- 5 years",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True age")
    ax.set_ylabel("Predicted age")
    ax.set_title(f"UTKFace test set: predicted vs. true age (MAE {mae:.2f} years)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote scatter plot -> {path}")


def stamp_test_mae(checkpoint: Path, test_mae: float) -> None:
    """Rewrite meta['test_mae'] so the shipped artifact carries the real number."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["meta"]["test_mae"] = float(test_mae)
    torch.save(payload, checkpoint)
    print(f"Stamped meta.test_mae={test_mae:.4f} into {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the age estimator")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scatter", type=Path, default=SCATTER_PATH)
    parser.add_argument(
        "--no-stamp",
        action="store_true",
        help="do not write the measured MAE back into the checkpoint meta",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    model, meta = load_checkpoint(args.checkpoint)
    model.to(device)
    print(f"Device: {device}\nCheckpoint: {args.checkpoint}\nMeta: {meta}")

    frame = load_split(args.split)
    loader = DataLoader(
        UTKFaceDataset(frame, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    pred, true, std = collect_predictions(model, loader, device)
    abs_err = np.abs(pred - true)
    mae = float(abs_err.mean())
    cs5 = float(100.0 * (abs_err <= 5).mean())
    rmse = float(np.sqrt(((pred - true) ** 2).mean()))

    print(f"\n=== {args.split} set ({len(true)} images) ===")
    print(f"MAE      : {mae:.3f} years")
    print(f"CS@5     : {cs5:.2f}%")
    print(f"RMSE     : {rmse:.3f} years")
    print(f"Mean bias: {float((pred - true).mean()):+.3f} years")
    print(f"Mean predicted uncertainty (sigma): {float(std.mean()):.2f} years")

    table = per_decade_table(true, pred)
    print("\n=== MAE per age decade ===")
    print(f"{'decade':>8} {'support':>8} {'MAE':>8} {'CS@5':>8} {'bias':>8}")
    for row in table.itertuples(index=False):
        print(
            f"{row.decade:>8} {row.support:>8} {row.mae:>8.2f} "
            f"{row.cs5:>7.1f}% {row.mean_bias:>+8.2f}"
        )

    save_scatter(true, pred, args.scatter, mae)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "split": args.split,
                "n": int(len(true)),
                "mae": mae,
                "cs5": cs5,
                "rmse": rmse,
                "mean_bias": float((pred - true).mean()),
                "mean_sigma": float(std.mean()),
                "per_decade": table.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(f"Wrote metrics -> {METRICS_PATH}")

    if args.split == "test" and not args.no_stamp:
        stamp_test_mae(args.checkpoint, mae)


if __name__ == "__main__":
    main()
