"""Decompose the external-validation error into model, perception and label terms.

``external_eval.py`` measures *how much* we are wrong on APPA-REAL and FG-NET.
This answers the harder question: *what kind* of wrong.

The motivating problem is that our headline in-corpus MAE is agreement with
UTKFace's DEX-generated labels, which are apparent-age estimates. Scored against
chronological age on an external corpus the number gets worse -- but "worse"
could mean the model is weak, or it could mean chronological age is not what the
model (or any observer) can read off a face. APPA-REAL separates these because
it ships both targets for the same image.

Four analyses, in increasing order of how much they change the conclusion:

1. **Correlation** of our error-vs-real with the human apparent-vs-real gap on
   the same faces. If we err in the direction humans err, our error is partly
   perceptual rather than arbitrary.
2. **Regression slope** of prediction against each target. A slope near 1.0
   indicates the target is tracked without scale compression; this turns out to
   discriminate the two targets much more sharply than MAE does.
3. **A fair human baseline.** The often-quoted 4.12-year human gap is the mean
   of ~34 raters, which averages rater noise away. One model is properly
   compared against *one* rater, estimated by re-injecting the inter-rater
   spread that APPA-REAL publishes per image.
4. **FG-NET quality confounds.** FG-NET is scanned film, much of it black and
   white. Grayscale is separated from age reasoning by matching on age band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data import REPO_ROOT

REPORT_DIR = Path(__file__).resolve().parent / "reports"
APPA_ROOT = REPO_ROOT / "data" / "appa-real-release"
DECADES = [(0, 10, "0-9"), (10, 20, "10-19"), (20, 30, "20-29"), (30, 40, "30-39"),
           (40, 50, "40-49"), (50, 60, "50-59"), (60, 70, "60-69"),
           (70, 80, "70-79"), (80, 200, "80+")]


def is_grayscale(path: str, tol: float = 2.0) -> bool | None:
    """True when the colour channels are near-identical, i.e. a B&W scan."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    b, g, r = (image[:, :, i].astype(np.int16) for i in range(3))
    return bool(np.abs(b - g).mean() < tol and np.abs(g - r).mean() < tol)


def appa_analysis(dump: Path, seed: int = 42) -> dict:
    data = pd.read_csv(dump)
    data = data[
        data["detected"] & data["real_age"].notna() & data["apparent_age"].notna()
    ].copy()
    data["model_err"] = data["predicted"] - data["real_age"]
    data["human_err"] = data["apparent_age"] - data["real_age"]
    data["vs_apparent"] = data["predicted"] - data["apparent_age"]

    out: dict = {"n": int(len(data))}

    # 1. Shared direction with human misperception.
    corr = float(np.corrcoef(data["model_err"], data["human_err"])[0, 1])
    out["error_correlation_with_human_gap"] = corr
    out["shared_variance_pct"] = float(corr**2 * 100)

    # 2. Which target is actually tracked.
    out["regression"] = {}
    for target in ("real_age", "apparent_age"):
        x = data[target].to_numpy(dtype=float)
        y = data["predicted"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        out["regression"][target] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(np.corrcoef(x, y)[0, 1] ** 2),
        }

    out["variance"] = {
        "pred_minus_real": float(data["model_err"].var()),
        "pred_minus_apparent": float(data["vs_apparent"].var()),
        "apparent_minus_real": float(data["human_err"].var()),
    }

    # 3. Fair single-rater human baseline.
    gt = pd.concat(
        [pd.read_csv(APPA_ROOT / f"gt_avg_{s}.csv") for s in ("train", "valid", "test")],
        ignore_index=True,
    )
    rng = np.random.default_rng(seed)
    gap = (gt["apparent_age_avg"] - gt["real_age"]).to_numpy(dtype=float)
    spread = gt["apparent_age_std"].to_numpy(dtype=float)
    single = np.abs(gap + rng.normal(0.0, np.maximum(spread, 1e-9)))
    out["human_baseline"] = {
        "mean_ratings_per_image": float(gt["num_ratings"].mean()),
        "mean_inter_rater_std": float(gt["apparent_age_std"].mean()),
        "crowd_mean_error_vs_real": float(np.abs(gap).mean()),
        "single_rater_error_vs_real": float(single.mean()),
    }

    # 4. Per-decade: model against both targets, beside the human gap.
    rows = []
    for low, high, label in DECADES:
        mask = (data["real_age"] >= low) & (data["real_age"] < high)
        sub = data[mask]
        if sub.empty:
            continue
        gmask = (gt["real_age"] >= low) & (gt["real_age"] < high)
        rows.append({
            "band": label,
            "n": int(len(sub)),
            "model_mae_vs_real": float(sub["model_err"].abs().mean()),
            "model_bias_vs_real": float(sub["model_err"].mean()),
            "model_mae_vs_apparent": float(sub["vs_apparent"].abs().mean()),
            "model_bias_vs_apparent": float(sub["vs_apparent"].mean()),
            "human_crowd_mae": float(np.abs(gap[gmask]).mean()),
            "human_crowd_bias": float(gap[gmask].mean()),
            "human_single_rater_mae": float(single[gmask].mean()),
        })
    out["by_decade"] = rows
    return out


def fgnet_analysis(dump: Path) -> dict:
    data = pd.read_csv(dump)
    data = data[data["detected"]].copy()
    data["err"] = data["predicted"] - data["real_age"]
    print("  classifying grayscale scans ...", flush=True)
    data["gray"] = data["path"].map(is_grayscale)
    data = data[data["gray"].notna()]

    def stats(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"n": 0}
        err = frame["err"]
        return {
            "n": int(len(frame)),
            "mae": float(err.abs().mean()),
            "cs5": float((err.abs() <= 5).mean() * 100),
            "bias": float(err.mean()),
        }

    out = {
        "all": stats(data),
        "colour": stats(data[~data["gray"]]),
        "grayscale": stats(data[data["gray"]]),
        # Grayscale correlates with era and therefore with subject age, so the
        # comparison is repeated inside a single age band to break that link.
        "matched_under20": {
            "colour": stats(data[(~data["gray"]) & (data["real_age"] < 20)]),
            "grayscale": stats(data[data["gray"] & (data["real_age"] < 20)]),
        },
        "padded": stats(data[data["padded"]]),
        "not_padded": stats(data[~data["padded"]]),
        "clean_subset": stats(data[(~data["gray"]) & (~data["padded"])]),
        "under12": {
            "all": stats(data[data["real_age"] < 12]),
            "colour": stats(data[(~data["gray"]) & (data["real_age"] < 12)]),
            "clean": stats(
                data[(~data["gray"]) & (~data["padded"]) & (data["real_age"] < 12)]
            ),
        },
        "age_range": [int(data["real_age"].min()), int(data["real_age"].max())],
        "n_over_69": int((data["real_age"] > 69).sum()),
        "subjects": int(data["subject"].nunique()),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation analysis")
    parser.add_argument("--decode", default="median")
    args = parser.parse_args()

    result: dict = {"decode": args.decode}

    appa_dump = REPORT_DIR / f"external_appa_{args.decode}.csv"
    if appa_dump.exists():
        print("=== APPA-REAL ===")
        appa = appa_analysis(appa_dump)
        result["appa"] = appa

        print(f"  n = {appa['n']}")
        print(f"\n  Our error vs human perceptual gap, same faces:")
        print(f"    correlation {appa['error_correlation_with_human_gap']:+.3f}"
              f"  ({appa['shared_variance_pct']:.1f}% shared variance)")

        print(f"\n  Which target does the model predict?")
        for target, reg in appa["regression"].items():
            print(f"    pred ~ {target:13s} slope {reg['slope']:.3f}"
                  f"  R2 {reg['r2']:.3f}")
        print("    (slope 1.0 = no scale compression; the slopes separate the")
        print("     two targets far more sharply than the MAEs do)")

        hb = appa["human_baseline"]
        print(f"\n  Human baseline ({hb['mean_ratings_per_image']:.1f} raters/image,"
              f" inter-rater std {hb['mean_inter_rater_std']:.2f}y):")
        print(f"    crowd mean vs real   {hb['crowd_mean_error_vs_real']:.2f}y")
        print(f"    single rater vs real {hb['single_rater_error_vs_real']:.2f}y"
              "   <- the fair comparison for one model")

        print(f"\n  {'band':<7}{'n':>6}{'MAEreal':>9}{'MAEapp':>9}"
              f"{'bias_r':>9}{'humanX':>9}{'human1':>9}")
        for row in appa["by_decade"]:
            print(f"  {row['band']:<7}{row['n']:>6}"
                  f"{row['model_mae_vs_real']:>9.2f}"
                  f"{row['model_mae_vs_apparent']:>9.2f}"
                  f"{row['model_bias_vs_real']:>9.2f}"
                  f"{row['human_crowd_mae']:>9.2f}"
                  f"{row['human_single_rater_mae']:>9.2f}")
        print("  humanX = 34-rater crowd mean; human1 = single rater, both vs real age")

    fgnet_dump = REPORT_DIR / f"external_fgnet_{args.decode}.csv"
    if fgnet_dump.exists():
        print("\n=== FG-NET ===")
        fg = fgnet_analysis(fgnet_dump)
        result["fgnet"] = fg
        print(f"  subjects {fg['subjects']}, age range {fg['age_range']},"
              f" images above 69: {fg['n_over_69']}")
        print(f"\n  {'subset':<22}{'n':>6}{'MAE':>8}{'bias':>8}")
        for label, key in [
            ("all", "all"), ("colour", "colour"), ("grayscale", "grayscale"),
            ("padded", "padded"), ("not padded", "not_padded"),
            ("clean (colour+nopad)", "clean_subset"),
        ]:
            s = fg[key]
            print(f"  {label:<22}{s['n']:>6}{s['mae']:>8.2f}{s['bias']:>+8.2f}")
        print("\n  matched on age (<20), which breaks the era/age confound:")
        for label in ("colour", "grayscale"):
            s = fg["matched_under20"][label]
            print(f"  {label:<22}{s['n']:>6}{s['mae']:>8.2f}{s['bias']:>+8.2f}")
        print("\n  under-12 (the reason FG-NET is here):")
        for label, key in [("all", "all"), ("colour", "colour"), ("clean", "clean")]:
            s = fg["under12"][key]
            print(f"  {label:<22}{s['n']:>6}{s['mae']:>8.2f}{s['bias']:>+8.2f}")

    out = REPORT_DIR / f"external_analysis_{args.decode}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nAnalysis -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
