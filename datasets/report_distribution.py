"""Characterise the combined corpus: age distribution, provenance, coverage.

The two questions this exists to answer, because they are the model's known
weak spots:

* how much **genuine** ground-truth data exists at 70+, where the current model
  is biased about -7.5 years, and
* how much at under-12, which UTKFace covers thinly and DEX labels worst.

"Genuine" is doing real work in that sentence. UTKFace's ages are DEX outputs
double-checked by a human, so a decade bucket that looks well populated may be
populated entirely by a model's guesses. Every table here therefore splits by
``real_ground_truth``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import MANIFEST_PATH, REPORTS_DIR  # noqa: E402

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


def decade_index(ages: pd.Series) -> pd.Series:
    return (ages // 10).clip(upper=8).astype(int)


def _table(counts: pd.DataFrame, title: str) -> str:
    lines = [f"\n=== {title} ===", ""]
    columns = list(counts.columns)
    header = f"{'decade':>8} " + " ".join(f"{c:>10}" for c in columns) + f" {'total':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for idx, label in enumerate(DECADE_LABELS):
        if idx not in counts.index:
            row = [0] * len(columns)
        else:
            row = [int(counts.loc[idx, c]) for c in columns]
        lines.append(f"{label:>8} " + " ".join(f"{v:>10,}" for v in row) + f" {sum(row):>9,}")
    totals = [int(counts[c].sum()) for c in columns]
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':>8} " + " ".join(f"{v:>10,}" for v in totals) + f" {sum(totals):>9,}")
    return "\n".join(lines)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "n/a"


def build_report(manifest: pd.DataFrame) -> str:
    manifest = manifest.copy()
    manifest["decade"] = decade_index(manifest["age"])
    real = manifest[manifest["real_ground_truth"]]
    estimated = manifest[~manifest["real_ground_truth"]]

    out: list[str] = []
    out.append("=== Corpus summary ===\n")
    out.append(f"{'source':<12} {'images':>8} {'real GT':>9} {'subjects':>9} {'age range':>12}")
    out.append("-" * 54)
    for source, group in manifest.groupby("source", sort=True):
        out.append(
            f"{source:<12} {len(group):>8,} {str(bool(group['real_ground_truth'].iloc[0])):>9} "
            f"{group['subject_id'].nunique():>9,} "
            f"{f'{group.age.min()}-{group.age.max()}':>12}"
        )
    out.append("-" * 54)
    out.append(
        f"{'TOTAL':<12} {len(manifest):>8,} {'':>9} {manifest['subject_id'].nunique():>9,}"
    )
    out.append(
        f"\nreal_ground_truth=True : {len(real):,} ({_pct(len(real), len(manifest))})"
        f"\nreal_ground_truth=False: {len(estimated):,} "
        f"({_pct(len(estimated), len(manifest))})  [utkface, DEX-estimated]"
    )

    by_source = manifest.pivot_table(
        index="decade", columns="source", values="path", aggfunc="count", fill_value=0
    )
    out.append(_table(by_source, "Age distribution by decade and source"))

    by_gt = manifest.assign(
        provenance=manifest["real_ground_truth"].map({True: "real GT", False: "DEX-estimated"})
    ).pivot_table(
        index="decade", columns="provenance", values="path", aggfunc="count", fill_value=0
    )
    out.append(_table(by_gt, "Age distribution by decade and label provenance"))

    by_split = manifest.pivot_table(
        index="decade", columns="split", values="path", aggfunc="count", fill_value=0
    )
    out.append(_table(by_split[[c for c in ("train", "val", "test") if c in by_split]], "Age distribution by decade and split"))

    out.append("\n=== Weak-spot coverage (the point of this corpus) ===\n")
    for label, mask in (
        ("under 12", manifest["age"] < 12),
        ("70 and over", manifest["age"] >= 70),
        ("80 and over", manifest["age"] >= 80),
    ):
        subset = manifest[mask]
        subset_real = subset[subset["real_ground_truth"]]
        out.append(f"{label}:")
        out.append(f"  total          {len(subset):>7,}  ({_pct(len(subset), len(manifest))} of corpus)")
        out.append(
            f"  real GT        {len(subset_real):>7,}  "
            f"({_pct(len(subset_real), len(subset))} of this band)"
        )
        for source, group in subset.groupby("source", sort=True):
            out.append(f"    {source:<10} {len(group):>7,}")
        train_real = subset_real[subset_real["split"] == "train"]
        out.append(f"  real GT in train {len(train_real):>5,}")
        out.append("")

    out.append("=== Split sizes ===\n")
    sizes = manifest.pivot_table(
        index="source", columns="split", values="path", aggfunc="count", fill_value=0
    )
    out.append(sizes.to_string())

    if "face_detected" in manifest.columns and manifest["face_detected"].notna().any():
        out.append("\n=== Face detection ===\n")
        det = manifest.dropna(subset=["face_detected"]).copy()
        det["face_detected"] = det["face_detected"].astype(bool)
        rows = det.groupby("source").agg(
            images=("path", "count"), detected=("face_detected", "sum")
        )
        rows["failed"] = rows["images"] - rows["detected"]
        rows["fail_pct"] = (100.0 * rows["failed"] / rows["images"]).round(2)
        out.append(rows.to_string())
        if "detect_pass" in det.columns:
            out.append("\nDetection pass used:\n")
            out.append(
                det.pivot_table(
                    index="source", columns="detect_pass", values="path",
                    aggfunc="count", fill_value=0,
                ).to_string()
            )

    return "\n".join(out) + "\n"


def plot(manifest: pd.DataFrame, out_path: Path) -> None:
    manifest = manifest.copy()
    manifest["decade"] = decade_index(manifest["age"])
    sources = sorted(manifest["source"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    bottom = pd.Series(0, index=range(9), dtype=float)
    for source in sources:
        counts = (
            manifest[manifest["source"] == source]["decade"]
            .value_counts()
            .reindex(range(9), fill_value=0)
            .sort_index()
        )
        ax.bar(range(9), counts.values, bottom=bottom.values, label=source)
        bottom += counts.values
    ax.set_xticks(range(9))
    ax.set_xticklabels(DECADE_LABELS, rotation=45)
    ax.set_title("Combined corpus by decade and source")
    ax.set_ylabel("images")
    ax.legend()

    ax = axes[1]
    width = 0.4
    for offset, (label, mask) in enumerate(
        (("real ground truth", manifest["real_ground_truth"]),
         ("DEX-estimated (UTKFace)", ~manifest["real_ground_truth"]))
    ):
        counts = (
            manifest[mask]["decade"].value_counts().reindex(range(9), fill_value=0).sort_index()
        )
        ax.bar([i + offset * width - width / 2 for i in range(9)], counts.values, width, label=label)
    ax.set_xticks(range(9))
    ax.set_xticklabels(DECADE_LABELS, rotation=45)
    ax.set_title("Label provenance by decade")
    ax.legend()

    ax = axes[2]
    for source in sources:
        ages = manifest[manifest["source"] == source]["age"]
        ax.hist(ages, bins=range(0, 102, 2), histtype="step", linewidth=1.6, label=source)
    ax.set_title("Per-year age histogram by source")
    ax.set_xlabel("age")
    ax.legend()

    fig.suptitle(
        f"Face-age corpus: {len(manifest):,} images, "
        f"{int(manifest['real_ground_truth'].sum()):,} with real chronological ages"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["real_ground_truth"] = manifest["real_ground_truth"].astype(bool)

    report = build_report(manifest)
    print(report)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "distribution.txt").write_text(report)
    plot(manifest, args.reports_dir / "age_distribution.png")
    print(f"Wrote {args.reports_dir / 'distribution.txt'}")
    print(f"Wrote {args.reports_dir / 'age_distribution.png'}")


if __name__ == "__main__":
    main()
