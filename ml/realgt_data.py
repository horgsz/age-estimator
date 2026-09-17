"""Loader for the real-chronological-ground-truth corpus in ``datasets/``.

Separate from ``data.py``, which owns the UTKFace pipeline, because the two
corpora differ in a way that matters structurally rather than cosmetically:

* **UTKFace is one image per person.** A random holdout is automatically a set
  of unseen faces.
* **AgeDB and FG-NET are longitudinal.** AgeDB averages 29 images per subject
  and FG-NET 12, so a random split would put the same person in train and test.
  The model could then score well by recognising individuals it had memorised
  rather than by reading age, and the test number would be meaningless.

The sibling session assigned identity-aware splits when building the manifest.
This module **re-asserts that invariant rather than trusting it** -- see
:func:`assert_no_subject_leakage`, which runs by default on every load. A silent
regression here would not produce an error, it would produce a *better* score,
which is the kind of bug that gets celebrated instead of caught.

The crops are gitignored and live in the dataset session's worktree, so
``crop_path`` is relative and must be resolved against ``--datasets-root``.
Nothing here writes to ``datasets/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data import REPO_ROOT, build_transforms

MANIFEST_PATH = REPO_ROOT / "datasets" / "manifest.csv"
# Default sibling worktree holding the gitignored crops. Override with
# --datasets-root; copying 25k images into this worktree would be wasteful and
# would drift from the source of truth.
DEFAULT_DATASETS_ROOT = Path(
    "/Users/horganm0/.copilot/repos/copilot-worktrees/age-estimator/"
    "horganm0-risk-symmetrical-guide"
)
SPLITS = ("train", "val", "test")


def load_manifest(
    manifest: Path = MANIFEST_PATH,
    real_only: bool = True,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Load the corpus manifest, keeping only usable rows.

    ``real_only`` is the whole point of this retrain: it drops UTKFace, whose
    labels are DEX estimates of *apparent* age, leaving only sources with
    chronological ground truth.
    """
    frame = pd.read_csv(manifest)
    if real_only:
        frame = frame[frame["real_ground_truth"] == True]  # noqa: E712
    if sources:
        frame = frame[frame["source"].isin(sources)]

    # Rows where detection failed carry no crop and cannot be used.
    missing = frame["crop_path"].isna().sum()
    if missing:
        print(f"  dropping {missing} rows with no detected face")
    frame = frame[frame["crop_path"].notna()]

    # The head covers ages 0..100; anything outside cannot be represented.
    outside = ((frame["age"] < 0) | (frame["age"] > 100)).sum()
    if outside:
        print(f"  dropping {outside} rows with age outside 0..100")
    frame = frame[(frame["age"] >= 0) & (frame["age"] <= 100)]

    return frame.reset_index(drop=True)


def assert_no_subject_leakage(frame: pd.DataFrame) -> None:
    """Fail loudly if any subject appears in more than one split.

    Deliberately an assertion and not a warning. Leakage inflates the test
    score, so it presents as good news; if this is ever downgraded to a log
    line it will be ignored exactly when it matters most.
    """
    by_split = {
        split: set(group["subject_id"])
        for split, group in frame.groupby("split")
    }
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            overlap = by_split.get(left, set()) & by_split.get(right, set())
            if overlap:
                sample = sorted(overlap)[:5]
                raise AssertionError(
                    f"Subject leakage between {left} and {right}: "
                    f"{len(overlap)} shared subjects, e.g. {sample}"
                )
    total = frame["subject_id"].nunique()
    print(f"  no-leakage invariant re-verified across {total} subjects")


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[frame["split"] == split].reset_index(drop=True)


class RealGTDataset(Dataset):
    """Returns ``(image_tensor, age_int)`` from pre-cropped faces.

    The crops were produced at ``CROP_MARGIN = 0.0``, matching the serving path,
    so no further framing is applied here beyond the shared augmentations.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        train: bool,
        datasets_root: Path = DEFAULT_DATASETS_ROOT,
        zoom_out_prob: float = 0.0,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = Path(datasets_root)
        self.transform = build_transforms(train, zoom_out_prob=zoom_out_prob)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image = Image.open(self.root / row["crop_path"]).convert("RGB")
        return self.transform(image), int(row["age"])


def describe(frame: pd.DataFrame) -> None:
    """Print the corpus composition that the training run will actually see."""
    print(f"  images: {len(frame)}   subjects: {frame['subject_id'].nunique()}")
    print(f"  {'split':<8}{'n':>8}{'subjects':>10}{'median age':>12}")
    for split in SPLITS:
        sub = frame[frame["split"] == split]
        if sub.empty:
            continue
        print(f"  {split:<8}{len(sub):>8}{sub['subject_id'].nunique():>10}"
              f"{sub['age'].median():>12.0f}")

    print(f"\n  {'source':<12}" + "".join(f"{s:>8}" for s in SPLITS))
    for source, group in frame.groupby("source"):
        counts = "".join(f"{(group['split'] == s).sum():>8}" for s in SPLITS)
        print(f"  {source:<12}{counts}")

    print(f"\n  {'decade':<10}{'n':>8}{'share':>9}")
    for low in range(0, 100, 10):
        high = low + 10 if low < 90 else 201
        n = int(((frame["age"] >= low) & (frame["age"] < high)).sum())
        if n:
            label = f"{low}-{low + 9}" if low < 90 else "90+"
            print(f"  {label:<10}{n:>8}{100 * n / len(frame):>8.1f}%")

    teens = int(((frame["age"] >= 10) & (frame["age"] <= 19)).sum())
    elderly = int((frame["age"] >= 70).sum())
    print(f"\n  teens 10-19: {teens}    70+: {elderly}")


def add_corpus_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI surface so every script resolves crops the same way."""
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT,
        help="worktree holding the gitignored datasets/crops tree",
    )
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="restrict to given sources (default: all real-GT sources)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the real-GT corpus")
    add_corpus_args(parser)
    args = parser.parse_args()

    frame = load_manifest(args.manifest, sources=args.sources)
    assert_no_subject_leakage(frame)
    describe(frame)

    root = Path(args.datasets_root)
    sample = frame["crop_path"].head(200)
    missing = sum(1 for p in sample if not (root / p).exists())
    print(f"\n  crop spot-check: {len(sample) - missing}/{len(sample)} resolve "
          f"under {root}")


if __name__ == "__main__":
    main()
