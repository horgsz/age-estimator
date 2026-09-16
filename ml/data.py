"""UTKFace data pipeline: manifest building, stratified splits and datasets.

Filenames follow ``{age}_{gender}_{race}_{datetime}.jpg.chip.jpg``. A handful of
files in the distributed tarball are missing fields; those are skipped with a
warning rather than aborting the run.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "UTKFace"
SPLIT_DIR = Path(__file__).resolve().parent / "splits"

SEED = 42
MIN_AGE = 1
MAX_AGE = 101
NUM_BINS = 101  # classifier heads cover ages 0..100
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STRATIFY_BIN_YEARS = 5
MIN_STRATUM_SIZE = 20

# Median implied YuNet margin over UTKFace; see ml/measure_crop_margin.py.
NATIVE_MARGIN = 0.0135


def parse_age(name: str) -> int | None:
    """Return the age encoded in a UTKFace filename, or None if malformed."""
    stem = name.split(".")[0]
    parts = stem.split("_")
    if len(parts) != 4:
        return None
    try:
        age = int(parts[0])
    except ValueError:
        return None
    return age


def build_manifest(data_root: Path = DATA_ROOT, verbose: bool = True) -> pd.DataFrame:
    """Scan ``data_root`` and return a DataFrame of (path, age) for valid files."""
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"UTKFace directory not found at {data_root}. Download and extract the "
            "dataset first (see ml/README.md)."
        )

    files = sorted(data_root.glob("*.jpg"))
    rows: list[dict[str, object]] = []
    malformed: list[str] = []
    out_of_range = 0

    for path in files:
        age = parse_age(path.name)
        if age is None:
            malformed.append(path.name)
            continue
        if age < MIN_AGE or age > MAX_AGE:
            out_of_range += 1
            continue
        rows.append({"path": str(path.relative_to(REPO_ROOT)), "age": age})

    if verbose:
        print(f"Scanned {len(files)} files in {data_root}")
        if malformed:
            print(f"WARNING: skipped {len(malformed)} malformed filename(s):")
            for name in malformed[:10]:
                print(f"  - {name}")
            if len(malformed) > 10:
                print(f"  ... and {len(malformed) - 10} more")
        if out_of_range:
            print(
                f"Skipped {out_of_range} file(s) with age outside "
                f"[{MIN_AGE}, {MAX_AGE}]"
            )
        print(f"Kept {len(rows)} usable images")

    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def _stratum_labels(ages: pd.Series) -> pd.Series:
    """Map ages to 5-year bins, merging rare bins so every stratum is splittable."""
    bins = (ages // STRATIFY_BIN_YEARS).astype(int)
    counts = Counter(bins)

    # Merge sparse high-age bins downward so each stratum can survive a 3-way split.
    remap: dict[int, int] = {}
    carry: list[int] = []
    for value in sorted(counts, reverse=True):
        carry.append(value)
        if sum(counts[v] for v in carry) >= MIN_STRATUM_SIZE:
            target = min(carry)
            for v in carry:
                remap[v] = target
            carry = []
    if carry:  # leftovers merge into the next-lowest already-assigned stratum
        target = min(remap.values()) if remap else min(carry)
        for v in carry:
            remap[v] = target

    return bins.map(remap)


def make_splits(
    manifest: pd.DataFrame, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Split 90/5/5 train/val/test, stratified by 5-year age bin."""
    strata = _stratum_labels(manifest["age"])

    train_df, holdout_df, _, holdout_strata = train_test_split(
        manifest,
        strata,
        test_size=0.10,
        random_state=seed,
        stratify=strata,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=seed,
        stratify=holdout_strata,
        shuffle=True,
    )

    return {
        "train": train_df.sort_values("path").reset_index(drop=True),
        "val": val_df.sort_values("path").reset_index(drop=True),
        "test": test_df.sort_values("path").reset_index(drop=True),
    }


def write_splits(splits: dict[str, pd.DataFrame], split_dir: Path = SPLIT_DIR) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(split_dir / f"{name}.csv", index=False)
        print(f"Wrote {split_dir / f'{name}.csv'} ({len(df)} rows)")


def load_split(name: str, split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    path = split_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file {path} missing. Run `python ml/data.py` to generate splits."
        )
    return pd.read_csv(path)


class RandomZoomOut:
    """Pad then resize back, simulating a looser crop than UTKFace framing.

    ``RandomResizedCrop`` only ever crops *inward*, so it gives the model
    tolerance to framings tighter than UTKFace and none at all to wider ones.
    At inference the YuNet box scatters around the ideal framing in both
    directions, so without this the model has near-zero headroom on the wide
    side. Edge padding matches how ``margin_sweep.py`` simulates wide framings.
    """

    def __init__(
        self,
        p: float = 0.5,
        max_margin: float = 0.15,
        native_margin: float = NATIVE_MARGIN,
    ) -> None:
        self.p = p
        self.max_margin = max_margin
        self.native_margin = native_margin

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() >= self.p:
            return image
        margin = random.uniform(self.native_margin, self.max_margin)
        ratio = (1.0 + 2.0 * margin) / (1.0 + 2.0 * self.native_margin)
        width, height = image.size
        pad_x = round(width * (ratio - 1.0) / 2.0)
        pad_y = round(height * (ratio - 1.0) / 2.0)
        if pad_x <= 0 and pad_y <= 0:
            return image
        padded = transforms.functional.pad(
            image, [pad_x, pad_y, pad_x, pad_y], padding_mode="edge"
        )
        return padded.resize((width, height), Image.BILINEAR)


def build_transforms(train: bool, zoom_out_prob: float = 0.0) -> transforms.Compose:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
                # Applied after the crop so the simulated wide framing is not
                # partly cropped back out.
                RandomZoomOut(p=zoom_out_prob),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            # Plain resize to exactly 224x224, matching the serving path (square
            # crop -> resize 224). A Resize+CenterCrop would keep only
            # (224/256)^2 = 0.77 of the frame, which is both tighter than
            # inference and below the train-time RandomResizedCrop scale floor.
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class UTKFaceDataset(Dataset):
    """Returns ``(image_tensor, age_int)`` pairs."""

    def __init__(
        self,
        frame: pd.DataFrame,
        train: bool,
        repo_root: Path = REPO_ROOT,
        zoom_out_prob: float = 0.0,
    ) -> None:
        self.paths = frame["path"].tolist()
        self.ages = frame["age"].astype(int).tolist()
        self.repo_root = repo_root
        self.transform = build_transforms(train, zoom_out_prob=zoom_out_prob)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.repo_root / self.paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.ages[index]


def print_age_histogram(manifest: pd.DataFrame) -> None:
    ages = manifest["age"]
    print("\n=== Age distribution ===")
    print(
        f"n={len(ages)}  min={ages.min()}  max={ages.max()}  "
        f"mean={ages.mean():.2f}  median={ages.median():.1f}  std={ages.std():.2f}"
    )

    print("\nPer-decade counts:")
    print(f"{'decade':>8} {'count':>7} {'pct':>7}  histogram")
    decade = (ages // 10).clip(upper=8)
    counts = decade.value_counts().sort_index()
    max_count = counts.max()
    for idx, count in counts.items():
        label = "80+" if idx == 8 else f"{idx * 10}-{idx * 10 + 9}"
        pct = 100 * count / len(ages)
        bar = "#" * max(1, round(50 * count / max_count))
        print(f"{label:>8} {count:>7} {pct:>6.2f}%  {bar}")

    print("\nPer-5-year-bin counts:")
    bins = (ages // 5).astype(int)
    bin_counts = bins.value_counts().sort_index()
    for idx, count in bin_counts.items():
        label = f"{idx * 5}-{idx * 5 + 4}"
        print(f"{label:>8} {count:>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UTKFace manifest and splits")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    manifest = build_manifest(args.data_root)
    print_age_histogram(manifest)

    splits = make_splits(manifest, seed=args.seed)
    print("\n=== Splits ===")
    for name, df in splits.items():
        pct = 100 * len(df) / len(manifest)
        print(f"{name:>5}: {len(df):>6} ({pct:.1f}%)  mean age {df['age'].mean():.2f}")
    write_splits(splits, args.split_dir)


if __name__ == "__main__":
    main()
