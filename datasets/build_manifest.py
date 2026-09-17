"""Build the unified face-age manifest at ``datasets/manifest.csv``.

Four sources, two label regimes:

``utkface``      DEX-estimated ages, human double-checked. NOT real ground
                 truth -- DEX is a CNN trained on IMDB-WIKI's scraped birth
                 dates, so these labels are a model's output two error layers
                 from reality, and they encode *apparent* age.
``appa-real``    Real chronological ages supplied with the images. Also ships
                 crowd-sourced apparent ages, kept in ``apparent_age`` for
                 reference but never used as a training target.
``fgnet``        Real ages from dated personal photographs.
``agedb``        Real ages manually transcribed from photograph metadata.

Splitting
---------
UTKFace carries the *exact* assignments already in ``ml/splits/*.csv`` so the
retrained model stays comparable to the current baseline and nothing that was
held out leaks into training.

APPA-REAL ships official train/valid/test directories; those are used verbatim.

FG-NET and AgeDB are longitudinal: the same person appears many times at
different ages (~12 images each over 82 subjects for FG-NET, ~29 over 567 for
AgeDB). A random per-image split would put the same face in train and test, and
the model could score well by memorising individuals rather than by reading age.
Both are therefore split with ``GroupShuffleSplit`` on the subject ID, and
``verify_no_subject_overlap`` asserts the result. This is the single most
important correctness property in this file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import MANIFEST_PATH, MAX_AGE, MIN_AGE, RAW_DIR, REPO_ROOT, SEED

# UTKFace's 90/5/5 works because every image is a different person, so a 5%
# holdout is 1,185 independent faces. Grouped sources spend their holdout in
# whole subjects: at 5% FG-NET's test set would be four people, and the score
# would say more about those four faces than about the model. 80/10/10 buys
# ~8 subjects per holdout, which is still small but no longer a coin flip.
GROUP_TEST_FRACTION = 0.10
GROUP_VAL_FRACTION = 0.10

COLUMNS = [
    "path",
    "age",
    "source",
    "split",
    "real_ground_truth",
    "apparent_age",
    "apparent_age_std",
    "subject_id",
]

# 078A11.JPG -> subject 078, age 11. A trailing a/b disambiguates two scans of
# the same subject at the same age.
FGNET_NAME_RE = re.compile(r"^(?P<subject>\d{3})A(?P<age>\d{2})[a-zA-Z]?$")
# 10001_GoldieHawn_23_f.jpg -> identity GoldieHawn, age 23.
AGEDB_NAME_RE = re.compile(r"^(?P<image_id>\d+)_(?P<identity>.+)_(?P<age>\d+)_(?P<gender>[fm])$")


def _rel(path: Path) -> str:
    """Repo-root-relative POSIX path, matching ml/splits/*.csv's convention."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


# --------------------------------------------------------------------------
# UTKFace
# --------------------------------------------------------------------------


def load_utkface(split_dir: Path) -> pd.DataFrame:
    """Carry ml/splits/{train,val,test}.csv through verbatim."""
    frames = []
    for split in ("train", "val", "test"):
        csv_path = split_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"{csv_path} not found. Point --utkface-splits at ml/splits/ so the "
                "existing UTKFace assignments are preserved."
            )
        frame = pd.read_csv(csv_path)
        frame["split"] = split
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out["source"] = "utkface"
    out["real_ground_truth"] = False
    out["apparent_age"] = pd.NA
    out["apparent_age_std"] = pd.NA
    # Every UTKFace image is a distinct person, so each is its own group. Giving
    # them unique IDs keeps the overlap assertion meaningful across all sources.
    out["subject_id"] = "utkface:" + out.index.astype(str)
    return out[COLUMNS]


# --------------------------------------------------------------------------
# APPA-REAL
# --------------------------------------------------------------------------


def load_appa_real(root: Path) -> pd.DataFrame:
    """Read the official split dirs, using ``real_age`` as the target.

    The release also ships pre-cropped faces (``*.jpg_face.jpg``) made with a
    40% margin -- far outside the measured-safe band -- so only the originals
    are indexed here; ``crop_faces.py`` re-frames them at margin 0.0.
    """
    split_map = {"train": "train", "valid": "val", "test": "test"}
    frames = []

    for disk_split, split in split_map.items():
        gt_path = root / f"gt_avg_{disk_split}.csv"
        if not gt_path.is_file():
            raise FileNotFoundError(f"{gt_path} not found; re-run download_appa_real.sh")
        gt = pd.read_csv(gt_path)

        image_dir = root / disk_split
        rows = []
        for record in gt.itertuples(index=False):
            image_path = image_dir / str(record.file_name)
            if not image_path.is_file():
                continue
            rows.append(
                {
                    "path": _rel(image_path),
                    "age": int(record.real_age),
                    "source": "appa-real",
                    "split": split,
                    "real_ground_truth": True,
                    "apparent_age": float(record.apparent_age_avg),
                    "apparent_age_std": float(record.apparent_age_std),
                    # No identity annotation ships with APPA-REAL, and the
                    # official splits are already disjoint by construction.
                    "subject_id": f"appa-real:{disk_split}:{record.file_name}",
                }
            )

        missing = len(gt) - len(rows)
        if missing:
            print(f"  appa-real/{disk_split}: {missing} row(s) in GT have no image on disk")
        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)[COLUMNS]


# --------------------------------------------------------------------------
# FG-NET
# --------------------------------------------------------------------------


def load_fgnet(root: Path, seed: int = SEED) -> pd.DataFrame:
    """Index FG-NET, parsing subject + age from the filename, split by subject."""
    image_dir = root / "FGNET" / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{image_dir} not found; re-run download_fgnet.sh")

    rows, malformed = [], []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg"} or image_path.name.startswith("."):
            continue
        match = FGNET_NAME_RE.match(image_path.stem)
        if match is None:
            malformed.append(image_path.name)
            continue
        rows.append(
            {
                "path": _rel(image_path),
                "age": int(match.group("age")),
                "source": "fgnet",
                "split": "",
                "real_ground_truth": True,
                "apparent_age": pd.NA,
                "apparent_age_std": pd.NA,
                "subject_id": f"fgnet:{match.group('subject')}",
            }
        )

    if malformed:
        print(f"  fgnet: skipped {len(malformed)} unparseable filename(s): {malformed[:5]}")

    return assign_group_splits(pd.DataFrame(rows), seed=seed)[COLUMNS]


# --------------------------------------------------------------------------
# AgeDB
# --------------------------------------------------------------------------


def load_agedb(root: Path, seed: int = SEED) -> pd.DataFrame:
    """Index the AgeDB mirror, splitting by celebrity identity.

    Labels come from the filename (``id_Identity_age_gender.jpg``), which is
    AgeDB's own scheme, rather than from the mirror's derived ``metadata.csv``;
    see datasets/README.md on why the mirror is treated as unverified.
    """
    image_dir = root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{image_dir} not found; re-run download_agedb.sh")

    rows, malformed = [], []
    for image_path in sorted(image_dir.rglob("*.jpg")):
        if image_path.name.startswith("."):
            continue
        match = AGEDB_NAME_RE.match(image_path.stem)
        if match is None:
            malformed.append(image_path.name)
            continue
        rows.append(
            {
                "path": _rel(image_path),
                "age": int(match.group("age")),
                "source": "agedb",
                "split": "",
                "real_ground_truth": True,
                "apparent_age": pd.NA,
                "apparent_age_std": pd.NA,
                "subject_id": f"agedb:{match.group('identity')}",
            }
        )

    if malformed:
        print(f"  agedb: skipped {len(malformed)} unparseable filename(s): {malformed[:5]}")

    return assign_group_splits(pd.DataFrame(rows), seed=seed)[COLUMNS]


# --------------------------------------------------------------------------
# Grouped splitting
# --------------------------------------------------------------------------


def assign_group_splits(
    frame: pd.DataFrame,
    seed: int = SEED,
    test_fraction: float = GROUP_TEST_FRACTION,
    val_fraction: float = GROUP_VAL_FRACTION,
) -> pd.DataFrame:
    """Assign train/val/test so no ``subject_id`` spans two splits."""
    frame = frame.reset_index(drop=True).copy()
    groups = frame["subject_id"]

    holdout_fraction = test_fraction + val_fraction
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed)
    train_idx, holdout_idx = next(splitter.split(frame, groups=groups))

    holdout = frame.iloc[holdout_idx]
    # Halve the holdout by group again, so val and test are also subject-disjoint
    # from each other -- otherwise val would leak into model selection for test.
    inner_fraction = test_fraction / holdout_fraction
    inner = GroupShuffleSplit(n_splits=1, test_size=inner_fraction, random_state=seed)
    val_local, test_local = next(inner.split(holdout, groups=holdout["subject_id"]))

    frame["split"] = "train"
    frame.loc[holdout.index[val_local], "split"] = "val"
    frame.loc[holdout.index[test_local], "split"] = "test"
    return frame


def verify_no_subject_overlap(manifest: pd.DataFrame) -> None:
    """Assert every subject lives in exactly one split. Raises on violation."""
    per_subject = manifest.groupby("subject_id")["split"].nunique()
    offenders = per_subject[per_subject > 1]
    if len(offenders):
        sample = manifest[manifest["subject_id"].isin(offenders.index[:5])]
        raise AssertionError(
            f"{len(offenders)} subject(s) appear in more than one split -- the test "
            f"score would be meaningless. Examples:\n{sample[['subject_id', 'split']]}"
        )

    print("\n=== Subject-overlap check ===")
    for source, group in manifest.groupby("source", sort=True):
        subjects = group.groupby("split")["subject_id"].agg(set)
        train, val, test = (subjects.get(name, set()) for name in ("train", "val", "test"))
        assert not (train & val), f"{source}: train/val subject overlap"
        assert not (train & test), f"{source}: train/test subject overlap"
        assert not (val & test), f"{source}: val/test subject overlap"
        print(
            f"  {source:<10} {group['subject_id'].nunique():>6} subjects  "
            f"train={len(train)} val={len(val)} test={len(test)}  OK, no overlap"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--utkface-splits",
        type=Path,
        default=REPO_ROOT / "ml" / "splits",
        help="Directory holding the existing train/val/test.csv to preserve.",
    )
    parser.add_argument("--appa-real-root", type=Path, default=RAW_DIR / "appa-real-release")
    parser.add_argument("--fgnet-root", type=Path, default=RAW_DIR / "fgnet")
    parser.add_argument("--agedb-root", type=Path, default=RAW_DIR / "agedb")
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Warn and continue when a source is not downloaded, instead of failing.",
    )
    args = parser.parse_args()

    loaders = {
        "utkface": lambda: load_utkface(args.utkface_splits),
        "appa-real": lambda: load_appa_real(args.appa_real_root),
        "fgnet": lambda: load_fgnet(args.fgnet_root, seed=args.seed),
        "agedb": lambda: load_agedb(args.agedb_root, seed=args.seed),
    }

    frames = []
    for name, loader in loaders.items():
        print(f"Loading {name} ...")
        try:
            frame = loader()
        except FileNotFoundError as exc:
            if not args.skip_missing:
                raise
            print(f"  SKIPPED {name}: {exc}")
            continue
        print(f"  {len(frame)} rows")
        frames.append(frame)

    manifest = pd.concat(frames, ignore_index=True)

    before = len(manifest)
    manifest = manifest[manifest["age"].between(MIN_AGE, MAX_AGE)].reset_index(drop=True)
    if before != len(manifest):
        print(f"\nDropped {before - len(manifest)} row(s) with age outside [{MIN_AGE}, {MAX_AGE}]")

    manifest = manifest.sort_values(["source", "path"]).reset_index(drop=True)
    verify_no_subject_overlap(manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(manifest)} rows)")


if __name__ == "__main__":
    main()
