"""Score checkpoints on the held-out test split, the way the server decodes.

Reports, per checkpoint, on the clean-label sources (AgeDB + APPA-REAL +
FG-NET, n~3818 -- the split every served model's headline figure uses) and on
the full test split including IMDB-Clean:

  MAE, MAE for ages 14-60, MAE for 13-19, % of under-18s shown as 18+,
  and the coverage of the 0.16/0.84 interval.

  ../.venv/bin/python score_checkpoints.py live=../checkpoints/age_model_realgt.pt \
      new=../checkpoints/age_model_candidate.pt --out reports/x.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import load_checkpoint
from realgt_data import REPO_ROOT, RealGTDataset, load_manifest, split_frame


def score(med, lo, hi, truth) -> dict:
    err = np.abs(med - truth)
    focus = (truth >= 14) & (truth <= 60)
    teens = (truth >= 13) & (truth <= 19)
    minors = truth < 18
    return {
        "n": int(len(truth)),
        "mae": round(float(err.mean()), 3),
        "mae_14_60": round(float(err[focus].mean()), 3),
        "mae_13_19": round(float(err[teens].mean()), 2),
        "cs5_14_60": round(float((err[focus] <= 5).mean() * 100), 1),
        "under18_shown_adult_pct": round(float((med[minors] >= 18).mean() * 100), 1),
        "interval_coverage": round(float(((lo <= truth) & (truth <= hi)).mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="label=path")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--datasets-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    frame = load_manifest(args.manifest) if args.manifest else load_manifest()
    test = split_frame(frame, "test")
    clean = (test["source"] != "imdb-clean").to_numpy() & (test["source"] != "cacd").to_numpy()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    results = {}
    for spec in args.checkpoints:
        label, _, path = spec.partition("=")
        model, meta = load_checkpoint(Path(path))
        model.to(device).eval()
        loader = DataLoader(RealGTDataset(test, train=False, datasets_root=args.datasets_root),
                            batch_size=args.batch_size, num_workers=4)
        logits, ages = [], []
        with torch.no_grad():
            for images, age in loader:
                logits.append(model(images.to(device)).float().cpu())
                ages.append(age)
        cdf = torch.softmax(torch.cat(logits), 1).cumsum(1)
        truth = torch.cat(ages).numpy().astype(float)
        med, lo, hi = [(cdf < q).sum(1).float().numpy() for q in (0.5, 0.16, 0.84)]
        results[label] = {
            "path": str(path), "backbone": meta.get("backbone"),
            "clean": score(med[clean], lo[clean], hi[clean], truth[clean]),
            "all": score(med, lo, hi, truth),
        }
        print(label, json.dumps(results[label]["clean"]), flush=True)
        print(" " * len(label), "all", json.dumps(results[label]["all"]), flush=True)
        del model
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
