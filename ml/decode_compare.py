"""Compare ways of collapsing the 101-bin distribution to a single age.

The DEX soft-expectation decode ``sum_i p_i * i`` is mean-reverting by
construction: probability mass cannot extend below bin 0 or above bin 100, so
near either boundary the surviving mass is one-sided and the expectation is
pushed inward. For the 0-9 band, where the model's MAE and its bias are nearly
equal, that predicts a *pure systematic offset* -- and a decode that does not
average over the whole support (mode, median, or a window around the mode)
should be immune to it.

This script measures that on an existing checkpoint. No retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import UTKFaceDataset, load_split
from model import CHECKPOINT_PATH, load_checkpoint

DECADE_EDGES = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49),
                (50, 59), (60, 69), (70, 79), (80, 200)]


def decode_all(probs, windows, smoothing=(0.1,)):
    """Return one predicted-age vector per decoding strategy."""
    bins = np.arange(probs.shape[1], dtype=np.float64)
    out: dict[str, np.ndarray] = {}

    out["expectation"] = probs @ bins
    out["mode"] = probs.argmax(axis=1).astype(np.float64)

    cdf = np.cumsum(probs, axis=1)
    out["median"] = (cdf < 0.5).sum(axis=1).astype(np.float64)

    # Local expectation: the mode's sub-bin refinement. Keeps the smoothness of
    # averaging but truncates the long tail that drags the mean inward.
    modes = probs.argmax(axis=1)
    for w in windows:
        lo = np.clip(modes - w, 0, probs.shape[1] - 1)
        hi = np.clip(modes + w, 0, probs.shape[1] - 1)
        idx = np.arange(probs.shape[1])[None, :]
        mask = (idx >= lo[:, None]) & (idx <= hi[:, None])
    # Label smoothing (0.1) trains the model to place a uniform pedestal of
    # eps/101 on every bin. That pedestal's own expectation is 50, so it drags
    # every prediction toward the middle: E ~= (1-eps)*age + eps*50. Subtracting
    # it back out is the principled fix, and unlike mode/median it keeps the
    # sub-bin resolution that makes the soft-expectation decode worth having.
    for eps in smoothing:
        floor = eps / probs.shape[1]
        adj = np.clip(probs - floor, 0.0, None)
        adj_sum = adj.sum(axis=1, keepdims=True)
        safe = np.where(adj_sum > 0, adj_sum, 1.0)
        out[f"ls{eps:g}"] = ((adj / safe) @ bins)

    # Closed-form inverse of the same effect, applied to the raw expectation.
    out["ls-linear"] = (out["expectation"] - 0.1 * 50.0) / 0.9
    return out


def metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    err = pred - true
    return float(np.abs(err).mean()), float(100.0 * (np.abs(err) <= 5).mean()), float(err.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare distribution decodes")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--windows", type=int, nargs="*", default=(5, 12))
    parser.add_argument("--smoothing", type=float, nargs="*",
                        default=(0.05, 0.1, 0.15))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("mps")
        if args.device == "auto" and torch.backends.mps.is_available()
        else torch.device("cpu" if args.device == "auto" else args.device)
    )
    model, _ = load_checkpoint(args.checkpoint)
    model.to(device).eval()

    frame = load_split(args.split)
    loader = DataLoader(
        UTKFaceDataset(frame, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    chunks, ages = [], []
    with torch.no_grad():
        for images, age in loader:
            logits = model(images.to(device))
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ages.append(age.numpy())
    probs = np.concatenate(chunks).astype(np.float64)
    true = np.concatenate(ages).astype(np.float64)

    decodes = decode_all(probs, tuple(args.windows), tuple(args.smoothing))
    names = list(decodes)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"{args.split} set: {len(true)} images | device {device}\n")

    print("=== overall ===")
    print(f"{'decode':>12} {'MAE':>8} {'CS@5':>8} {'bias':>8}")
    for name in names:
        mae, cs5, bias = metrics(decodes[name], true)
        print(f"{name:>12} {mae:>8.3f} {cs5:>7.1f}% {bias:>+8.2f}")

    print("\n=== MAE per decade ===")
    print(f"{'decade':>8} {'n':>5}" + "".join(f"{n:>12}" for n in names))
    for lo, hi in DECADE_EDGES:
        sel = (true >= lo) & (true <= hi)
        if not sel.any():
            continue
        label = f"{lo}-{hi}" if hi < 200 else f"{lo}+"
        row = "".join(f"{np.abs(decodes[n][sel] - true[sel]).mean():>12.2f}"
                      for n in names)
        print(f"{label:>8} {int(sel.sum()):>5}{row}")

    print("\n=== bias per decade (systematic offset) ===")
    print(f"{'decade':>8} {'n':>5}" + "".join(f"{n:>12}" for n in names))
    for lo, hi in DECADE_EDGES:
        sel = (true >= lo) & (true <= hi)
        if not sel.any():
            continue
        label = f"{lo}-{hi}" if hi < 200 else f"{lo}+"
        row = "".join(f"{(decodes[n][sel] - true[sel]).mean():>+12.2f}"
                      for n in names)
        print(f"{label:>8} {int(sel.sum()):>5}{row}")


if __name__ == "__main__":
    main()
