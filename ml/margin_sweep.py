"""Measure how test MAE degrades when inference framing differs from training.

``measure_crop_margin.py`` answers *what* margin reproduces UTKFace framing.
This answers *how much it costs to get it wrong*, which is what decides whether
the serving constant has to be exact or merely close.

A UTKFace image is itself a crop at margin ``m0`` (see --native-margin). To
simulate serving at margin ``m'`` we rescale the frame by
``(1 + 2m') / (1 + 2m0)``: tighter framings are an exact center crop, wider
framings need padding, which only approximates real background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, REPO_ROOT, load_split
from model import CHECKPOINT_PATH, load_checkpoint

NATIVE_MARGIN = 0.0135  # measured median over 300 UTKFace images
DEFAULT_MARGINS = [0.0, 0.0135, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def reframe(image: Image.Image, margin: float, native: float) -> Image.Image:
    """Re-frame a UTKFace crop as if it had been taken at ``margin``."""
    width, height = image.size
    ratio = (1.0 + 2.0 * margin) / (1.0 + 2.0 * native)
    target_w, target_h = width * ratio, height * ratio

    if ratio <= 1.0:  # tighter framing: exact center crop
        left = (width - target_w) / 2.0
        top = (height - target_h) / 2.0
        return image.crop(
            (round(left), round(top), round(left + target_w), round(top + target_h))
        )

    pad_x = max(0, round((target_w - width) / 2.0))
    pad_y = max(0, round((target_h - height) / 2.0))
    tensor = transforms.functional.to_tensor(image)
    padded = torch.nn.functional.pad(
        tensor.unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode="replicate"
    ).squeeze(0)
    return transforms.functional.to_pil_image(padded)


class ReframedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, margin: float, native: float) -> None:
        self.paths = frame["path"].tolist()
        self.ages = frame["age"].astype(int).tolist()
        self.margin = margin
        self.native = native
        self.post = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(REPO_ROOT / self.paths[index]) as img:
            image = img.convert("RGB")
        return self.post(reframe(image, self.margin, self.native)), self.ages[index]


@torch.no_grad()
def mae_at(model, frame, margin, native, device, batch_size, workers):
    loader = DataLoader(
        ReframedDataset(frame, margin, native),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    errs = []
    for images, ages in loader:
        pred, _ = model.expectation(model(images.to(device)))
        errs.append((pred.cpu() - ages.float()).abs().numpy())
    errs = np.concatenate(errs)
    return float(errs.mean()), float(100.0 * (errs <= 5).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop-margin sensitivity sweep")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--margins", type=float, nargs="*", default=DEFAULT_MARGINS)
    parser.add_argument("--native-margin", type=float, default=NATIVE_MARGIN)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("mps")
        if args.device == "auto" and torch.backends.mps.is_available()
        else torch.device("cpu" if args.device == "auto" else args.device)
    )
    model, _ = load_checkpoint(args.checkpoint)
    model.to(device).eval()

    frame = load_split("test")
    print(f"Device: {device} | test images: {len(frame)}")
    print(f"Native UTKFace margin assumed: {args.native_margin}\n")
    print(f"{'margin':>8} {'MAE':>8} {'CS@5':>8}  {'vs native':>10}")

    baseline = None
    for margin in args.margins:
        mae, cs5 = mae_at(
            model, frame, margin, args.native_margin, device, args.batch_size,
            args.workers,
        )
        if baseline is None or abs(margin - args.native_margin) < 1e-9:
            baseline = baseline if baseline is not None else mae
        delta = f"{mae - baseline:+.2f}" if baseline is not None else "--"
        print(f"{margin:>8.4f} {mae:>8.3f} {cs5:>7.1f}% {delta:>11}")


if __name__ == "__main__":
    main()
