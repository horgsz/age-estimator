"""Measure how test MAE degrades when inference framing differs from training.

``measure_crop_margin.py`` answers *what* margin reproduces UTKFace framing.
This answers *how much it costs to get it wrong*, which is what decides whether
the serving constant has to be exact or merely close.

A UTKFace image is itself a crop at margin ``m0`` (see --native-margin). To
simulate serving at margin ``m'`` we rescale the frame by
``(1 + 2m') / (1 + 2m0)``: tighter framings are an exact center crop, wider
framings need synthetic surroundings, which only approximate real background.

Wide framings are therefore *bracketed*, not measured:

``replicate``  smears edge pixels outward. Benign and photometrically
               consistent with the face, so it understates the real penalty.
``texture``    composites onto upscaled patches of other UTKFace photos (real
               photographic colour/texture statistics, no competing face).
``noise``      uniform random noise: an adversarial upper bound.
``gray``       flat neutral fill.

``texture``/``noise``/``gray`` introduce a hard seam at the original crop
boundary that real serving would not have, so they *overstate* the penalty.
Truth sits between ``replicate`` and ``texture``.
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
PAD_MODES = ("replicate", "texture", "noise", "gray")
_BG_PATCH = 32  # source patch size, upscaled to fill the canvas


def _texture_canvas(size, pool, rng):
    """Real photographic texture: a small patch of another photo, upscaled.

    Upscaling a 32x32 patch keeps real colour/texture statistics while
    destroying any recognisable second face that would compete for attention.
    """
    src_path = pool[rng.integers(len(pool))]
    with Image.open(REPO_ROOT / src_path) as img:
        src = img.convert("RGB")
    max_x = max(1, src.width - _BG_PATCH)
    max_y = max(1, src.height - _BG_PATCH)
    left = int(rng.integers(max_x))
    top = int(rng.integers(max_y))
    patch = src.crop((left, top, left + _BG_PATCH, top + _BG_PATCH))
    return patch.resize(size, Image.BICUBIC)


def _background(size, mode, pool, rng):
    if mode == "gray":
        return Image.new("RGB", size, (128, 128, 128))
    if mode == "noise":
        noise = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(noise, "RGB")
    if mode == "texture":
        return _texture_canvas(size, pool, rng)
    raise ValueError(mode)


def reframe(image, margin, native, mode="replicate", pool=None, rng=None):
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

    # Identical geometry for every mode so the fill is the only difference.
    pad_x = max(0, round((target_w - width) / 2.0))
    pad_y = max(0, round((target_h - height) / 2.0))
    if mode == "replicate":
        tensor = transforms.functional.to_tensor(image)
        padded = torch.nn.functional.pad(
            tensor.unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode="replicate"
        ).squeeze(0)
        return transforms.functional.to_pil_image(padded)

    canvas = _background((width + 2 * pad_x, height + 2 * pad_y), mode, pool, rng)
    canvas.paste(image, (pad_x, pad_y))
    return canvas


class ReframedDataset(Dataset):
    def __init__(self, frame, margin, native, mode="replicate", pool=None, seed=42):
        self.paths = frame["path"].tolist()
        self.ages = frame["age"].astype(int).tolist()
        self.margin = margin
        self.native = native
        self.mode = mode
        self.pool = pool
        self.seed = seed
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
        # Seeded per-sample so every mode/margin sees the same backgrounds.
        rng = np.random.default_rng(self.seed + index)
        framed = reframe(image, self.margin, self.native, self.mode, self.pool, rng)
        return self.post(framed), self.ages[index]


@torch.no_grad()
def mae_at(model, frame, margin, native, device, batch_size, workers,
           mode="replicate", pool=None):
    loader = DataLoader(
        ReframedDataset(frame, margin, native, mode, pool),
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
    parser.add_argument("--pad-modes", nargs="*", default=["replicate"],
                        choices=list(PAD_MODES))
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
    pool = load_split("train")["path"].tolist()
    print(f"Device: {device} | test images: {len(frame)}")
    print(f"Native UTKFace margin assumed: {args.native_margin}")
    print(f"Pad modes: {', '.join(args.pad_modes)}\n")

    header = f"{'margin':>8}" + "".join(f"{m:>20}" for m in args.pad_modes)
    print(header)
    print(f"{'':>8}" + "".join(f"{'MAE':>10}{'CS@5':>10}" for _ in args.pad_modes))

    for margin in args.margins:
        cells = ""
        for mode in args.pad_modes:
            mae, cs5 = mae_at(
                model, frame, margin, args.native_margin, device,
                args.batch_size, args.workers, mode, pool,
            )
            cells += f"{mae:>10.3f}{cs5:>9.1f}%"
        print(f"{margin:>8.4f}{cells}")

    print("\nTighter-than-native framings are exact centre crops, so all modes")
    print("agree there. Wide rows bracket the truth: replicate understates the")
    print("penalty, hard-composited modes overstate it (seam artefact).")


if __name__ == "__main__":
    main()
