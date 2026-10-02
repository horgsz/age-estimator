"""Train the DEX-style age estimator on UTKFace (Apple MPS by default)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import NUM_BINS, UTKFaceDataset, load_split
from realgt_data import (
    DEFAULT_DATASETS_ROOT,
    RealGTDataset,
    assert_no_subject_leakage,
    load_manifest,
    split_frame,
)
from model import AgeEstimator, load_checkpoint, save_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "age_model.pt"
HISTORY_PATH = Path(__file__).resolve().parent / "reports" / "train_history.json"


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# The ages the product cares about. Teenagers through mid-adulthood matter most,
# the elderly less, young children least. --age-focus samples by this
# importance and selects checkpoints on MAE inside FOCUS_BAND.
FOCUS_BAND = (14, 60)


def age_importance(age: int) -> float:
    if age < 10:
        return 0.25
    if age < FOCUS_BAND[0]:
        return 0.5
    if age <= FOCUS_BAND[1]:
        return 1.0
    if age < 80:
        return 0.6
    return 0.35


def focus_sample_weights(ages: pd.Series) -> torch.Tensor:
    """Per-sample draw weights: importance / sqrt(frequency of that age).

    The square root partially flattens the corpus's 20-50 peak so thin bands
    inside the focus range (teenagers especially) are seen more often, without
    the noise amplification of full inverse-frequency balancing.
    """
    counts = ages.map(ages.value_counts())
    weights = ages.map(age_importance) / counts.pow(0.5)
    return torch.as_tensor(weights.to_numpy(), dtype=torch.double)


def build_loaders(
    batch_size: int, workers: int, limit: int | None, seed: int, zoom_out: float = 0.0,
    corpus: str = "utkface", datasets_root=DEFAULT_DATASETS_ROOT, age_focus: bool = False,
    sources: list[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders for either corpus.

    The two corpora share this function, and the augmentations, deliberately:
    the real-GT retrain is meant to isolate the *data* change, so every other
    part of the recipe has to be the same code rather than merely the same
    intent.
    """
    if corpus == "realgt":
        frame = load_manifest(sources=sources)
        # Re-checked here, not just at corpus-build time, because leakage shows
        # up as a better score rather than an error.
        assert_no_subject_leakage(frame)
        train_df, val_df = split_frame(frame, "train"), split_frame(frame, "val")
        make = lambda df, tr: RealGTDataset(  # noqa: E731
            df, train=tr, datasets_root=datasets_root,
            zoom_out_prob=zoom_out if tr else 0.0,
        )
    else:
        train_df, val_df = load_split("train"), load_split("val")
        make = lambda df, tr: UTKFaceDataset(  # noqa: E731
            df, train=tr, zoom_out_prob=zoom_out if tr else 0.0,
        )
    if limit:
        train_df = train_df.sample(
            n=min(limit, len(train_df)), random_state=seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(max(limit // 8, 64), len(val_df)), random_state=seed
        ).reset_index(drop=True)

    sampler = None
    if age_focus:
        sampler = WeightedRandomSampler(
            focus_sample_weights(train_df["age"]), num_samples=len(train_df),
            replacement=True, generator=torch.Generator().manual_seed(seed),
        )
    train_loader = DataLoader(
        make(train_df, True),
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        make(val_df, False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    return train_loader, val_loader


def cosine_warmup_lambda(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def amp_is_stable(model: nn.Module, device: torch.device, criterion: nn.Module) -> bool:
    """Probe one autocast forward/backward; fall back to fp32 if it misbehaves."""
    if device.type == "cpu":
        return False
    try:
        dummy = torch.randn(2, 3, 224, 224, device=device)
        target = torch.tensor([25, 40], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss = criterion(model(dummy), target)
        loss.backward()
        ok = torch.isfinite(loss).item() and all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        )
    except Exception as exc:  # pragma: no cover - backend dependent
        print(f"AMP probe raised {type(exc).__name__}: {exc}")
        ok = False
    finally:
        model.zero_grad(set_to_none=True)
    return bool(ok)


class DLDLLoss(nn.Module):
    """Deep Label Distribution Learning: a distance-aware soft target.

    ``label_smoothing`` spreads its mass *uniformly*, which for an ordinal
    target is a strange claim: it says a 3-year-old and a 90-year-old are
    equally plausible alternatives for a 5-year-old. It also plants a uniform
    pedestal whose own expectation is 50, which is what made the soft-expectation
    decode mean-revert (see ml/README.md).

    DLDL replaces that with a Gaussian centred on the true age, so probability
    mass is placed on *nearby* ages in proportion to how near they are. The
    supervision then matches the metric: being wrong by one year should cost
    less than being wrong by forty.

    Loss is KL divergence between the predicted distribution and the target.
    """

    def __init__(self, num_bins: int, sigma: float = 2.5) -> None:
        super().__init__()
        self.sigma = sigma
        centers = torch.arange(num_bins, dtype=torch.float32)
        self.register_buffer("centers", centers)

    def forward(self, logits: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(logits.device)
        diff = centers.unsqueeze(0) - ages.float().unsqueeze(1)
        target = torch.exp(-(diff ** 2) / (2.0 * self.sigma ** 2))
        # Renormalise after truncation: ages near 0 or 100 lose the tail that
        # falls outside the support, and an unnormalised target would silently
        # down-weight exactly the extreme ages this retrain is meant to fix.
        target = target / target.sum(dim=1, keepdim=True)
        log_pred = F.log_softmax(logits.float(), dim=1)
        return F.kl_div(log_pred, target, reduction="batchmean")


def build_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "dldl":
        return DLDLLoss(NUM_BINS, sigma=args.dldl_sigma)
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def mixup_batch(
    images: torch.Tensor, targets: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed images plus the pair of targets and the mixing weight."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    lam = max(lam, 1.0 - lam)  # keep the dominant label dominant
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, targets, targets[perm], lam


@torch.no_grad()
def evaluate(
    model: AgeEstimator, loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    """Return (MAE, CS@5, MAE inside FOCUS_BAND)."""
    model.eval()
    abs_err_sum = 0.0
    within5 = 0
    n = 0
    focus_err_sum = 0.0
    focus_n = 0
    for images, ages in loader:
        images = images.to(device, non_blocking=True)
        pred, _ = model.expectation(model(images))
        pred = pred.cpu()
        err = (pred - ages.float()).abs()
        abs_err_sum += err.sum().item()
        within5 += (err <= 5).sum().item()
        n += len(ages)
        in_band = (ages >= FOCUS_BAND[0]) & (ages <= FOCUS_BAND[1])
        focus_err_sum += err[in_band].sum().item()
        focus_n += int(in_band.sum())
    focus_mae = focus_err_sum / focus_n if focus_n else float("nan")
    return abs_err_sum / n, 100.0 * within5 / n, focus_mae


def train(args: argparse.Namespace) -> float:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(
        args.batch_size, args.workers, args.limit, args.seed, args.zoom_out,
        corpus=args.corpus, datasets_root=args.datasets_root, age_focus=args.age_focus,
        sources=args.sources,
    )
    print(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
        f"| batch_size={args.batch_size}"
    )

    if args.init_from:
        model, _ = load_checkpoint(args.init_from)
        model = model.to(device)
        print(f"Fine-tuning from {args.init_from}")
    else:
        model = AgeEstimator(num_bins=NUM_BINS, pretrained=not args.no_pretrained).to(device)
    criterion = build_criterion(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(total_steps, warmup_steps)
    )

    use_amp = False
    if args.amp != "off":
        use_amp = amp_is_stable(model, device, criterion)
        if args.amp == "on" and not use_amp:
            print("AMP requested but probe failed; continuing in fp32.")
    print(f"Precision: {'fp16 autocast' if use_amp else 'fp32'}")

    best_mae = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0

        for step, (images, ages) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = ages.clamp(0, NUM_BINS - 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if args.mixup > 0:
                images, target_a, target_b, lam = mixup_batch(
                    images, targets, args.mixup
                )

                def compute_loss() -> torch.Tensor:
                    logits = model(images)
                    return lam * criterion(logits, target_a) + (1.0 - lam) * criterion(
                        logits, target_b
                    )

            else:

                def compute_loss() -> torch.Tensor:
                    return criterion(model(images), targets)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss = compute_loss()
            else:
                loss = compute_loss()

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * len(ages)
            seen += len(ages)
            if step % args.log_every == 0 or step == steps_per_epoch:
                print(
                    f"  epoch {epoch:>2} step {step:>4}/{steps_per_epoch} "
                    f"loss {running_loss / seen:.4f} "
                    f"lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        val_mae, val_cs5, val_focus_mae = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        # With --age-focus, "best" means best inside the band that matters.
        selection = val_focus_mae if args.age_focus else val_mae
        improved = selection < best_mae
        print(
            f"epoch {epoch:>2}/{args.epochs} | loss {running_loss / seen:.4f} "
            f"| val MAE {val_mae:.3f} | val MAE {FOCUS_BAND[0]}-{FOCUS_BAND[1]} "
            f"{val_focus_mae:.3f} | val CS@5 {val_cs5:.2f}% "
            f"| {elapsed:.1f}s{'  <-- best' if improved else ''}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "val_mae": val_mae,
                "val_focus_mae": val_focus_mae,
                "val_cs5": val_cs5,
                "lr": scheduler.get_last_lr()[0],
                "seconds": elapsed,
            }
        )

        if improved:
            best_mae = selection
            # test_mae provisionally holds the best *val* MAE. That caveat used
            # to live only in this comment, where no consumer of the file could
            # see it -- a checkpoint straight from training advertised a test
            # figure it had never earned. Stamp the provenance into meta so it
            # travels with the artifact instead of with the source.
            save_checkpoint(
                model,
                best_mae,
                args.checkpoint,
                decode="expectation",
                extra={
                    "val_mae": float(val_mae),
                    "val_focus_mae": float(val_focus_mae),
                    "selection": "val_focus_mae" if args.age_focus else "val_mae",
                    "test_mae_source": (
                        "PROVISIONAL: this is the best validation MAE, not a "
                        "held-out test measurement. Run eval.py (utkface) or "
                        "realgt_compare.py (realgt) to replace it."
                    ),
                },
            )
            print(f"  saved checkpoint -> {args.checkpoint}")

    # Derived from the checkpoint name so parallel variants cannot silently
    # overwrite each other's curves.
    stem = Path(args.checkpoint).stem
    history_path = (
        HISTORY_PATH if stem == "age_model"
        else HISTORY_PATH.with_name(f"train_history_{stem}.json")
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    label = f"val MAE {FOCUS_BAND[0]}-{FOCUS_BAND[1]}" if args.age_focus else "val MAE"
    print(f"\nBest {label}: {best_mae:.3f}\nHistory -> {history_path}")
    return best_mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the UTKFace age estimator")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument(
        "--zoom-out",
        type=float,
        default=0.5,
        help="probability of simulating a wider crop; 0 disables. Gives the "
        "model tolerance to loose YuNet boxes, which RandomResizedCrop cannot.",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="mixup alpha; 0 disables. Helps when the head overfits.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--limit", type=int, default=None, help="subset size for smoke tests")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument(
        "--corpus", choices=["utkface", "realgt"], default="utkface",
        help="utkface = DEX-labelled (the shipped model); realgt = chronological",
    )
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument(
        "--loss", choices=["ce", "dldl"], default="ce",
        help="ce = cross-entropy with uniform label smoothing; "
             "dldl = distance-aware Gaussian soft target",
    )
    parser.add_argument("--dldl-sigma", type=float, default=2.5)
    parser.add_argument(
        "--age-focus", action="store_true",
        help=f"sample by age importance (ages {FOCUS_BAND[0]}-{FOCUS_BAND[1]} first) and "
             "select the checkpoint on MAE inside that band",
    )
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="realgt corpus: restrict to these sources (default: all real-GT sources)",
    )
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="fine-tune from this checkpoint instead of ImageNet weights",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
