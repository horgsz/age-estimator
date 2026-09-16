"""Train the DEX-style age estimator on UTKFace (Apple MPS by default)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import NUM_BINS, UTKFaceDataset, load_split
from model import AgeEstimator, save_checkpoint

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


def build_loaders(
    batch_size: int, workers: int, limit: int | None, seed: int, zoom_out: float = 0.0
) -> tuple[DataLoader, DataLoader]:
    train_df = load_split("train")
    val_df = load_split("val")
    if limit:
        train_df = train_df.sample(
            n=min(limit, len(train_df)), random_state=seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(max(limit // 8, 64), len(val_df)), random_state=seed
        ).reset_index(drop=True)

    train_loader = DataLoader(
        UTKFaceDataset(train_df, train=True, zoom_out_prob=zoom_out),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        UTKFaceDataset(val_df, train=False),
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
) -> tuple[float, float]:
    model.eval()
    abs_err_sum = 0.0
    within5 = 0
    n = 0
    for images, ages in loader:
        images = images.to(device, non_blocking=True)
        pred, _ = model.expectation(model(images))
        pred = pred.cpu()
        err = (pred - ages.float()).abs()
        abs_err_sum += err.sum().item()
        within5 += (err <= 5).sum().item()
        n += len(ages)
    return abs_err_sum / n, 100.0 * within5 / n


def train(args: argparse.Namespace) -> float:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(
        args.batch_size, args.workers, args.limit, args.seed, args.zoom_out
    )
    print(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
        f"| batch_size={args.batch_size}"
    )

    model = AgeEstimator(num_bins=NUM_BINS, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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

        val_mae, val_cs5 = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        improved = val_mae < best_mae
        print(
            f"epoch {epoch:>2}/{args.epochs} | loss {running_loss / seen:.4f} "
            f"| val MAE {val_mae:.3f} | val CS@5 {val_cs5:.2f}% "
            f"| {elapsed:.1f}s{'  <-- best' if improved else ''}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "val_mae": val_mae,
                "val_cs5": val_cs5,
                "lr": scheduler.get_last_lr()[0],
                "seconds": elapsed,
            }
        )

        if improved:
            best_mae = val_mae
            # test_mae is provisionally the best val MAE; ml/eval.py rewrites it
            # with the real held-out test MAE once training finishes.
            save_checkpoint(model, best_mae, args.checkpoint)
            print(f"  saved checkpoint -> {args.checkpoint}")

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"\nBest val MAE: {best_mae:.3f}\nHistory -> {HISTORY_PATH}")
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
    train(parser.parse_args())


if __name__ == "__main__":
    main()
