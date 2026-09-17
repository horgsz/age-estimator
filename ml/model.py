"""DEX-style soft-expectation age regressor on a MobileNetV3-Small backbone."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import nn

BACKBONE = "mobilenetv3_small_100"
PRETRAINED_TAG = "lamb_in1k"
NUM_BINS = 101
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "age_model.pt"
PRETRAINED_DIR = REPO_ROOT / "checkpoints" / "pretrained"
PRETRAINED_URL = (
    f"https://huggingface.co/timm/{BACKBONE}.{PRETRAINED_TAG}"
    "/resolve/main/model.safetensors"
)


def ensure_pretrained_weights(
    dest: Path | None = None, url: str = PRETRAINED_URL
) -> Path:
    """Fetch the ImageNet backbone weights, returning the local file path.

    Hugging Face's ``us.aws.cdn.hf.co`` edge serves an incomplete certificate
    chain that Python's ``ssl`` module rejects, so ``timm``'s own downloader
    fails here. ``curl`` completes the chain from the system trust store, so we
    mirror the weights locally once and load them with ``pretrained_cfg_overlay``.
    """
    dest = dest or PRETRAINED_DIR / f"{BACKBONE}.{PRETRAINED_TAG}.safetensors"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if shutil.which("curl") is None:
        raise RuntimeError(
            f"curl not available and {dest} is missing; download {url} manually."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pretrained backbone -> {dest}")
    subprocess.run(["curl", "-sL", "--fail", "-o", str(dest), url], check=True)
    return dest


class AgeEstimator(nn.Module):
    """MobileNetV3-Small-100 with a 101-way head over ages 0..100.

    The predicted age is the soft expectation ``sum_i softmax(logits)_i * i``
    (DEX, Rothe et al.), which yields a continuous estimate from a classifier
    and a usable uncertainty via the distribution's standard deviation.
    """

    def __init__(
        self,
        backbone: str = BACKBONE,
        num_bins: int = NUM_BINS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.backbone_name = backbone
        overlay = None
        if pretrained:
            overlay = {"file": str(ensure_pretrained_weights())}
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_bins,
            pretrained_cfg_overlay=overlay,
        )
        self.register_buffer(
            "bin_centers", torch.arange(num_bins, dtype=torch.float32), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def expectation(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(expected_age, std)`` for a batch of logits."""
        probs = F.softmax(logits.float(), dim=1)
        centers = self.bin_centers.to(probs.device)
        mean = (probs * centers).sum(dim=1)
        var = (probs * (centers.unsqueeze(0) - mean.unsqueeze(1)) ** 2).sum(dim=1)
        return mean, var.clamp_min(0).sqrt()

    def median(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(median_age, iqr)`` for a batch of logits.

        The soft-expectation decode averages over the whole 101-bin support, so
        tail mass drags it toward the middle. Two things put tail mass there:
        ``label_smoothing=0.1`` trains a uniform pedestal onto every bin (whose
        own expectation is 50), and the predictive distribution is genuinely
        right-skewed for young faces because mass cannot extend below bin 0.
        The median is robust to both, and measurably so -- see ``ml/README.md``.
        """
        probs = F.softmax(logits.float(), dim=1)
        cdf = probs.cumsum(dim=1)
        med = (cdf < 0.5).sum(dim=1).to(probs.dtype)
        q25 = (cdf < 0.25).sum(dim=1).to(probs.dtype)
        q75 = (cdf < 0.75).sum(dim=1).to(probs.dtype)
        return med, q75 - q25

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience inference helper returning ``(age, uncertainty)``."""
        return self.expectation(self(x))


def build_meta(
    test_mae: float,
    decode: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Artifact-contract metadata block. Keep these keys and names stable.

    ``decode`` names the decode that ``test_mae`` was measured under. It exists
    because an artifact carrying a bare number is ambiguous: the same weights
    score 6.39 under one decode and 9.13 under another, and a consumer reading
    ``test_mae`` has no way to tell which it is holding. Omitted for the
    original artifact, whose contract predates the key.
    """
    meta: dict[str, object] = {
        "backbone": BACKBONE,
        "num_bins": NUM_BINS,
        "input_size": INPUT_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "test_mae": float(test_mae),
    }
    if decode is not None:
        meta["decode"] = decode
    if extra:
        meta.update(extra)
    return meta


def save_checkpoint(
    model: nn.Module,
    test_mae: float,
    path: Path = CHECKPOINT_PATH,
    decode: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    """Write the artifact with a *bare timm* state dict.

    ``AgeEstimator`` is a passthrough wrapper, so ``model.state_dict()`` would
    prefix every key with ``backbone.``. ``meta`` advertises the backbone name
    and bin count, so a consumer is entitled to do

        timm.create_model(meta["backbone"], num_classes=meta["num_bins"])
        model.load_state_dict(ckpt["state_dict"])

    and that must work without key surgery. Strip the prefix on save so the
    artifact matches the documented contract rather than our module layout.
    """
    inner = model.backbone if isinstance(model, AgeEstimator) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    _warn_if_clobbering(path)
    torch.save(
        {
            "state_dict": inner.state_dict(),
            "meta": build_meta(test_mae, decode=decode, extra=extra),
        },
        path,
    )


# Paths whose contents are published artifacts that other sessions consume.
# Writing one from a training run or a variant experiment is almost always a
# mistake: it is how a published file silently acquires different weights.
PUBLISHED_PATHS = {"age_model.pt", "age_model_realgt.pt"}


def _warn_if_clobbering(path: Path) -> None:
    """Refuse to let a training run overwrite a published artifact in place.

    Learned the hard way: ``age_model_realgt.pt`` served as both a variant
    output *and* the publish target, so re-using it as the publish target
    destroyed the variant it had been holding. The weights were unrecoverable
    without a retrain, and nothing complained at the time -- the file was still
    a valid checkpoint, just not the one the reports referenced. Keep publish
    paths and experiment paths disjoint; this guard enforces that.
    """
    if path.name in PUBLISHED_PATHS and path.exists():
        raise RuntimeError(
            f"{path.name} is a published artifact; refusing to overwrite it "
            f"from a training run. Write the variant to a distinct name "
            f"(e.g. {path.stem}_<variant>{path.suffix}) and publish explicitly."
        )


def _strip_backbone_prefix(state: dict) -> dict:
    """Accept both the bare and legacy ``backbone.``-prefixed layouts."""
    if any(k.startswith("backbone.") for k in state):
        return {k.removeprefix("backbone."): v for k, v in state.items()}
    return state


def load_checkpoint(
    path: Path = CHECKPOINT_PATH, map_location: str | torch.device = "cpu"
) -> tuple[AgeEstimator, dict[str, object]]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}. Train first.")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    meta = payload["meta"]
    model = AgeEstimator(
        backbone=meta["backbone"], num_bins=meta["num_bins"], pretrained=False
    )
    model.backbone.load_state_dict(_strip_backbone_prefix(payload["state_dict"]))
    model.eval()
    return model, meta
