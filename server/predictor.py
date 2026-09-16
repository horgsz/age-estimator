"""Age prediction.

Two implementations share one interface so the whole stack can be developed and
tested before the real checkpoint exists:

* :class:`StubPredictor` -- deterministic pseudo-random ages derived from a hash
  of the face crop. Stable across calls for the same pixels.
* :class:`TorchPredictor` -- loads ``checkpoints/age_model.pt`` and runs the real
  DEX-style soft-expectation regressor.

Swapping the stub for the real model is a one-class change; everything else in
the server is identical.

Checkpoint contract (pinned with the ``ml/`` side of the project)::

    torch.save({
        "state_dict": <timm mobilenetv3_small_100 with a 101-way head>,
        "meta": {"backbone": "mobilenetv3_small_100", "num_bins": 101,
                 "input_size": 224, "mean": [...], "std": [...],
                 "test_mae": <float>},
    }, "checkpoints/age_model.pt")

The head is a distribution over ages 0..100. The point estimate is the
soft expectation ``sum_i softmax(logits)_i * i`` and the uncertainty is the
standard deviation of that same distribution.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import config, preprocessing
from .detector import FaceDetector

log = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]

# Interval half-width (years) at which confidence is 0.5. Larger => more
# forgiving. Confidence is a function of the *width* of the reported interval;
# this is expressed as a half-width so the numbers stay comparable with the
# mean-centred +/- 1 sigma interval this replaced.
CONFIDENCE_STD_SCALE = 6.0

MAX_AGE = 100.0

# Quantiles used for the reported interval. 0.16/0.84 is the +/- 1 sigma mass of
# a normal, so the width stays on the same scale as the old mean +/- std range,
# but it is read straight off the CDF and so needs no symmetry assumption --
# which matters because these distributions are visibly skewed at the tails.
LOW_Q, HIGH_Q = 0.16, 0.84


@dataclass(frozen=True)
class FaceResult:
    """One detected face and its age estimate, in original-image coordinates."""

    bbox: BBox
    age: float
    low: float
    high: float
    confidence: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [int(v) for v in self.bbox]
        return d


class Decoded(NamedTuple):
    """One batch decoded from logits, every statistic kept.

    ``age`` is what ships. The rest are retained so the decode can be compared
    against alternatives later without another inference pass -- the choice of
    median over expectation was worth 0.7 years of MAE and was only findable
    because both were measurable on the same weights.
    """

    age: np.ndarray  # distribution median -- the shipped point estimate
    low: np.ndarray  # LOW_Q quantile
    high: np.ndarray  # HIGH_Q quantile
    expectation: np.ndarray  # soft-expectation decode (the old default)
    std: np.ndarray  # std about the mean


def confidence_from_width(width: float) -> float:
    """Monotonically decreasing map from interval width (years) to [0, 1]."""
    width = max(float(width), 0.0)
    return round(1.0 / (1.0 + width / (2.0 * CONFIDENCE_STD_SCALE)), 4)


def confidence_from_std(std: float) -> float:
    """Back-compat shim: confidence of a symmetric ``+/- std`` interval."""
    return confidence_from_width(2.0 * max(float(std), 0.0))


def build_result(bbox: BBox, age: float, low: float, high: float) -> FaceResult:
    """Assemble a :class:`FaceResult` from a point estimate and its interval."""
    age = float(np.clip(age, 0.0, MAX_AGE))
    low = float(np.clip(low, 0.0, MAX_AGE))
    high = float(np.clip(high, 0.0, MAX_AGE))
    # The median can sit outside a degenerate interval; keep the range coherent.
    low, high = min(low, age), max(high, age)
    return FaceResult(
        bbox=tuple(int(v) for v in bbox),  # type: ignore[arg-type]
        age=round(age, 1),
        low=round(low, 1),
        high=round(high, 1),
        confidence=confidence_from_width(high - low),
    )


class AgePredictor:
    """Detect faces then estimate each one's age.

    Subclasses only implement :meth:`_estimate`, which receives a batch of
    already-preprocessed crops (float32, NCHW) and returns ``(ages, stds)``.
    """

    #: Reported by ``GET /health``.
    model_name = "base"
    is_stub = True

    def __init__(self, model_path: str | None = None, detector: FaceDetector | None = None) -> None:
        self.model_path = model_path
        self._detector = detector

    def describe_checkpoint(self) -> dict | None:
        """Identify the loaded artifact, for ``GET /health`` and the harness.

        The checkpoint was once republished to the same path mid-evaluation with
        a different model inside it, which silently invalidated an in-flight
        comparison. Surfacing a content hash and the checkpoint's own claimed
        test MAE makes "which model is actually live?" answerable at a glance
        instead of by inference from the numbers.
        """
        return None

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    def predict(self, image_bgr: np.ndarray) -> list[FaceResult]:
        """Detect faces in ``image_bgr`` and age each one."""
        return self.predict_boxes(image_bgr, self.detector.detect(image_bgr))

    def predict_boxes(
        self,
        image_bgr: np.ndarray,
        boxes: Sequence[BBox],
        margin: float | None = None,
    ) -> list[FaceResult]:
        """Age pre-detected ``boxes``, skipping detection.

        This is the crop + model half of :meth:`predict`, split out so callers
        that already have boxes -- notably the offline eval harness, which
        detects once and then sweeps several crop margins -- run the *exact*
        same path the server does rather than a lookalike reimplementation.
        """
        if not boxes:
            return []

        batch = np.stack(
            [preprocessing.preprocess_face(image_bgr, box, margin) for box in boxes]
        ).astype(np.float32)
        d = self.estimate_batch(batch)
        return [
            build_result(box, a, lo, hi)
            for box, a, lo, hi in zip(boxes, d.age, d.low, d.high)
        ]

    def estimate_batch(self, batch: np.ndarray) -> "Decoded":
        """Run the model over a batch of preprocessed crops (float32, NCHW)."""
        return self._estimate(batch)

    def _estimate(self, batch: np.ndarray) -> "Decoded":
        raise NotImplementedError


class StubPredictor(AgePredictor):
    """Deterministic placeholder used until the real checkpoint lands.

    The age is derived from a SHA-256 hash of the preprocessed crop, so the same
    face pixels always produce the same number -- which keeps the UI and the
    tests stable -- while different faces produce different numbers.
    """

    model_name = "stub"
    is_stub = True

    def __init__(self, model_path: str | None = None, detector: FaceDetector | None = None) -> None:
        super().__init__(model_path=None, detector=detector)

    def _estimate(self, batch: np.ndarray) -> Decoded:
        ages, stds = [], []
        for crop in batch:
            # Quantise before hashing so imperceptible float noise (e.g. a
            # re-encoded JPEG) does not flip the result.
            digest = hashlib.sha256(np.round(crop, 3).tobytes()).digest()
            a = int.from_bytes(digest[:4], "big")
            b = int.from_bytes(digest[4:8], "big")
            ages.append(6.0 + (a % 7000) / 100.0)  # 6.0 .. 76.0
            stds.append(2.0 + (b % 900) / 100.0)  # 2.0 .. 11.0
        age = np.asarray(ages, dtype=np.float32)
        std = np.asarray(stds, dtype=np.float32)
        # The stub has no distribution, so it fakes a symmetric one.
        return Decoded(
            age=age,
            low=np.clip(age - std, 0.0, MAX_AGE),
            high=np.clip(age + std, 0.0, MAX_AGE),
            expectation=age,
            std=std,
        )


_WRAPPER_PREFIXES = ("backbone.", "model.", "module.", "net.")


def _strip_wrapper_prefix(state_dict: dict, model) -> dict:
    """Unwrap a state dict saved from a module that *contains* the timm model.

    The contract says the checkpoint holds a state dict for a bare timm
    ``mobilenetv3_small_100``. In practice training code usually wraps it
    (``self.backbone = timm.create_model(...)``, ``nn.DataParallel``, ...), which
    prefixes every key. The tensors are identical, so refusing to load would be
    pedantry — but silently loading the *wrong* tensors would be far worse, so
    only strip a prefix if doing so actually makes the keys line up.
    """
    if not state_dict:
        return state_dict

    expected = set(model.state_dict())
    if set(state_dict) & expected:
        return state_dict

    for prefix in _WRAPPER_PREFIXES:
        if not all(key.startswith(prefix) for key in state_dict):
            continue
        stripped = {key[len(prefix) :]: value for key, value in state_dict.items()}
        if set(stripped) >= expected:
            # Loud, not a quiet note: this is a live deviation from the pinned
            # checkpoint contract. If ml/ ever fixes it at source this banner
            # disappears, and if some *other* drift appears it is visible rather
            # than silently absorbed.
            log.warning("=" * 72)
            log.warning(
                "CHECKPOINT CONTRACT DEVIATION: state dict keys are prefixed %r.",
                prefix,
            )
            log.warning(
                "The contract specifies a bare timm state dict; this one was saved "
                "from a wrapper module."
            )
            log.warning("Stripping the prefix. The tensors are identical, so this is safe.")
            log.warning("=" * 72)
            return stripped

    return state_dict


class TorchPredictor(AgePredictor):
    """Real model: timm backbone + 101-way DEX head, soft-expectation decode."""

    is_stub = False

    def __init__(self, model_path: str | None = None, detector: FaceDetector | None = None) -> None:
        super().__init__(model_path=model_path or config.AGE_MODEL_PATH, detector=detector)

        import torch  # imported lazily so the stub path stays cheap
        import timm

        self._torch = torch

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError(
                f"{path} is not a valid age-model checkpoint (expected a dict with 'state_dict')"
            )

        meta = dict(ckpt.get("meta") or {})
        self.meta = meta
        self.model_name = str(meta.get("backbone", "mobilenetv3_small_100"))
        self.num_bins = int(meta.get("num_bins", config.NUM_BINS))
        self.input_size = int(meta.get("input_size", config.INPUT_SIZE))

        # Honour the checkpoint's own normalisation constants if it carries them,
        # so inference can never drift from training.
        if meta.get("mean") and meta.get("std"):
            config.IMAGENET_MEAN = tuple(float(v) for v in meta["mean"])  # type: ignore[assignment]
            config.IMAGENET_STD = tuple(float(v) for v in meta["std"])  # type: ignore[assignment]
        config.INPUT_SIZE = self.input_size

        self.model = timm.create_model(
            self.model_name, pretrained=False, num_classes=self.num_bins
        )
        self.model.load_state_dict(_strip_wrapper_prefix(ckpt["state_dict"], self.model))
        self.model.eval()

        self._digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        self._size_bytes = path.stat().st_size

        self._bins = torch.arange(self.num_bins, dtype=torch.float32)
        log.info(
            "Loaded age model %s (%d bins, input %d) from %s; "
            "sha256:%s, reported test MAE: %s",
            self.model_name,
            self.num_bins,
            self.input_size,
            path,
            self._digest,
            meta.get("test_mae", "n/a"),
        )

    def describe_checkpoint(self) -> dict | None:
        test_mae = self.meta.get("test_mae")
        return {
            "path": str(self.model_path),
            "sha256": self._digest,
            "bytes": self._size_bytes,
            # The checkpoint's own recorded figure, NOT the accuracy of what we
            # serve. It was measured with the soft-expectation decode; we ship
            # the median decode, which is materially better. Reported under a
            # name that cannot be mistaken for current accuracy.
            "recorded_test_mae": round(float(test_mae), 4) if test_mae is not None else None,
            "recorded_test_mae_decode": "expectation",
            "serving_decode": "median",
        }

    def _estimate(self, batch: np.ndarray) -> Decoded:
        torch = self._torch
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(batch))
            probs = torch.softmax(logits.float(), dim=1)

            # Soft-expectation decode: kept for comparison, no longer shipped.
            expectation = (probs * self._bins).sum(dim=1)
            var = (probs * (self._bins.unsqueeze(0) - expectation.unsqueeze(1)) ** 2).sum(dim=1)
            std = torch.sqrt(torch.clamp(var, min=0.0))

            # Quantile decode. `(cdf < q).sum()` counts the leading bins that
            # have not yet reached q, which *is* the index of the first bin that
            # has -- i.e. the smallest i with cumsum(p)[i] >= q.
            cdf = probs.cumsum(dim=1)
            last = probs.shape[1] - 1

            def q(level: float) -> "torch.Tensor":
                return torch.clamp((cdf < level).sum(dim=1), 0, last).float()

            median, low, high = q(0.5), q(LOW_Q), q(HIGH_Q)

        return Decoded(
            age=median.numpy(),
            low=low.numpy(),
            high=high.numpy(),
            expectation=expectation.numpy(),
            std=std.numpy(),
        )


def load_predictor(model_path: str | None = None) -> AgePredictor:
    """Return a :class:`TorchPredictor` if a checkpoint exists, else the stub."""
    path = model_path or os.environ.get("AGE_MODEL_PATH") or config.AGE_MODEL_PATH

    if path and Path(path).exists():
        try:
            return TorchPredictor(path)
        except Exception as exc:
            log.error("=" * 72)
            log.error("FAILED TO LOAD AGE MODEL from %s: %s", path, exc)
            log.error("Falling back to the STUB predictor -- ages are FAKE.")
            log.error("=" * 72)
            return StubPredictor()

    log.warning("=" * 72)
    log.warning("NO AGE MODEL CHECKPOINT FOUND at %s", path)
    log.warning("Using the STUB predictor: returned ages are FAKE placeholder values.")
    log.warning("Train the model (see ml/) or set AGE_MODEL_PATH to a real checkpoint.")
    log.warning("=" * 72)
    return StubPredictor()
