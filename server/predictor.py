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

import numpy as np

from . import config, preprocessing
from .detector import FaceDetector

log = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]

# std (years) at which confidence is 0.5. Larger => more forgiving.
CONFIDENCE_STD_SCALE = 6.0

MAX_AGE = 100.0


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


def confidence_from_std(std: float) -> float:
    """Monotonically decreasing map from prediction std (years) to [0, 1]."""
    std = max(float(std), 0.0)
    return round(1.0 / (1.0 + std / CONFIDENCE_STD_SCALE), 4)


def build_result(bbox: BBox, age: float, std: float) -> FaceResult:
    """Assemble a :class:`FaceResult` from a point estimate and its std."""
    age = float(np.clip(age, 0.0, MAX_AGE))
    std = max(float(std), 0.0)
    return FaceResult(
        bbox=tuple(int(v) for v in bbox),  # type: ignore[arg-type]
        age=round(age, 1),
        low=round(max(0.0, age - std), 1),
        high=round(min(MAX_AGE, age + std), 1),
        confidence=confidence_from_std(std),
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
        ages, stds = self.estimate_batch(batch)
        return [build_result(box, a, s) for box, a, s in zip(boxes, ages, stds)]

    def estimate_batch(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the model over a batch of preprocessed crops (float32, NCHW)."""
        return self._estimate(batch)

    def _estimate(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def _estimate(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ages, stds = [], []
        for crop in batch:
            # Quantise before hashing so imperceptible float noise (e.g. a
            # re-encoded JPEG) does not flip the result.
            digest = hashlib.sha256(np.round(crop, 3).tobytes()).digest()
            a = int.from_bytes(digest[:4], "big")
            b = int.from_bytes(digest[4:8], "big")
            ages.append(6.0 + (a % 7000) / 100.0)  # 6.0 .. 76.0
            stds.append(2.0 + (b % 900) / 100.0)  # 2.0 .. 11.0
        return np.asarray(ages, dtype=np.float32), np.asarray(stds, dtype=np.float32)


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
            "test_mae": round(float(test_mae), 4) if test_mae is not None else None,
        }

    def _estimate(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(batch))
            probs = torch.softmax(logits.float(), dim=1)
            ages = (probs * self._bins).sum(dim=1)
            var = (probs * (self._bins.unsqueeze(0) - ages.unsqueeze(1)) ** 2).sum(dim=1)
            stds = torch.sqrt(torch.clamp(var, min=0.0))
        return ages.numpy(), stds.numpy()


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
