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
import math
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

# Decodes we know how to serve. A checkpoint may declare its own via
# meta["decode"]; see TorchPredictor for why we honour it.
SUPPORTED_DECODES = ("median", "expectation")
DEFAULT_DECODE = "median"

# Accuracy figures, keyed by the exact artifact they were measured on.
#
# Accuracy is a property of a specific set of weights, not of "the model", so
# swapping the checkpoint must invalidate the numbers rather than silently
# relabel another model's performance. An artifact absent from this table
# reports nulls and says why.
#
# Every figure here is measured end to end through THIS server's path (YuNet
# detect -> our crop -> model -> decode), not copied from a training report.
MEASURED_ACCURACY_BY_DIGEST = {
    # Real chronological ground truth: AgeDB + APPA-REAL + FG-NET, no UTKFace
    # and no DEX-derived labels anywhere in training or evaluation.
    "fb629f49987a": {
        "real_age_mae": 6.34,
        "real_age_corpus": "AgeDB + APPA-REAL + FG-NET held-out test, n=3807",
        "in_corpus_mae_utkface": None,
        "accuracy_note": (
            "real_age_mae is the error against real chronological age, measured "
            "end to end through this server's detect-crop-decode path. This "
            "model never saw UTKFace, so there is no in-corpus DEX figure and "
            "none should be invented for it."
        ),
        "user_facing": {
            "typical_error_years": 6.3,
            "typical_error_basis": "against a person's real age",
            # About 30% of true under-18s display as 18 or over. Measured end
            # to end on these weights; the observable converse is 4.1%, and the
            # ~7x divergence is pure base rate.
            "gating_under18_shown_adult_pct": 29.6,
            # No band caveat is supportable on this model. Binned by DISPLAYED
            # age the largest bias split at any threshold is 1.52 years, and
            # the residual precision gradient is already carried per face by
            # the confidence bar (r = -0.343). See overlay.ts for the full
            # derivation and the standard a future caveat must clear.
            "caveat": None,
        },
    },
    # The original UTKFace model. Kept because it is still a supported fallback.
    # Its two figures differ by ~4 years and measure different things, which is
    # the whole reason this table exists.
    "56894c480044": {
        # Measured end to end through this server on the SAME held-out split,
        # with the same harness and crop, as the real-age model's 6.34. The two
        # user-facing figures sit side by side under a toggle, so they have to
        # be like-for-like or the comparison the UI invites is invalid. The ml/
        # side independently gets 9.127 on this split.
        "real_age_mae": 9.11,
        "real_age_corpus": "AgeDB + APPA-REAL + FG-NET held-out test, n=3807",
        # Retained, but NOT the user-facing number: a different corpus. It is
        # a valid measurement of these weights, just not comparable to 6.34.
        "real_age_mae_appa_all": 8.52,
        "real_age_corpus_appa_all": "APPA-REAL, n=7534",
        "in_corpus_mae_utkface": 4.762,
        "accuracy_note": (
            "in_corpus_mae_utkface measures agreement with UTKFace's "
            "DEX-derived labels, not accuracy. real_age_mae is the error "
            "against real chronological age, measured on the same split as "
            "the real-age model so the two are directly comparable."
        ),
        "user_facing": {
            "typical_error_years": 9.1,
            "typical_error_basis": (
                "against a person's real age -- this model predicts how old "
                "someone looks, which is not the same target"
            ),
            "gating_under18_shown_adult_pct": 40.3,
            # This caveat IS supportable on these weights and is false on the
            # real-age model, which is exactly why it lives next to a digest
            # rather than in the UI. Re-binned by DISPLAYED age on the real-GT
            # held-out split: >=40 n=1852 MAE 11.62 bias +5.70, against <40
            # n=1966 MAE 6.78 bias -0.55. Independently reproduced to the
            # decimal by the ml/ side from the same dump.
            "caveat": {
                "min_age": 40,
                "short": "reads high",
                "long": (
                    "Faces shown as 40 or over tend to read about 6 years high "
                    "on this model, and are roughly twice as imprecise as "
                    "younger ones. The estimate is not corrected for it."
                ),
            },
        },
    },
}


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
    #: How the 101-bin distribution becomes a point estimate. Overridden per
    #: checkpoint by ``TorchPredictor``, which honours ``meta["decode"]``.
    decode = DEFAULT_DECODE

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


def _as_float(value) -> float | None:
    """Coerce a metadata value to float, tolerating absence and junk.

    Checkpoint metadata is written by another codebase, so a field may be
    missing, ``None``, or a string. A malformed value must not take the server
    down at startup -- it degrades to "not declared", which the caller already
    handles.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

        # Honour the checkpoint's declared decode. This exists because the right
        # decode is a property of how the model was TRAINED, not a fixed choice:
        # with label smoothing a uniform pedestal (own expectation exactly 50)
        # drags an expectation decode toward the middle, so the median wins by
        # ~0.2-0.4 years. Remove the smoothing and the sign flips -- expectation
        # then wins by ~0.12. Hardcoding either one silently leaves accuracy on
        # the table the next time training changes.
        declared = self.meta.get("decode")
        if declared is None:
            self.decode = DEFAULT_DECODE
        elif str(declared) in SUPPORTED_DECODES:
            self.decode = str(declared)
            if self.decode != DEFAULT_DECODE:
                log.warning(
                    "Checkpoint declares decode=%r, overriding the default %r. "
                    "Serving the checkpoint's choice.",
                    self.decode,
                    DEFAULT_DECODE,
                )
        else:
            self.decode = DEFAULT_DECODE
            log.warning(
                "Checkpoint declares unsupported decode=%r; falling back to %r. "
                "Supported: %s.",
                declared,
                DEFAULT_DECODE,
                ", ".join(SUPPORTED_DECODES),
            )

        # Check the serving crop against the one the checkpoint was TRAINED
        # with. The crop is the single largest preprocessing lever we have, and
        # a mismatch is silent: every face is simply framed differently from
        # training, degrading accuracy with no error and no visible symptom.
        # Checkpoints now record `crop_margin`, so this is checkable rather than
        # a comment asking someone to remember. We warn instead of overriding --
        # CROP_MARGIN is deliberately tunable at runtime for A/B work, and
        # silently ignoring an operator's explicit setting would be its own bug.
        self.trained_crop_margin = _as_float(self.meta.get("crop_margin"))
        if self.trained_crop_margin is not None and not math.isclose(
            self.trained_crop_margin, config.CROP_MARGIN, abs_tol=1e-6
        ):
            log.warning(
                "CROP MARGIN MISMATCH: checkpoint was trained with crop_margin=%s "
                "but this server is serving crop_margin=%s. Every crop will be "
                "framed differently from training, which degrades accuracy "
                "silently. Set CROP_MARGIN=%s unless you are deliberately "
                "sweeping it.",
                self.trained_crop_margin,
                config.CROP_MARGIN,
                self.trained_crop_margin,
            )

        log.info(
            "Loaded age model %s (%d bins, input %d) from %s; "
            "sha256:%s, decode: %s, recorded test MAE: %s",
            self.model_name,
            self.num_bins,
            self.input_size,
            path,
            self._digest,
            self.decode,
            meta.get("test_mae", "n/a"),
        )

    def describe_checkpoint(self) -> dict | None:
        test_mae = self.meta.get("test_mae")
        info = {
            "path": str(self.model_path),
            "sha256": self._digest,
            "bytes": self._size_bytes,
            # The checkpoint's own recorded figure, NOT the accuracy of what we
            # serve. Reported under a name that cannot be mistaken for current
            # accuracy. `recorded_test_mae_decode` says which decode produced
            # it, since that alone is worth ~0.2-0.4 years.
            "recorded_test_mae": round(float(test_mae), 4) if test_mae is not None else None,
            "recorded_test_mae_decode": str(self.meta.get("decode", "expectation")),
            "serving_decode": self.decode,
            # What the recorded figure is an error *against*. A number is
            # meaningless without this: 5.5472 against DEX-estimated apparent
            # age and 6.393 against real chronological age are not comparable,
            # and the smaller one is the weaker result. Passed through verbatim
            # so the artifact's own provenance travels with its number instead
            # of being re-narrated here, where it would go stale.
            "recorded_test_mae_corpus": self.meta.get("corpus"),
            "label_semantics": self.meta.get("label_semantics"),
            # Present only while a figure is provisional. train.py stamps the
            # best *validation* MAE into test_mae and relies on eval.py to
            # overwrite it; until that happens the field is a selection-set
            # score flattering itself. These keys let that announce itself.
            "recorded_test_mae_source": self.meta.get("test_mae_source"),
            "recorded_val_mae": _as_float(self.meta.get("val_mae")),
            # Experiment intermediates carry a role. Surfaced so an artifact
            # that exists to demonstrate a point is never mistaken for the
            # published model in a pasted /health payload.
            "role": self.meta.get("role"),
            "trained_crop_margin": self.trained_crop_margin,
            "serving_crop_margin": config.CROP_MARGIN,
            "crop_margin_matches_training": (
                None
                if self.trained_crop_margin is None
                else math.isclose(self.trained_crop_margin, config.CROP_MARGIN, abs_tol=1e-6)
            ),
        }

        # Our own measured accuracy figures are pinned to the exact artifact
        # they were measured on. They are NOT properties of "the model" -- point
        # AGE_MODEL_PATH at a different checkpoint and they become fiction. This
        # is the same failure the whole accuracy-labelling exercise was about,
        # so it fails closed: an unrecognised artifact reports nulls and says
        # why, rather than confidently serving another model's numbers.
        measured = MEASURED_ACCURACY_BY_DIGEST.get(self._digest)
        if measured is not None:
            info.update(measured)
        else:
            info.update(
                {
                    "real_age_mae": None,
                    "real_age_corpus": None,
                    "in_corpus_mae_utkface": None,
                    # Fails closed: no figures and NO CAVEAT for an artifact we
                    # have not measured. A caveat is a directional claim about
                    # specific weights; asserting one we have not verified is
                    # the same class of error as reporting another model's MAE.
                    "user_facing": None,
                    "accuracy_note": (
                        "Unmeasured artifact: we have no end-to-end accuracy "
                        "figures for this checkpoint (known: "
                        f"{', '.join(sorted(MEASURED_ACCURACY_BY_DIGEST))}). Its "
                        "own recorded_test_mae is not comparable across corpora "
                        "-- real-age and DEX-label MAE measure different things. "
                        "Re-run server/tools/eval_end_to_end.py before quoting "
                        "any number."
                    ),
                }
            )
        return info

    def _estimate(self, batch: np.ndarray) -> Decoded:
        torch = self._torch
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(batch))
            probs = torch.softmax(logits.float(), dim=1)

            # Soft-expectation decode. Always computed: it is the point estimate
            # when the checkpoint declares decode="expectation", and otherwise
            # it is retained for comparison.
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

        # The interval stays the CDF quantiles under either decode: it needs no
        # symmetry assumption, and sigma is inflated by any label-smoothing
        # pedestal in exactly the way the point estimate is.
        age = expectation if self.decode == "expectation" else median

        return Decoded(
            age=age.numpy(),
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
