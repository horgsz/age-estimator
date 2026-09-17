"""The set of models this server can serve, and which one a request gets.

Two checkpoints are kept resident and switchable per request. They are not two
versions of one model: they were trained on different corpora, against
different labels, and they answer *different questions*.

    key         predicts                      trained on
    ---------   ---------------------------   ---------------------------------
    real        how old this person IS        AgeDB + APPA-REAL + FG-NET,
                                              real chronological ages
    apparent    how old this person LOOKS     UTKFace, DEX-estimated apparent
                                              ages

Neither is "the accurate one". Their recorded MAEs (6.393 and 5.5472) are not
comparable, because they are errors against different targets -- the smaller
number is the one measured against the easier, softer target. Anything that
presents them as a single accuracy ranking is wrong, which is why the catalog
carries a `question` rather than just a name.

Identity is keyed by content digest, not by filename
----------------------------------------------------
Each entry declares the sha256 prefix it expects. A slot that loads a file whose
digest belongs to a *different* catalog entry is refused outright rather than
served under the wrong label.

That is not hypothetical tidiness: this artifact has been silently republished
to the same path twice during this project, and the entire accuracy-labelling
exercise exists because a number that is correct about one set of weights is a
lie about another. Serving apparent-age weights under a label that says "how old
this person actually is" is the same bug with a UI attached, so the registry
makes it unrepresentable instead of merely unlikely.

An *unrecognised* digest is a weaker case -- it may simply be a retrain -- so it
is served, but stripped: neutral labelling, no accuracy figures, and no caveat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import config
from .detector import FaceDetector
from .predictor import AgePredictor, StubPredictor, TorchPredictor

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    """One selectable model, loaded or explained-away."""

    key: str
    label: str
    question: str
    explanation: str
    predictor: AgePredictor | None = None
    path: str | None = None
    unavailable_reason: str | None = None
    #: False when the loaded artifact is not the one this slot expects. The
    #: predictor still works; its labelling and accuracy claims do not apply.
    identity_verified: bool = True

    @property
    def available(self) -> bool:
        return self.predictor is not None

    def describe(self) -> dict:
        """Public description for ``GET /health``."""
        info: dict = {
            "key": self.key,
            "label": self.label,
            "question": self.question,
            "explanation": self.explanation,
            "available": self.available,
            "identity_verified": self.identity_verified,
        }
        if self.unavailable_reason:
            info["unavailable_reason"] = self.unavailable_reason
        if self.predictor is not None:
            info["model"] = self.predictor.model_name
            info["stub"] = self.predictor.is_stub
            info["checkpoint"] = self.predictor.describe_checkpoint()
        else:
            info["model"] = None
            info["stub"] = None
            info["checkpoint"] = None
        return info


class ModelRegistry:
    """Keyed collection of resident predictors, with one designated default."""

    def __init__(self, entries: list[ModelEntry], default_key: str) -> None:
        self._entries = {e.key: e for e in entries}
        self._default_key = default_key

    @property
    def keys(self) -> list[str]:
        return list(self._entries)

    @property
    def available_keys(self) -> list[str]:
        return [k for k, e in self._entries.items() if e.available]

    @property
    def default_key(self) -> str:
        return self._default_key

    def entry(self, key: str | None) -> ModelEntry:
        """Look up a model by key, falling back to the default for ``None``.

        Raises :class:`KeyError` for an unknown key and :class:`LookupError`
        for a known-but-unloaded one, so the caller can distinguish "you asked
        for something that does not exist" (client error) from "that model is
        not on this machine" (environment), and say which.
        """
        resolved = self._default_key if key is None else key
        entry = self._entries.get(resolved)
        if entry is None:
            raise KeyError(resolved)
        if not entry.available:
            raise LookupError(entry.unavailable_reason or f"Model {resolved!r} is not loaded")
        return entry

    def predictor(self, key: str | None = None) -> AgePredictor:
        return self.entry(key).predictor  # type: ignore[return-value]

    def describe(self) -> dict:
        return {
            "default": self._default_key,
            "available": self.available_keys,
            "models": [e.describe() for e in self._entries.values()],
        }


def _resolve_path(spec: config.ModelSpec) -> str | None:
    """Where this model's checkpoint should be, honouring env overrides."""
    import os

    explicit = os.environ.get(spec.env_var)
    if explicit:
        return explicit
    # AGE_MODEL_PATH keeps its old meaning -- "the checkpoint to serve" -- but
    # now applies only to the default slot. Applying it to every slot would
    # silently load one artifact under both labels.
    #
    # Only an explicitly *set* env var counts. config.AGE_MODEL_PATH always
    # holds a computed default, so honouring it here made the default slot
    # ignore AGE_MODEL_DIR and look for a file that was never there.
    if spec.key == config.DEFAULT_MODEL_KEY:
        legacy = os.environ.get("AGE_MODEL_PATH", "").strip()
        if legacy:
            return legacy
    candidate = Path(config.MODEL_DIR) / spec.filename
    return str(candidate)


def _load_entry(spec: config.ModelSpec, detector: FaceDetector) -> ModelEntry:
    base = {
        "key": spec.key,
        "label": spec.label,
        "question": spec.question,
        "explanation": spec.explanation,
    }
    path = _resolve_path(spec)

    if not path or not Path(path).exists():
        log.warning("Model %r unavailable: no checkpoint at %s", spec.key, path)
        return ModelEntry(
            **base,
            unavailable_reason=f"No checkpoint found at {path}",
        )

    try:
        predictor = TorchPredictor(path, detector=detector)
    except Exception as exc:
        log.error("Model %r failed to load from %s: %s", spec.key, path, exc)
        return ModelEntry(**base, unavailable_reason=f"Failed to load {path}: {exc}")

    digest = predictor.describe_checkpoint().get("sha256")  # type: ignore[union-attr]

    # A file whose digest belongs to a different catalog entry must never be
    # served under this one's label. Refusing is safe: the other slot will load
    # the same artifact under its correct key if it is also pointed at it.
    wrong_slot = config.digest_owner(digest)
    if wrong_slot is not None and wrong_slot != spec.key:
        log.error("=" * 72)
        log.error("REFUSING to serve %r under model key %r.", path, spec.key)
        log.error(
            "Its digest %s is the %r model. Serving it here would label "
            "'%s' with an answer to a different question.",
            digest,
            wrong_slot,
            spec.question,
        )
        log.error("=" * 72)
        return ModelEntry(
            **base,
            unavailable_reason=(
                f"Checkpoint at {path} (sha256 {digest}) is the {wrong_slot!r} "
                f"model, not {spec.key!r}. Refusing to serve it under the wrong "
                "label; set the correct path or remove the override."
            ),
        )

    verified = digest == spec.expected_digest
    if not verified:
        log.warning("=" * 72)
        log.warning(
            "Model %r loaded an UNRECOGNISED artifact (sha256 %s, expected %s).",
            spec.key,
            digest,
            spec.expected_digest,
        )
        log.warning(
            "Serving it, but with no accuracy figures, no caveat, and neutral "
            "labelling: those are properties of specific weights, not of a slot."
        )
        log.warning("=" * 72)
        base = {
            **base,
            "label": f"{spec.label} (unverified build)",
            "explanation": (
                "This checkpoint is not the artifact this slot was measured on, "
                "so its accuracy and its known limitations are unknown. Treat "
                "the numbers as unvalidated."
            ),
        }

    log.info("Model %r ready: %s (sha256 %s)", spec.key, path, digest)
    return ModelEntry(**base, predictor=predictor, path=path, identity_verified=verified)


def build_registry(detector: FaceDetector | None = None) -> ModelRegistry:
    """Load every catalog model that is present, keeping all of them resident.

    Both checkpoints are ~6.6 MB and inference is milliseconds, so there is
    nothing to gain by loading on demand and it would make the first request
    after every toggle slow. Missing checkpoints are reported, not fatal: the
    app must still work with whichever artifacts are on the machine.
    """
    shared = detector or FaceDetector()
    entries = [_load_entry(spec, shared) for spec in config.MODEL_CATALOG]

    if not any(e.available for e in entries):
        log.warning("=" * 72)
        log.warning("NO MODEL CHECKPOINTS FOUND -- serving the STUB. Ages are FAKE.")
        log.warning("Looked in %s", config.MODEL_DIR)
        log.warning("=" * 72)
        stub_key = config.DEFAULT_MODEL_KEY
        entries = [
            ModelEntry(
                key=spec.key,
                label=spec.label,
                question=spec.question,
                explanation=spec.explanation,
                predictor=StubPredictor(detector=shared) if spec.key == stub_key else None,
                unavailable_reason=(
                    None
                    if spec.key == stub_key
                    else "No checkpoint found; only the stub is available."
                ),
            )
            for spec in config.MODEL_CATALOG
        ]

    default_key = config.DEFAULT_MODEL_KEY
    by_key = {e.key: e for e in entries}
    if not by_key[default_key].available:
        fallback = next((e.key for e in entries if e.available), default_key)
        if fallback != default_key:
            log.warning(
                "Default model %r is unavailable; falling back to %r.",
                default_key,
                fallback,
            )
            default_key = fallback

    return ModelRegistry(entries, default_key)
