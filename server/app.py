"""FastAPI app exposing the age-estimation API.

Endpoints
---------
``GET  /health``    -> ``{"status": "ok", "model": "<backbone or 'stub'>", "stub": <bool>}``
``POST /estimate``  -> ``{"faces": [{"bbox": [x, y, w, h], "age", "low", "high", "confidence"}]}``

``bbox`` is in the ORIGINAL uploaded image's pixel coordinate space. An image
with no detectable face returns ``{"faces": []}`` with HTTP 200.

``POST /estimate`` accepts an optional ``crop_margin`` (query parameter or form
field) that overrides :data:`server.config.CROP_MARGIN` for that request only,
so crop margins can be A/B'd against real webcam photos without restarting. The
margin actually used comes back in the ``X-Crop-Margin`` response header; the
response body shape is unchanged.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .predictor import AgePredictor, load_predictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("age_estimator.server")

_predictor: AgePredictor | None = None


def get_predictor() -> AgePredictor:
    """Return the process-wide predictor, creating it on first use."""
    global _predictor
    if _predictor is None:
        _predictor = load_predictor()
    return _predictor


def set_predictor(predictor: AgePredictor | None) -> None:
    """Override the process-wide predictor (used by the tests)."""
    global _predictor
    _predictor = predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Re-read settings from the environment *before* the predictor is built, so
    # CROP_MARGIN and friends can be changed without touching code, and so a
    # checkpoint's own normalisation values (applied during load) survive.
    config.reload_from_env()

    predictor = get_predictor()
    log.info(
        "Age estimator ready (model=%s, stub=%s) settings=%s",
        predictor.model_name,
        predictor.is_stub,
        config.describe(),
    )
    if predictor.is_stub:
        log.warning("Serving FAKE ages from the stub predictor.")
    yield


app = FastAPI(title="age-estimator", version="0.1.0", lifespan=lifespan)

# CORS is the one setting bound at import rather than in `lifespan`, because
# Starlette captures the origin list when the middleware is added. Under uvicorn
# the module is imported after the process environment is set, so `CORS_ORIGINS`
# still comes from the environment as expected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    # Without this the browser silently hides X-Crop-Margin from fetch(): only
    # CORS-safelisted response headers are readable cross-origin by default, and
    # the Vite dev server is a different origin to the API.
    expose_headers=["X-Crop-Margin"],
)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject oversized bodies up front using the declared Content-Length."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > config.MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Upload too large (limit {config.MAX_UPLOAD_BYTES} bytes)"
                    },
                )
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
    return await call_next(request)


@app.get("/health")
async def health() -> dict:
    predictor = get_predictor()
    # `status`, `model` and `stub` are the pinned contract; `checkpoint` is
    # additive, and is null for the stub. It exists because the artifact was
    # once republished to the same path with a different model inside it, which
    # silently invalidated an in-flight evaluation.
    return {
        "status": "ok",
        "model": predictor.model_name,
        "stub": predictor.is_stub,
        "checkpoint": predictor.describe_checkpoint(),
    }


@app.post("/estimate")
async def estimate(
    response: Response,
    image: UploadFile = File(...),
    crop_margin: float | None = Query(
        None,
        ge=config.MIN_CROP_MARGIN,
        le=config.MAX_CROP_MARGIN,
        description=(
            "Override the face crop margin for this request only. Defaults to "
            "the server's CROP_MARGIN. Lets margins be A/B'd against real "
            "webcam photos without a restart."
        ),
    ),
    crop_margin_form: float | None = Form(
        None,
        alias="crop_margin",
        ge=config.MIN_CROP_MARGIN,
        le=config.MAX_CROP_MARGIN,
        description="Same as the crop_margin query parameter, as a form field.",
    ),
) -> dict:
    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type {content_type!r}. "
                f"Expected one of: {', '.join(sorted(config.ALLOWED_CONTENT_TYPES))}"
            ),
        )

    raw = await image.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Upload too large (limit {config.MAX_UPLOAD_BYTES} bytes)"
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Form field wins over the query parameter; neither means "use the default".
    margin = crop_margin_form if crop_margin_form is not None else crop_margin
    effective_margin = config.CROP_MARGIN if margin is None else margin
    # Echoed in a header rather than the body, so the pinned response shape is
    # untouched but an A/B run can always prove which margin produced it.
    response.headers["X-Crop-Margin"] = f"{effective_margin:.4f}"

    try:
        predictor = get_predictor()
        faces = predictor.predict_boxes(
            frame, predictor.detector.detect(frame), effective_margin
        )
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None

    return {"faces": [f.to_dict() for f in faces]}
