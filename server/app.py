"""FastAPI app exposing the age-estimation API.

Endpoints
---------
``GET  /health``    -> ``{"status": "ok", "model": "<backbone or 'stub'>", "stub": <bool>}``
``POST /estimate``  -> ``{"faces": [{"bbox": [x, y, w, h], "age", "low", "high", "confidence"}]}``

``bbox`` is in the ORIGINAL uploaded image's pixel coordinate space. An image
with no detectable face returns ``{"faces": []}`` with HTTP 200.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
    predictor = get_predictor()
    log.info(
        "Age estimator ready (model=%s, stub=%s, crop_margin=%.2f, input=%d)",
        predictor.model_name,
        predictor.is_stub,
        config.CROP_MARGIN,
        config.INPUT_SIZE,
    )
    yield


app = FastAPI(title="age-estimator", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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
    return {"status": "ok", "model": predictor.model_name, "stub": predictor.is_stub}


@app.post("/estimate")
async def estimate(image: UploadFile = File(...)) -> dict:
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

    try:
        faces = get_predictor().predict(frame)
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None

    return {"faces": [f.to_dict() for f in faces]}
