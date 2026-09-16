# server/ — age estimation API

FastAPI + uvicorn service that detects faces with OpenCV **YuNet** and returns an
age estimate with an uncertainty interval for each one.

It runs **today, without a trained model**: if no checkpoint is present it falls
back to a deterministic `StubPredictor` and says so loudly at startup. Swapping
in the real weights is a one-class change — nothing else in the stack moves.

## Quick start

```bash
python3 -m venv .venv                       # from the repo root
.venv/bin/pip install -r server/requirements.txt
.venv/bin/python -m uvicorn server.app:app --reload --port 8000
```

The YuNet ONNX weights (~230 KB) are downloaded into `server/models/` on first
run and are gitignored. To run the API and the web UI together, use
`./scripts/dev.sh` or `make dev` from the repo root.

## Endpoints

### `GET /health`

```json
{ "status": "ok", "model": "stub", "stub": true }
```

`model` is the backbone name once a real checkpoint is loaded
(e.g. `mobilenetv3_small_100`), or `"stub"` otherwise.

### `POST /estimate`

`multipart/form-data` with a single file field named **`image`**.

```bash
curl -F "image=@server/tests/assets/face.jpg;type=image/jpeg" \
     http://127.0.0.1:8000/estimate
```

```json
{
  "faces": [
    { "bbox": [148, 36, 97, 133], "age": 23.6, "low": 18.9, "high": 28.3, "confidence": 0.5602 }
  ]
}
```

* `bbox` is `[x, y, w, h]` in **integer pixel coordinates of the original
  uploaded image** — not of any internally resized copy.
* `age` is the point estimate, `low`/`high` are `age ∓ std` clamped to `[0, 100]`.
* `confidence` is a monotonically decreasing function of the prediction's
  standard deviation.

An image containing no detectable face returns `{"faces": []}` with **HTTP 200**,
not an error.

**Optional `crop_margin`** (query parameter or form field) overrides
`CROP_MARGIN` for that one request; see [Changing the margin](#changing-the-margin).
The margin used is echoed in the `X-Crop-Margin` response header.

| Situation                              | Status |
| -------------------------------------- | ------ |
| Success (including zero faces)          | 200    |
| Empty or undecodable image bytes        | 400    |
| Missing `image` field                   | 422    |
| `crop_margin` outside −0.5 … 2.0        | 422    |
| Non-image `Content-Type`                | 415    |
| Body larger than `MAX_UPLOAD_BYTES`     | 413    |
| Model/inference failure                 | 500    |

## The model contract

`checkpoints/age_model.pt` (produced by the `ml/` side of the project) is a
`torch.save` dict:

```python
{
  "state_dict": ...,   # timm mobilenetv3_small_100 with a 101-way classifier
  "meta": {
    "backbone": "mobilenetv3_small_100",
    "num_bins": 101,            # ages 0..100
    "input_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
    "test_mae": 5.43,
  },
}
```

It is a **DEX-style soft-expectation regressor**. The head is a distribution
over ages 0..100; the server decodes it as

```
age = Σ_i softmax(logits)_i · i
std = sqrt( Σ_i softmax(logits)_i · (i − age)² )
low, high = clamp(age ∓ std, 0, 100)
confidence = 1 / (1 + std / 6)
```

`TorchPredictor` also adopts `meta["mean"]`, `meta["std"]` and
`meta["input_size"]` from the checkpoint, so normalisation can never drift from
whatever training actually used.

## Cropping — keep in sync with training

`server/preprocessing.py` is the **single source of truth** for how a face is
cropped, and it is the easiest thing in the project to get silently wrong.

1. Take the YuNet bbox.
2. Expand it to a **square** with side `max(w, h) · (1 + 2·CROP_MARGIN)`.
3. Slide the square back inside the frame if it overhangs an edge (so a face
   near the border keeps its context rather than being lopsided), then clamp.
   If the image is too small to hold the square at all, the crop is
   edge-padded back to square so the aspect ratio is never distorted.
4. Resize to `INPUT_SIZE` (224).
5. Convert BGR→RGB, scale to `[0, 1]`, ImageNet-normalise, transpose to CHW.

UTKFace aligned+cropped images are essentially the raw detector box, so
`CROP_MARGIN` defaults to **0.0135**. That number is measured, not agreed:
running YuNet over 300 random UTKFace images (300/300 detected) and solving
`m = (200 / max(w_det, h_det) - 1) / 2` for UTKFace's own native framing gives a
median of 0.0135, IQR [0.0012, 0.0268], p05 −0.0149. It was 0.4 before it was
measured, which framed the face at ~31% of the crop area against ~94% in
training.

**Erring wide is the dangerous direction.** Training augments with
`RandomResizedCrop(scale=(0.8, 1.0))`, which only ever crops *in*, so the model
has seen framings tighter than UTKFace but never wider ones.

**If the training pipeline changes its crop geometry, this module must change
with it** — otherwise inference feeds the model out-of-distribution crops and
the ages drift with no visible error. `make eval` (below) is how you find out.

### Changing the margin

`CROP_MARGIN` is read at **startup**, not at import, so no code edit is needed:

```bash
CROP_MARGIN=0.05 make api
```

It can also be overridden **per request**, which is how to A/B margins against
real webcam photos without restarting anything:

```bash
curl -sS -D- -F image=@face.jpg 'http://127.0.0.1:8000/estimate?crop_margin=0.05'
```

Query parameter or multipart form field, both named `crop_margin` (the form
field wins). Valid range is −0.5 to 2.0 — negatives are allowed because
UTKFace's implied margins genuinely went negative at p05. The margin actually
used comes back in the `X-Crop-Margin` response header, so an A/B run can always
prove which margin produced which number. The response body is unchanged.

The web UI exposes the same control under **Crop margin (advanced)**, with a
"Re-analyse last frame" button that re-sends byte-identical pixels at a new
margin.

## Configuration

All via environment variables, re-read at startup.

| Variable                 | Default                      | Meaning |
| ------------------------ | ---------------------------- | ------- |
| `AGE_MODEL_PATH`         | `checkpoints/age_model.pt`   | Checkpoint to load; stub if absent |
| `CROP_MARGIN`            | `0.0135`                     | Face crop margin (fraction of box side) |
| `INPUT_SIZE`             | `224`                        | Model input resolution |
| `NUM_BINS`               | `101`                        | Age bins in the DEX head |
| `MAX_UPLOAD_BYTES`       | `10485760`                   | Request body cap (10 MiB) |
| `DETECT_SCORE_THRESHOLD` | `0.7`                        | YuNet score threshold |
| `DETECT_NMS_THRESHOLD`   | `0.3`                        | YuNet NMS threshold |
| `DETECT_TOP_K`           | `50`                         | Max detections considered |
| `CORS_ORIGINS`           | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins |
| `SERVER_MODELS_DIR`      | `server/models`              | Where YuNet weights are cached |

## Layout

```
server/
├── app.py            FastAPI app: routes, CORS, upload guards
├── config.py         env-driven settings
├── detector.py       YuNet wrapper + weight download, bbox rescaling
├── preprocessing.py  the shared crop/normalise pipeline (sync with training!)
├── predictor.py      AgePredictor / StubPredictor / TorchPredictor
├── tools/
│   └── eval_end_to_end.py   offline MAE of the full deployed path
└── tests/
```

## End-to-end evaluation

`ml/eval.py` measures the model on **training-framed** crops. This harness
measures the **deployed system**: YuNet detect → this server's crop → model. It
calls `AgePredictor.predict_boxes`, the same function `POST /estimate` calls, so
it is the real path and not a lookalike reimplementation.

**The gap between the two numbers is the crop mismatch, quantified.**

```bash
make eval                                   # ml/splits/test.csv, active margin
make eval EVAL_MARGINS="0 0.0135 0.05 0.1"  # sweep margins, pick the best
```

or directly:

```bash
.venv/bin/python -m server.tools.eval_end_to_end \
    --csv ml/splits/test.csv \
    --checkpoint checkpoints/age_model.pt \
    --margins 0 0.0135 0.05 0.1 \
    --save-crops /tmp/crops \
    --json /tmp/report.json
```

It reads `ml/` **read-only** — it just consumes the CSV. Column names are
sniffed (`path`/`filepath`/`image`… and `age`/`true_age`/`label`…), headerless
files work, and relative paths resolve against the CSV's directory, the repo
root, or `--image-root`. `--dir` instead reads ages from UTKFace-style
filenames.

Reported per margin: MAE, median AE, RMSE, bias (signed — tells you whether the
framing makes the model read old or young), ±5yr/±10yr hit rates, and how often
the true age fell inside the returned `low`–`high` interval. Detection failures
and multi-face images are counted separately rather than silently averaged in.

`--save-crops` writes a few example crops per margin so you can *look* at what
the model is being fed. That is usually faster at spotting a framing bug than
any metric.

Run with no checkpoint and it uses the stub, which is useful for checking the
harness itself — the report is loudly labelled as meaningless in that case.

**Known limitation:** UTKFace test images share the training framing, so this
cannot detect a systematic difference between YuNet-on-a-tight-200×200-crop and
YuNet-on-a-full-webcam-frame. Per-request `crop_margin` A/B against real webcam
photos is the only way to close that loop.

## Tests

```bash
.venv/bin/python -m pytest        # from the repo root
```

Everything runs against the stub, so no trained model is required. Coverage
includes the stub path, an image with no face, malformed/oversized/wrong-type
uploads, bbox coordinate-space invariants (including the detector's internal
downscale path for large images), crop geometry (including faces flush against
every frame edge and a face filling the whole frame), the per-request
`crop_margin` override, the soft-expectation decode, the offline eval harness,
and loading a contract-shaped checkpoint with `TorchPredictor`.

`server/tests/assets/face.jpg` is a public-domain US federal government
photograph; see `server/tests/assets/ATTRIBUTION.txt`.
