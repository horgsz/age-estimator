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
{ "status": "ok", "model": "stub", "stub": true, "checkpoint": null }
```

`model` is the backbone name once a real checkpoint is loaded
(e.g. `mobilenetv3_small_100`), or `"stub"` otherwise.

`checkpoint` identifies **which artifact is actually live**. It is `null` for the
stub; with real weights it reports the path, a short content hash, the file size
and the MAE the trainer claimed:

```json
{
  "status": "ok",
  "model": "mobilenetv3_small_100",
  "stub": false,
  "checkpoint": {
    "path": "/abs/path/checkpoints/age_model.pt",
    "sha256": "56894c480044",
    "bytes": 6606539,
    "recorded_test_mae": 5.5472,
    "recorded_test_mae_decode": "expectation",
    "serving_decode": "median",
    "in_corpus_mae_utkface": 4.762,
    "real_age_mae_appa_real": 8.52,
    "accuracy_note": "..."
  }
}
```

`recorded_test_mae` is the figure **the checkpoint records about itself**, not
the accuracy of what this server returns. It was measured with the expectation
decode; we serve the median decode, which measures 4.762 end to end. The field
is named that way so the two can never be confused — do not relabel it
`test_mae`.

> **Neither 5.5472 nor 4.762 is real-world accuracy.** Both are measured against
> UTKFace labels, and those labels are themselves DEX-algorithm estimates — so
> they measure *agreement with a labelling method*. Against real chronological
> ages (APPA-REAL, 7,534 images) the MAE is **8.52**. `/health` serves both so
> the distinction travels with the number instead of living only here. Anything
> user-facing must quote 8.52.

`sha256` is the first 12 hex chars of the digest of the file's bytes, so it is
directly comparable with `shasum -a 256 <path> | cut -c1-12`.

This exists because the checkpoint was twice republished to the same path
mid-evaluation, silently invalidating measurements taken against it. `test_mae`
alone is not sufficient to tell two artifacts apart — the second republish kept
it byte-identical while changing the file — so compare the hash.

`status`, `model` and `stub` are the pinned contract; `checkpoint` is additive
and clients may ignore it.

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

The head is a distribution over ages 0..100. The server decodes it by
**quantile**, not by expectation:

```
age        = min{ i : cumsum(softmax(logits))_i >= 0.50 }   # median
low, high  = the 0.16 and 0.84 quantiles, read the same way
confidence = 1 / (1 + (high − low) / 12)
```

### Why the median and not the mean

The checkpoint was trained with `label_smoothing=0.1`, which trains a uniform
pedestal across all 101 bins. That pedestal's own expectation is exactly 50, so
decoding by expectation returns roughly `0.9 · age + 5` — it drags every
estimate toward the middle of the range. The median ignores the pedestal.

Measured end to end over the 1,184-image UTKFace test set at `CROP_MARGIN = 0.0`
— these are **in-corpus** figures, see the accuracy section below:

| decode | MAE | CS@5 | bias | 85-year-old displays as |
|---|---|---|---|---|
| soft expectation | 5.477 | 55.5% | +2.11 | ~68 |
| **median (shipped)** | **4.762** | **69.0%** | **+0.17** | **~78** |

The decode change itself was validated in-corpus only; there is no
expectation-decode run against real ages, so the 0.7-year win is measured
against DEX-derived labels and its true magnitude is unconfirmed.

The interval is built from true CDF quantiles rather than `age ± σ`. σ is
computed about the *mean*, so pairing it with a median point estimate would be
subtly inconsistent, and σ is itself inflated by the same pedestal — it averaged
12.7 years, which is why confidence used to read low on every face. The quantile
interval averages 11.8 years wide and achieves 75% empirical coverage against a
68% nominal target.

`Decoded` retains the expectation and σ alongside the shipped median so the two
decodes stay comparable on the same weights without another inference pass.

`TorchPredictor` also adopts `meta["mean"]`, `meta["std"]` and
`meta["input_size"]` from the checkpoint, so normalisation can never drift from
whatever training actually used.

### Wrapper-prefixed state dicts

The contract says `state_dict` belongs to a *bare* timm model. The checkpoint
actually delivered prefixes every key with `backbone.`, because training wrapped
the model (`self.backbone = timm.create_model(...)`). The tensors are identical,
so the loader tolerates this: `_strip_wrapper_prefix` tries `backbone.`,
`model.`, `module.` and `net.`, and strips one **only if** the raw keys do not
already match and the stripped keys then cover every key the model expects.
Anything looser could silently load unrelated tensors of a compatible shape,
which is worse than refusing.

A strip is logged as a loud multi-line `WARNING` banner reading
`CHECKPOINT CONTRACT DEVIATION`, so that drift is visible in the startup log
rather than silently absorbed.

**Status:** the prefix was fixed at source — the current artifact ships bare
timm keys and the banner no longer fires. The tolerance is retained anyway: it
costs nothing, it is the correct robustness, and it failed *safe* (falling back
to the stub) rather than crashing when it was needed. Do not remove it because
"the checkpoint is fine now" — the artifact has already been republished to the
same path three times.

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
`CROP_MARGIN` defaults to **0.0**. That number is measured end to end, not
agreed. Two independent sweeps over the 1,185-image test split — `ml/`'s, which
re-frames ground-truth crops, and this server's, which runs real YuNet detection
and then the crop below — both bottom out at 0.0:

| margin | `ml/` (re-framed) | server (detect → crop → model) |
| -----: | ----------------: | -----------------------------: |
| −0.050 | 5.548 | 5.590 |
| −0.025 | 5.501 | 5.574 |
| **0.000** | **5.495** | **5.477** |
| 0.0135 | 5.547 | 5.551 |
| 0.050 | 5.706 | 5.688 |
| 0.100 | 5.949 | 5.880 |
| 0.200 | 6.264 | 6.190 |
| 0.400 | — | 8.442 |

The curves agree within **0.07 years everywhere**, which is the load-bearing
result: running a real detector instead of re-framing ground truth does not
shift the framing, so an offline-measured constant transfers to the deployed
path. An earlier value of 0.0135 came from measuring UTKFace's own framing
geometrically (median of `m = (200 / max(w_det, h_det) - 1) / 2` over 300
images); it costs 0.07 years, which is within noise but not an improvement.

It was **0.4** before any of this was measured, which framed the face at ~31% of
the crop area against ~94% in training and cost ~3 years of MAE while the
offline eval still reported ~5.5 — a silent 1.5x degradation.

**Erring wide is still the dangerous direction.** The curve is markedly
asymmetric: tighter-than-training costs almost nothing (−0.05 is +0.11 years),
wider degrades steeply. Anything in [−0.05, +0.05] is within ~0.11 years, which
comfortably absorbs detector jitter.

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
| `CROP_MARGIN`            | `0.0`                        | Face crop margin (fraction of box side) |
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
