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
    "path": "/abs/path/checkpoints/age_model_realgt.pt",
    "sha256": "fb629f49987a",
    "bytes": 6618465,
    "recorded_test_mae": 6.393,
    "recorded_test_mae_decode": "median",
    "serving_decode": "median",
    "recorded_test_mae_corpus": "real_ground_truth (AgeDB 16487 + APPA-REAL 7591 + FG-NET 1002)",
    "label_semantics": "real chronological age",
    "trained_crop_margin": 0.0,
    "serving_crop_margin": 0.0,
    "crop_margin_matches_training": true,
    "real_age_mae": 6.34,
    "real_age_corpus": "AgeDB + APPA-REAL + FG-NET held-out test, n=3807",
    "in_corpus_mae_utkface": null,
    "accuracy_note": "..."
  }
}
```

`recorded_test_mae` is the figure **the checkpoint records about itself**;
`real_age_mae` is what we measured end to end through this server's own path.
They agree here (6.393 vs 6.34) because both are against real chronological
ages on the same split — the 0.05 gap is cv2-vs-PIL resizing and detector
framing.

> **6.393 is NOT an improvement on the previous model's 5.5472.** They measure
> different things. The old figure was agreement with UTKFace's DEX-estimated
> *apparent* ages; this one is error against *real chronological* age. On a
> like-for-like comparison — the same real-ground-truth held-out split — the
> previous model scores **9.127** and this one **6.393**. Reading the two
> recorded numbers side by side suggests a regression, and that is exactly
> backwards. This is why `/health` publishes `recorded_test_mae_corpus` and
> `label_semantics` next to the float, and why `in_corpus_mae_utkface` is
> `null` here rather than being filled in with something plausible: this model
> never saw UTKFace.

**Quote ~6.3 years to users.**

`sha256` is the first 12 hex chars of the digest of the file's bytes, so it is
directly comparable with `shasum -a 256 <path> | cut -c1-12`.

### The accuracy figures are pinned to the artifact they were measured on

`real_age_mae` and `in_corpus_mae_utkface` come from a table keyed by the
loaded checkpoint's `sha256`. Two artifacts are in it:

| sha256 | model | real-age MAE | in-corpus UTKFace MAE |
| --- | --- | ---: | ---: |
| `fb629f49987a` | real-GT (served) | **6.34** | `null` — never saw UTKFace |
| `56894c480044` | UTKFace/DEX (fallback) | 9.11 | 4.762 |

Both `real_age_mae` values are measured on the **same** held-out split through
the same harness, because the toggle shows them side by side and an unequal
comparison is worse than no comparison. The fallback also has an 8.52 recorded
against all of APPA-REAL (`real_age_mae_appa_all`); that is a valid number for
those weights but not comparable to 6.34, so it is not the user-facing one.

Any checkpoint not in the table reports `null` with an `accuracy_note` saying
so. Sibling artifacts exist (no-smoothing and CE variants), and pointing
`AGE_MODEL_PATH` at one to evaluate it must not make `/health` report another
model's accuracy for those weights. Accuracy is a property of a specific set of
weights, not of "the model", so an unrecognised artifact **fails closed** rather
than inheriting numbers it never earned. Re-run
`server/tools/eval_end_to_end.py` to measure a new one, then add it to the table.

`in_corpus_mae_utkface` is `null` for the real-GT model rather than being filled
with something plausible. It never trained on or was evaluated against UTKFace,
so carrying the other model's 4.762 across — or relabelling its real-age figure
— would manufacture a result that was never measured. Absent means absent.

### `serving_decode` follows the checkpoint

A checkpoint may declare `meta["decode"]`; if it does, we serve that decode and
log the override. Absent the key we default to `median`, which is correct for
the shipped artifact.

The right decode is a property of **how the model was trained**, not a fixed
choice. With `label_smoothing=0.1` a uniform pedestal (whose own expectation is
exactly 50) drags an expectation decode toward the middle, so median wins by
~0.2–0.4 years. Trained *without* smoothing the sign flips and expectation wins
by ~0.12. Hardcoding either one silently leaves accuracy on the table the next
time training changes — hence the key, and hence that we honour it.

### The serving crop is checked against the training crop

Checkpoints record the `crop_margin` they were trained with. On load we compare
it to the margin this server is serving and log a loud `CROP MARGIN MISMATCH`
warning if they differ, naming the value to set. `/health` reports
`trained_crop_margin`, `serving_crop_margin` and `crop_margin_matches_training`.

The crop is the largest preprocessing lever we have and a mismatch is **silent**
— no error, no visible symptom, just every face framed differently from training
and accuracy quietly degraded. This used to rest on a comment asking whoever
came next to keep the two in sync; now it is checked.

We **warn rather than override**. `CROP_MARGIN` is deliberately tunable at
runtime for A/B sweeps, so silently replacing an operator's explicit setting
with the checkpoint's would break the margin harness and ignore a deliberate
instruction. An undeclared margin reports `null`, not `true` — absence of a
declaration is not evidence of agreement, and claiming a match we never verified
would be a false assurance.

### Provenance travels with the number

`/health` passes several metadata fields straight through:

| field | why it is there |
| --- | --- |
| `recorded_test_mae_corpus` | An MAE is meaningless without knowing what it is an error *against*. 5.5472 against DEX-estimated apparent age and 6.393 against real chronological age are not comparable, **and the smaller one is the weaker result**. |
| `label_semantics` | Distinguishes real chronological age from DEX-estimated apparent age. |
| `recorded_test_mae_source` / `recorded_val_mae` | Training stamps the best *validation* MAE into `test_mae` and relies on a later eval pass to overwrite it. Until that happens the field is a selection-set score flattering itself — by up to ~0.3 years — under a name that reads as a held-out result. These fields let a provisional number announce itself. |
| `role` | Marks experiment intermediates, so an artifact that exists to demonstrate a point is never mistaken for the published model in a pasted `/health` payload. |

The general rule, learned the hard way on both sides of this project: a caveat
that lives in the producing code, or in a README, is invisible at the point of
use. It has to be carried by the artifact and surfaced by the consumer.

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

`checkpoints/age_model_realgt.pt` (produced by the `ml/` side of the project)
is a `torch.save` dict:

```python
{
  "state_dict": ...,   # timm mobilenetv3_small_100 with a 101-way classifier
  "meta": {
    "backbone": "mobilenetv3_small_100",
    "num_bins": 101,            # ages 0..100
    "input_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
    "decode": "median",         # honoured, not assumed — see below
    "crop_margin": 0.0,         # checked against ours at load
    "test_mae": 6.393,
    "corpus": "real_ground_truth (AgeDB + APPA-REAL + FG-NET)",
    "label_semantics": "real chronological age",
  },
}
```

### Two models, switchable per request

Both checkpoints are loaded at startup and stay resident. `POST /estimate`
takes an optional `model` field (query or form) selecting between them, and
echoes the one used in the `X-Model` response header:

| key | predicts | artifact |
| --- | --- | --- |
| `real` (default) | how old the person **is** | `age_model_realgt.pt` |
| `apparent` | how old the person **looks** | `age_model.pt` |

Point `AGE_MODEL_DIR` at the directory holding both, e.g.

```bash
AGE_MODEL_DIR=/path/to/checkpoints make dev
```

`AGE_DEFAULT_MODEL` changes which one answers an unqualified request.
`GET /health` gains a `models` block listing both with their digests.

Their recorded MAEs (6.393 and 5.5472) are **not** comparable and neither model
is "the accurate one" -- they are errors against different targets. A slot that
is pointed at the *other* model's weights refuses to load rather than serving
them under the wrong label, because identity is checked by content digest.

### Which checkpoint gets loaded

`AGE_MODEL_PATH`, if set, wins outright and is never second-guessed. With it
unset the server tries, in order:

1. `checkpoints/age_model_realgt.pt` — the served model
2. `checkpoints/age_model.pt` — the older UTKFace/DEX model, kept as a fallback
3. the stub, with a loud startup banner

The fallback is ordering only; it never silently substitutes accuracy figures,
because those are keyed by digest (above).

The head is a distribution over ages 0..100. The server decodes it by
**quantile**, not by expectation:

```
age        = min{ i : cumsum(softmax(logits))_i >= 0.50 }   # median
low, high  = the 0.16 and 0.84 quantiles, read the same way
confidence = 1 / (1 + (high − low) / 12)
```

### Why the median and not the mean

The served decode is read from `meta["decode"]`, not assumed. It is `median`
for both artifacts on disk, but for **different reasons**, and that distinction
is the point of the key existing.

*The old UTKFace model* was trained with `label_smoothing=0.1`, which trains a
uniform pedestal across all 101 bins. That pedestal's own expectation is exactly
50, so decoding by expectation returns roughly `0.9 · age + 5` — dragging every
estimate toward the middle. The median ignores the pedestal. In-corpus, median
beat expectation 4.762 vs 5.477 MAE.

The ML side later demonstrated that mechanism causally: with the pedestal
removed the sign **flips** and expectation wins (−0.122), and expectation-decode
bias collapses monotonically as smoothing goes away
(+3.835 → +1.400 → −0.379 → +0.002). `test_median_decode_ignores_a_label_smoothing_pedestal`
in `tests/test_preprocessing.py` pins that mechanism and should stay.

*The served real-GT model* has **no label smoothing** — it uses a DLDL Gaussian
soft target (σ = 2.5). So the pedestal argument does not apply to it. Under
DLDL the two decodes tie on MAE (6.390 expectation vs 6.393 median — noise), and
median is kept on a different basis: **+4.1pp CS@5** (56.2% vs 52.1%),
reproduced on validation.

The durable warning: if a future artifact drops smoothing and nobody re-examines
the decode, median silently leaves accuracy on the table. That is why we read
`meta["decode"]` rather than hardcoding. A checkpoint that declares a different
decode gets it, with a loud log line if it disagrees with our default.

The interval is built from true CDF quantiles rather than `age ± σ`. σ is
computed about the *mean*, so pairing it with a median point estimate would be
subtly inconsistent.

**Interval coverage is 60.8% against a 68% nominal target** on the served model
— the interval is slightly too narrow.

An earlier revision of this file called that a *regression*, comparing it to 75%
on the previous model. That comparison was invalid: the 75% was measured on the
UTKFace split and the 60.8% on the real-ground-truth split. Two variables moved
at once. Running the old checkpoint through this same harness on the *same*
real-GT split (n=3,807) isolates them:

| model | MAE | coverage | mean width | sd of width | r(width, \|error\|) |
|---|---:|---:|---:|---:|---:|
| shipped (UTKFace/DEX) | 9.110 | 61.7% | 20.61 | 12.92 | +0.309 |
| realgt (DLDL σ=2.5) | 6.343 | **60.8%** | **11.62** | 4.22 | **+0.364** |

**The coverage difference between the two models is 0.9pp.** The undercoverage
is a property of this construction on this corpus, present in both models
equally — not something the new model introduced. The drop from 75% was the
corpus, not the weights.

What the new model *did* change is width: the same coverage is achieved with an
interval **44% narrower**, and width remains slightly more correlated with
absolute error than before. Narrower at equal coverage is an improvement, not a
regression.

It is deliberately **not** retuned. Widening the quantiles until coverage hit
68% on the test split would be fitting to the evaluation set. It is documented
instead, and the interval is advanced-panel-only.

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
| `AGE_MODEL_PATH`         | first existing of `age_model_realgt.pt`, `age_model.pt` | Checkpoint to load; stub if absent |
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
    --checkpoint checkpoints/age_model_realgt.pt \
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
