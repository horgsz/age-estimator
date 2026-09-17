# Parity harness

Two implementations of the same inference pipeline live in this repository:

| | Python (`server/`) | TypeScript (`web/src/browser/`) |
|---|---|---|
| detection | `cv2.FaceDetectorYN` | the same YuNet ONNX, via onnxruntime-web |
| crop | `server/preprocessing.py` | `preprocess.ts` |
| resize | `cv2.resize` | `cv-resize.ts` |
| model | `checkpoints/*.pt` via PyTorch | `checkpoints/*.onnx` via onnxruntime-web |
| decode | `server/predictor.py` | `decode.ts` |

The second exists only because GitHub Pages cannot run Python. It is a
liability, not a feature: **a silent mismatch between the two degrades every
prediction while leaving every test green.**

That is not hypothetical. This project has already shipped a preprocessing
mismatch once — a crop margin of 0.4 against a training distribution of 0.0,
which framed the face at ~31% of the crop area against ~94% in training and cost
3.0 years of MAE. The unit tests passed. The offline evaluation reported 5.5.
Nothing was visibly wrong, because nothing in the system compared the two things
that had diverged.

This harness is that comparison.

## What it does

Runs a fixed set of images through both paths and diffs every intermediate:

1. **Detected bounding boxes.** Both sides run the same YuNet weights, so the
   expected difference is exactly 0. Anything else means the post-processing
   port is wrong — not that "detectors differ".
2. **The 224×224×3 float32 tensor**, element by element. This is the layer that
   catches crop geometry, channel order, normalisation and resampling. Reported
   as max and mean absolute difference and as a count of differing elements.
3. **Raw logits.** Separates ONNX-vs-PyTorch numerics from everything upstream.
4. **The predicted age**, the range and the confidence — what a user reads.

Face-count mismatches, tensor-shape mismatches and missing cases are failures in
their own right.

### The strict case

`strict-no-resize` pins a bounding box whose square crop side is exactly 224, so
`resize_crop` resizes 224 → 224. OpenCV's `INTER_LINEAR` at scale 1 is an exact
identity, which means that case contains **no resampling at all**.

It exists because resampling is the one difference that could be argued away. If
that case disagrees, the cause is cropping, channel order or normalisation, and
there is no resize to blame it on.

### Why the fixtures are PNG

The browser and OpenCV use different JPEG decoders, and they disagree by a
least-significant bit or two on some coefficients. Comparing lossy-decoded
pixels would put a floor under the achievable tensor agreement and make a real
preprocessing bug indistinguishable from a decoder difference. PNG is lossless,
so both decoders must produce identical bytes and any remaining tensor
difference is ours.

The fixtures are all derived from `server/tests/assets/face.jpg` — an official
White House portrait, a US federal government work, public domain. Nothing from
`datasets/` is used or redistributed. `make_fixtures.py` regenerates them.

### Why it drives a real browser

`run_browser.mjs` serves the **built** `web/dist` over plain HTTP and drives
`parity.html` in headless Chromium. The static file server deliberately does not
send `Cross-Origin-Opener-Policy` or `Cross-Origin-Embedder-Policy`, because
GitHub Pages cannot send them either — so `SharedArrayBuffer` is unavailable in
the harness exactly as it is in production, and the harness cannot accidentally
measure a threaded WASM path the deployment can never take. It asserts this and
records it in the output.

`parity.html` ships with the site rather than being a test-only entry point, so
the comparison can also be pointed at the live deployment with `--url`.

## Running it

```sh
make parity
```

or, step by step:

```sh
# 1. Fixtures (only needed once; they are committed)
.venv/bin/python parity/make_fixtures.py

# 2. The reference side
.venv/bin/python parity/run_python.py \
    --cases parity/fixtures/cases.json --out parity/python.json

# 3. The build under test
cd web && npm run build:static && cd ..

# 4. The browser side
cd parity && npm install && npx playwright install chromium && cd ..
node parity/run_browser.mjs --dist web/dist --base / \
    --cases parity/fixtures/cases.json --out parity/browser.json

# 5. The comparison
.venv/bin/python parity/compare.py \
    --python parity/python.json --browser parity/browser.json --strict
```

Against the deployed site instead of a local build:

```sh
node parity/run_browser.mjs --url https://horgsz.github.io/age-estimator/parity.html \
    --cases parity/fixtures/cases.json --out parity/browser.json
```

## Tolerances

| layer | limit | expected |
|---|---|---|
| bounding box | 0 px | 0 |
| tensor, max abs | 1e-3 | 0 |
| logits, max abs | 1e-2 | ~1e-5 |
| age | 0.1 years | 0 |

The tensor limit is a guard, not a rubber stamp. After normalisation one
least-significant bit of an 8-bit pixel is `1/255/0.225 ≈ 1.7e-2`, so 1e-3 is
comfortably under a single-bit disagreement: it cannot be satisfied by a
pipeline that is merely close.

**If a threshold is breached, find out why. Do not raise it.** Raising a
tolerance to get a green run reproduces the original failure exactly: a
measurement that reports agreement it has not established.

Likely causes, in order:

1. **The resize.** `cv-resize.ts` is a port of OpenCV's kernels including its
   fixed-point arithmetic and round-half-to-even. Check
   `resizeAreaFast`'s 2×2 integer path and the fused-multiply-add emulation in
   `resizeAreaGeneric` first; both were found by differential testing and
   neither is obvious from reading the algorithm.
2. **Colour space.** The crop is carried as BGR end to end, matching OpenCV.
   Canvas pixel data is RGBA. YuNet's input is BGR 0..255 *unnormalised* — the
   age model's is RGB and ImageNet-normalised. Mixing those up produces
   plausible output.
3. **Detector box conventions.** If boxes differ, check the NMS port: boxes are
   truncated to `int` before IoU, the sort is stable and descending, `top_k`
   applies after sorting, and NMS is skipped entirely when only one candidate
   cleared the score threshold.
4. **Rounding.** Python's `round()` and OpenCV's `cvRound` are both
   round-half-to-even; JavaScript's `Math.round` is not. `pyRound` and `cvRound`
   exist for this and are easy to forget at a new call site.
5. **The crop margin.** `models.json` carries `preprocessing.crop_margin` from
   `server/config.py`. If the two paths disagree only on framing, check that
   first.

## Results

Measured numbers, with the commit they were measured at, are in
[`RESULTS.md`](RESULTS.md).
