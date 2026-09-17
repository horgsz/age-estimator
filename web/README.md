# web/ — camera + upload UI

Vite + vanilla TypeScript. No framework.

Take a photo with your webcam (or drop in an image); the app draws each detected
face with an **age range**, a point estimate, and a confidence indicator.

## Two builds, one UI

The same interface runs against two interchangeable inference engines. Which one
is compiled in is a build-time choice (`VITE_ENGINE`), overridable per visit with
`?engine=server` / `?engine=browser`.

| | `server` (default) | `browser` (`--mode static`) |
| --- | --- | --- |
| where inference runs | the [`server/`](../server/README.md) FastAPI app | this tab |
| how the frame gets there | JPEG POSTed to `/estimate` | never leaves the device |
| face detection | `cv2.FaceDetectorYN` | the same YuNet ONNX via onnxruntime-web |
| age model | `checkpoints/*.pt` via PyTorch | `checkpoints/*.onnx` via onnxruntime-web |
| model metadata | `GET /health` | `models/models.json`, baked at build time |
| deployed at | localhost | <https://horgsz.github.io/age-estimator/> |

The browser build exists because GitHub Pages serves static files and cannot run
Python. It is a **second implementation of logic that already exists**, which is
a liability rather than a feature: a silent mismatch degrades every prediction
while leaving every test green. [`parity/`](../parity/README.md) is what stops
that — it runs the same images through both paths and requires the 224×224
tensors to be identical. Read it before changing anything under `src/browser/`.

## Run

```bash
cd web
npm install
npm run dev          # http://localhost:5173, talks to the API
```

The API must be running on `http://127.0.0.1:8000` — from the repo root,
`make api`, or `make dev` / `./scripts/dev.sh` to start both at once.

For the fully client-side build:

```bash
npm run dev:static     # http://localhost:5173, no API needed
npm run build:static   # bundle to dist/
```

| Script                 | Does |
| ---------------------- | ---- |
| `npm run dev`          | Dev server with HMR on port 5173 (server-backed) |
| `npm run dev:static`   | Same, but running inference in the browser |
| `npm run stage`        | Copy models + the ONNX runtime into `public/` |
| `npm run build`        | Type-check (`tsc --noEmit`) then bundle to `dist/` |
| `npm run build:static` | Stage, type-check, and bundle the client-side build |
| `npm run preview`      | Serve the production bundle |
| `npm run typecheck`    | Type-check only |

### Configuration

| Variable        | Default                 | Meaning |
| --------------- | ----------------------- | ------- |
| `VITE_API_BASE` | `http://127.0.0.1:8000` | API origin (server engine only) |
| `VITE_ENGINE`   | `server`                | `server` or `browser`; set to `browser` by `.env.static` |
| `VITE_BASE`     | `/`                     | Deployed path prefix, read by Vite's `base` |

```bash
VITE_API_BASE=http://127.0.0.1:9000 npm run dev
```

Whatever origin the UI is served from must be in the server's `CORS_ORIGINS`.

`VITE_BASE` matters only for the deployment: GitHub Pages serves this as a
*project* site under `/age-estimator/`, so the default `/` would emit asset URLs
that 404 there. The browser engine reads the same value back through
`import.meta.env.BASE_URL` to locate `models/` and `ort/`.

## What it does

* **Live preview** via `getUserMedia`, with a device picker that appears when
  more than one camera is present, and a `devicechange` listener so hot-plugged
  cameras show up.
* **Capture & estimate** draws the current frame to a canvas, exports JPEG at
  quality 0.9, and POSTs it as multipart `image`.
* **Upload** the same way, via the file picker or drag-and-drop, so the app is
  fully testable without a camera. Large images are downscaled to a 1600 px
  longest side before upload.
* **Results** are drawn as boxes over the analysed frame. The headline is the
  point estimate, rounded to a whole year and labelled **Age Estimate** — a
  decimal would imply a precision the model does not have. Confidence drives the
  box's colour (red → green), its opacity, and a small bar.
* **Crop margin (advanced)** — a collapsible panel that overrides the server's
  `CROP_MARGIN` for a single request, plus a **Re-analyse last frame** button
  that re-sends *byte-identical* pixels at the new margin. A/B'ing margins is
  only meaningful if the input does not change between runs, so re-capturing
  from the webcam would defeat the point. The margin the server actually used is
  read from the `X-Crop-Margin` response header and shown below the control.

  Opening the panel also reveals a per-face debug line with the raw `low–high`
  range, the unrounded estimate, and the bbox. The API still returns all of it
  on every request; the panel just decides whether the user sees it.

  The default is **0.0**, the joint minimum of two independent end-to-end
  sweeps. This is the single highest-risk number in the system: too wide and the
  model sees framings it never saw during training. See `server/README.md`.

* **Tail caveats.** The model's error is not uniform across ages, so the UI says
  so where it matters. See below.

### Switching models

A radio group above the capture button chooses what to estimate: **how old this
person actually is** (default) or **how old this person looks**. Changing it
re-analyses the retained frame rather than re-capturing, so the two answers come
from byte-identical pixels.

**Every derived figure is keyed to the active model**, not to the app. The
typical-error number, the age-gating percentage and the caveat band are all
served per model by `/health`, keyed by the checkpoint's own content digest,
and swapped when the toggle changes:

| | real (default) | apparent |
| --- | ---: | ---: |
| typical error vs real age | ~6.3 yr | ~9.1 yr |
| under-18s shown as 18+ | 29.6% | 40.3% |
| band caveat | none | shown 40+ reads ~6 yr high |

Both error figures are measured on the **same** held-out split through the same
server path, so the comparison the toggle invites is a fair one. An earlier
revision quoted 8.5 for the apparent model from a different corpus — a real
measurement, but not one you can subtract from 6.3.

Neither is "the accurate one": they optimise different targets, and the apparent
model is being scored here against something it was never trained to predict.

The caveat is the sharp case: **it is true of the apparent-age model and false
of the real-age one**, so a single hardcoded band would be actively wrong for
one of them. `index.html` therefore hardcodes no figure at all — the header
starts unquantified and is filled in once the active model is known.

An unmeasured checkpoint supplies none of these, and the copy falls back to
unquantified wording rather than borrowing another model's numbers. The
age-verification warning is never suppressed, only de-quantified: a model with
no measured figure is not thereby safer, just unmeasured.

### Not usable for age verification

**About 30% of people under 18 are displayed as 18 or over.** Measured end to
end through this app's own path (YuNet detect → our crop → model → median
decode) on the real-ground-truth held-out test split (AgeDB + APPA-REAL +
FG-NET, n = 3,807, real chronological ages):

| threshold | true minors shown as adult | shown-adult who are minors |
| ---: | ---: | ---: |
| 13 | 19.2% | 1.7% |
| 16 | 26.0% | 2.9% |
| **18** | **29.6%** | **4.1%** |
| 21 | 33.9% | 6.5% |

The two columns answer different questions and **diverge by roughly 7×** at the
18 threshold. The right-hand column is the one you can observe without knowing
true ages — it makes the gate look 96% correct. The left-hand column is the one
that matters if a gate is protecting a minor. The difference is pure base rate:
adults outnumber minors, so most predictions above the line really are adults,
while the minors who slip through are a large fraction of a small group.

This improved with the real-ground-truth model (was 40.3%) and is still far too
high to gate on. It is the same true-versus-predicted conditioning trap that
runs through this project, in its highest-consequence form — and note that here
the *observable* number is the reassuring one, which is the configuration least
likely to prompt anyone to look harder. The UI states the limitation in the
page header rather than burying it here.

### Accuracy — the number that matters

**Against real chronological ages the MAE is ~6.3 years**, measured end to end
through this app's path on the held-out real-GT test split (n = 3,807).

The model is trained on real chronological ages (AgeDB + APPA-REAL + FG-NET).
It has no UTKFace in-corpus figure, and none should be invented for it.

> **Do not compare 6.393 against the previous model's 5.5472.** Those measure
> different things — agreement with DEX-estimated *apparent* age versus error
> against *real* age. Like-for-like on the same real-GT split, the previous
> model scores **9.127** and this one **6.393**. The bare floats suggest a
> regression; the truth is a 2.7-year improvement.

Residual compression is real and not a labelling artefact: the fitted slope is
**0.808**, up from 0.744, but still well short of 1.0 after removing DEX labels
from training entirely. Extremes are still pulled toward the middle.

### Where the estimate is least reliable

**No band caveat currently ships.** `tailCaveat()` returns `null`. The plumbing
is kept on purpose — see the end of this section.

Thresholds must be conditioned on the **predicted** age, because that is the
only thing the UI knows. Binning by *true* age answers a question the UI cannot
ask. This has inverted a conclusion three times in this project, so check it
first before changing anything here.

Re-measured end to end on the real-GT held-out test split (n = 3,807):

| shown | n | MAE | bias | CS@5 |
| ----: | -: | --: | ---: | ---: |
| 0–10 | 225 | 2.15 | −0.34 | 95.1% |
| 10–18 | 159 | 4.72 | −1.90 | 75.5% |
| 18–25 | 353 | 5.66 | −1.72 | 61.2% |
| 25–40 | 1527 | 6.20 | −0.15 | 55.7% |
| 40–50 | 595 | 7.30 | +1.14 | 51.4% |
| 50–60 | 448 | 7.70 | −0.16 | 47.5% |
| 60–70 | 280 | 7.33 | −0.51 | 48.6% |
| 70+ | 220 | 7.35 | −0.23 | 46.8% |

**Why the old "shown 40+ reads high" caveat was removed.** It was correct and
well corroborated *for the previous model*. Bias is now near zero in every band.
The old model split +5.70 above 40 against −0.55 below — 6.25 years. The largest
equivalent split now available at **any** threshold is 1.52 years (at 30), and
just 0.80 at 40. There is no direction left to warn about, and carrying the old
warning across would fire it on predictions that are approximately unbiased.

**Why no precision caveat replaced it.** Precision does still degrade with
displayed age (MAE 5.60 below 40 vs 7.43 above, ratio 1.33). That was the
obvious replacement and it was rejected on measurement: the **confidence bar
already carries this signal, per face, and better than a band could**.

| confidence quartile | Q1 | Q2 | Q3 | Q4 |
| --- | ---: | ---: | ---: | ---: |
| MAE | 8.86 | 6.46 | 5.84 | 4.21 |

Pearson *r*(confidence, |error|) = **−0.343**, and mean confidence already
tracks the displayed-age drop (0.596 → 0.485 as MAE goes 4.37 → 7.47). A band
caveat would restate a continuous per-face signal as a step function, and would
be wrong for the many high-confidence faces above 40.

**Why the plumbing is kept.** The correct caveat has changed three times, twice
reversing direction, and each time the trap was re-deriving from true-age bins.
Keeping the derivation and the negative result means the next person starts from
the evidence. The standard to clear is explicit: a caveat ships only if it is
supportable on **displayed** age, on the **weights actually served**, and is not
already better conveyed by the confidence bar.

The estimate is **not corrected**. Subtracting a bias would bury a known,
measured limitation inside a number that looks authoritative.

#### Interval calibration — known and slightly optimistic

The `low`–`high` interval is the 16th/84th percentile of the predicted
distribution, so it is **nominally 68%**. Measured coverage is **60.8%** — the
interval is slightly too narrow.

This was previously written up as a *reversal* from 75% on the old model. That
was wrong, and the error was ours: the 75% came from the UTKFace split and the
60.8% from the real-ground-truth split, so the comparison changed corpus and
weights together. Re-running the old checkpoint on the same real-GT split
(n=3,807, same harness, same crop) gives **61.7%** — a 0.9pp difference from the
current model.

So the undercoverage is not new and not caused by the new training objective. It
is how this quantile construction behaves on this corpus, for both models. The
nominal "68%" label is the part that does not hold; it is off by roughly 7pp
regardless of which checkpoint is selected, which makes it the one derived
number here that is *not* per-model.

What did change is sharpness: the new model reaches the same coverage with an
interval 44% narrower (mean width 20.6 → 11.6 years), and width stays slightly
more predictive of absolute error (r +0.309 → +0.364). Equal coverage at half
the width is a better interval, not a worse one.

It has deliberately **not** been retuned. Widening the quantiles until coverage
hit 68% on the test split would be fitting to the evaluation set — the exact
self-flattering move this project has criticised elsewhere. It is documented
instead. The interval is advanced-panel-only, so the user-facing impact is
limited to the confidence bar, which remains monotonic in interval width and
demonstrably informative (see the quartile table above).

### UI states

| State | What you see |
| ----- | ------------ |
| Camera off / permission denied | Overlay on the preview explaining why, with a retry button; upload still works |
| Insecure origin | Explains that `getUserMedia` needs `localhost` or https |
| Analysing | Busy status, controls disabled |
| 1+ faces | Boxes + per-face cards, each with a rounded "Age Estimate" |
| 0 faces | `{"faces": []}` is a success: a hint about lighting/distance, not an error |
| Server error / unreachable | The server's `detail` message, or a "start the API" hint |
| Invalid crop margin | The server rejects it with 422 and the message is surfaced |
| Stub model | A banner from `GET /health` warning that the ages are fake |
| Downloading a model (browser build) | A progress bar with real byte counts |

## The browser build

### Why the model download has a progress bar

Each age model is 6.2 MB. An indeterminate spinner on a multi-megabyte fetch is
indistinguishable from a hang, and the user has no way to judge whether waiting
is worthwhile. `Content-Length` makes a real percentage available for free, so
it is shown. When the header is absent the bar goes indeterminate and shows
bytes-so-far rather than inventing a percentage.

### Why only one model is fetched

The models are loaded **on first use, per model**, not at boot. Most visitors
never touch the toggle, so fetching both up front would cost everyone 12.4 MB to
use 6.2 MB of it. The detector (230 KB) comes down alongside whichever model is
requested first. The browser's HTTP cache makes every subsequent visit free,
which is why the assets are served under stable, content-independent names — a
build that fingerprinted them per deploy would throw that away.

### Why the detector is YuNet and not MediaPipe

MediaPipe Face Detector is the easier option — a packaged WASM task with its own
model and no post-processing to write. It was rejected because the crop geometry
is calibrated to *YuNet's box convention*.

`CROP_MARGIN` was chosen by sweeping end-to-end MAE, and the curve either side
of the optimum is strongly asymmetric: −0.05 costs 0.11 years, +0.2 costs 0.71,
+0.4 costs 2.97. A detector whose boxes run systematically wider than YuNet's
does not announce itself; it shifts the *effective* margin into the expensive
side of that curve, invisibly, with no test to fail.

Running the same ONNX weights OpenCV runs does not answer that question, it
removes it: the boxes are not close to YuNet's, they are YuNet's. Measured
across the parity fixtures, every box matches the Python path exactly. The price
is that `src/browser/yunet.ts` has to reimplement the post-processing
`cv2.FaceDetectorYN` does in C++.

### Why the resize is hand-written

`ctx.drawImage(src, 0, 0, 224, 224)` uses an unspecified downscaling filter that
differs between browser engines and between GPU and software paths.
`src/browser/cv-resize.ts` is instead a port of OpenCV's own kernels, including
its fixed-point bilinear arithmetic and round-half-to-even, so the tensor is not
merely close to the server's — it is identical. See the file header and
[`parity/README.md`](../parity/README.md).

### Single-threaded WASM

GitHub Pages cannot send `Cross-Origin-Opener-Policy` or
`Cross-Origin-Embedder-Policy`, so `SharedArrayBuffer` is unavailable and
multi-threaded WASM is impossible. `ort.env.wasm.numThreads` is set to `1`
explicitly rather than left to autodetection: a failed probe surfaces as a
worker that never initialises, which looks like a hang rather than an error.
The parity harness asserts `crossOriginIsolated === false` so the measurement
can never come from a threaded path the deployment cannot take.

One 224×224 inference takes ~6 ms and detection 2–20 ms depending on frame size,
so nothing is lost.

### Images never leave the device

The browser build states this in the header, and it is the one genuine advantage
it has over the server build. It is asserted by the engine rather than by
`index.html`, so the server build — where it would be false — cannot
accidentally claim it.

## Mirroring — the easy bug

The preview is mirrored so it behaves like a mirror rather than a video call of
a stranger. Mirroring is where bbox overlays usually go wrong: the server sees
un-mirrored pixels, the user sees mirrored ones, and every box lands on the
wrong side of the face.

This app sidesteps the whole class of bug with one invariant:

> **The image POSTed to the server is exactly the image rendered in the result
> view.**

`Camera.captureFrame()` bakes the mirror into the captured canvas (via
`ctx.translate(w, 0); ctx.scale(-1, 1)`) whenever the preview is mirrored, so
the uploaded pixels *are* the pixels the user was looking at. The returned
bboxes are therefore already in the displayed image's coordinate space and the
overlay applies no flip at all. Toggling "Mirror (selfie view)" off changes both
the preview and the capture together, so the invariant holds either way.

The same reasoning covers uploads: files are decoded to a canvas with EXIF
orientation applied and re-encoded before upload, because browsers honour EXIF
rotation and OpenCV's `imdecode` does not — otherwise a phone photo would come
back with boxes rotated 90° away from the face.

Overlay boxes are positioned in **percentages** of the analysed image, so they
stay aligned at any display size, and the result canvas is laid out at its
intrinsic aspect ratio so it is never letterboxed inside its container.

## Layout

```
web/
├── index.html
├── parity.html          driver surface for the parity harness
├── .env.static          VITE_ENGINE=browser, loaded by `--mode static`
├── scripts/
│   └── stage-assets.mjs copies models + the ONNX runtime into public/
├── public/              staged build inputs; gitignored except models.json
└── src/
    ├── main.ts          wiring, UI states, upload/capture flows
    ├── engine.ts        the interface the UI talks to, and the engine choice
    ├── engine-server.ts the FastAPI path (a wrapper over api.ts)
    ├── camera.ts        getUserMedia, device selection, mirrored capture
    ├── api.ts           fetch wrappers, timeouts, error extraction
    ├── overlay.ts       canvas draw + bbox/confidence rendering
    ├── types.ts         API + models.json response types
    ├── parity.ts        exposes window.__parity for the harness
    ├── styles.css
    └── browser/         the client-side inference path
        ├── engine.ts    orchestration, lazy per-model loading
        ├── yunet.ts     YuNet + a port of OpenCV's post-processing
        ├── preprocess.ts port of server/preprocessing.py
        ├── cv-resize.ts port of cv2.resize (INTER_AREA + INTER_LINEAR)
        ├── decode.ts    port of the median decode in server/predictor.py
        └── ort.ts       onnxruntime-web setup + progress-reporting fetch
```
