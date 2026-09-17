# web/ — camera + upload UI

Vite + vanilla TypeScript. No framework, no runtime dependencies.

Take a photo with your webcam (or drop in an image), and the app posts it to the
[`server/`](../server/README.md) API and draws each detected face with an **age
range**, a point estimate, and a confidence indicator.

## Run

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

The API must be running on `http://127.0.0.1:8000` — from the repo root,
`make api`, or `make dev` / `./scripts/dev.sh` to start both at once.

| Script            | Does |
| ----------------- | ---- |
| `npm run dev`     | Dev server with HMR on port 5173 |
| `npm run build`   | Type-check (`tsc --noEmit`) then bundle to `dist/` |
| `npm run preview` | Serve the production bundle |
| `npm run typecheck` | Type-check only |

### Configuration

| Variable        | Default                 | Meaning |
| --------------- | ----------------------- | ------- |
| `VITE_API_BASE` | `http://127.0.0.1:8000` | API origin |

```bash
VITE_API_BASE=http://127.0.0.1:9000 npm run dev
```

Whatever origin the UI is served from must be in the server's `CORS_ORIGINS`.

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

### Accuracy — the number that matters

**Against real chronological ages the MAE is 8.52 years** (APPA-REAL, 7,534
images, real ages, YuNet on full original scenes).

The in-corpus figure of 4.76 on UTKFace is *not* accuracy. UTKFace's labels are
DEX-algorithm estimates, so that number measures agreement with a labelling
method. Quote 8.52 to users.

The gap is not a cropping problem — framing accounts for only ~0.4 years of it.
The revealing part is the slopes: **0.935** against *apparent* age versus
**0.817** against *real* age. We are a well-calibrated predictor of how old a
face **looks** and a compressed predictor of how old someone **is**. Our
per-image error also correlates **+0.445** with the human apparent-vs-real gap —
when people misjudge a face, we misjudge it the same way, and in the same
direction.

### Where the estimate is least reliable

Thresholds and wording are conditioned on the **predicted** age, because that is
the only thing the UI knows. Binning by *true* age answers a question the UI
cannot ask. This distinction has now inverted a conclusion twice, so check it
first before changing anything here.

Binned by displayed age (APPA-REAL, n = 7,534):

| shown | n | MAE | bias |
| ----: | -: | --: | ---: |
| 0–10 | 617 | 3.34 | −1.01 |
| 10–20 | 1126 | 6.77 | −4.43 |
| 20–30 | 1897 | 6.34 | +0.00 |
| 30–40 | 1377 | 8.04 | +3.04 |
| 40–50 | 651 | 10.27 | +6.78 |
| 50–60 | 1198 | 13.72 | +10.94 |
| 60–70 | 467 | 12.48 | +7.76 |
| 70+ | 201 | 12.22 | +7.30 |

**One caveat ships: ages shown 40 and over.** MAE 12.48 against 6.53 below 40,
with bias **+8.98** — so the number is both about twice as imprecise and
systematically high. Someone displayed as 55 averages 46. It fires on 33% of
faces, which is a lot, but the effect is large, monotonic in displayed age, and
stable across APPA-REAL's splits (+8.75 / +9.14 / +9.07). Independently
corroborated on FG-NET (same direction, same shape).

Two warnings are deliberately **not** shipped:

* **No teenager caveat.** By *true* age, 10–19 is our worst region relative to
  human raters (bias +7.42 — we read teenagers as much older than they are).
  But by *displayed* age the sign flips: those shown as 10–20 average bias
  **−4.43**. A "teenagers read old" warning would fire on a population for whom
  the opposite is true. The band is also *more* accurate than average (MAE 6.77
  vs 8.52), so there is no precision case either, and the dip is non-monotonic
  against its neighbours (−1.01, −4.43, +0.00) — the signature of a local
  artefact rather than a stable effect.
* **No old-age caveat.** At true 70–79 our MAE is 10.08 against a single human
  rater's 10.01, and our bias (−7.12) is *smaller* than the human crowd's own
  (−7.99). Those faces genuinely read young to people. Not our defect.

`tailCaveat()` in `src/overlay.ts` adds a short flag to the box label and a
fuller explanation to the face card, both folded into the box's `aria-label` so
the warning is not purely visual.

The estimate is **not corrected**. Subtracting the bias would bury a known,
measured limitation inside a number that looks authoritative.

For context on why a single number is shown at all: the reported interval
averages 11.8 years wide. The product decision is to lead with one number; the
range remains in the API response and under the advanced panel.

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
| Age shown 40+ | A caveat: reads high by ~9 years and is ~2x less precise |

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
└── src/
    ├── main.ts      wiring, UI states, upload/capture flows
    ├── camera.ts    getUserMedia, device selection, mirrored capture
    ├── api.ts       fetch wrappers, timeouts, error extraction
    ├── overlay.ts   canvas draw + bbox/confidence rendering
    ├── types.ts     API response types
    └── styles.css
```
