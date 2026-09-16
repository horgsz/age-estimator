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

### Where the estimate is least reliable

Thresholds and wording here are conditioned on the **predicted** age, because
that is the only thing the UI knows. Binning by *true* age answers a question
the UI cannot ask.

These were re-derived after the decode changed from soft-expectation to the
distribution median, and the change **inverted the previous conclusion**. The
old decode suffered a label-smoothing pedestal that dragged every estimate
toward 50, so both extremes were badly offset and both needed a warning. The
median removes that, and with it both tail caveats.

Measured end to end over 1,184 UTKFace test images at the shipped margin:

| shown | n | MAE | bias |
| ----: | -: | --: | ---: |
| 0–5 | 109 | 0.98 | −0.71 |
| 5–10 | 49 | 2.33 | −0.82 |
| 10–12 | 13 | 3.54 | −2.31 |
| 20–30 | 386 | 3.52 | −0.49 |
| 30–40 | 226 | 5.90 | −0.02 |
| 40–50 | 85 | 6.61 | +2.71 |
| 50–60 | 138 | 7.18 | +2.09 |
| 60–70 | 81 | 7.98 | +0.80 |
| 70–75 | 16 | 7.81 | +2.56 |
| 75+ | 34 | 4.82 | +0.59 |

**There is no longer a young caveat.** Ages shown under 12 are now the most
accurate region the model has — MAE 1.76 — and the output reaches down to 1, so
there is no floor to warn about.

**There is no longer an old-age caveat either**, and this one is subtle. Binned
by *true* age the top still compresses (bias −7.8 at 80+), which is what an
offline eval sees and is tempting to warn about. But binned by *displayed* age,
everything shown at 80+ has bias **+1.10** — a ">= 80, reads low" rule would
fire on a population it is not actually wrong about. There is no threshold at
which a directional old-age warning is supportable.

What remains is a real precision story in mid-to-late adulthood:

* Shown **40–75** → 27% of cases, MAE **7.27** against **3.83** everywhere else,
  with a mild tendency to read old. Nearly a 2x difference, so the UI says so.

`tailCaveat()` in `src/overlay.ts` adds a short flag to the box label and a
fuller explanation to the face card, both folded into the box's `aria-label` so
the warning is not purely visual.

The estimate is **not corrected**. Quietly subtracting a bias would bake a
dataset artefact into the served answer and hide the model's real behaviour.

For context on why a single number is shown at all: the reported interval
averages **11.8 years** wide, far wider than the 4.8-year MAE implies. The
product decision is to lead with one number; the range remains in the API
response and under the advanced panel.

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
| Age shown 40–75 | A caveat under the estimate: the model's least precise range |

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
