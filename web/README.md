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

### Caveats at the age extremes

Thresholds and wording here are conditioned on the **predicted** age, because
that is the only thing the UI knows. This matters: binned by *true* age the
model looks catastrophic at the top (MAE 10.8 at 80+), but binned by what it
actually displays the picture is quite different, because it rarely commits to
an extreme number and is roughly right when it does.

Measured end to end over 1,184 UTKFace test images at the shipped margin:

| shown | n | MAE | bias |
| ----: | -: | --: | ---: |
| 0–5 | 25 | 3.09 | +1.54 |
| 5–10 | 103 | 3.98 | +3.55 |
| 10–12 | 23 | 4.63 | +3.49 |
| 40–50 | 121 | 7.60 | +2.83 |
| 65–70 | 34 | 8.19 | −0.20 |
| 70–75 | 18 | 6.56 | −0.69 |
| 75+ | 22 | 5.83 | −3.48 |

So the honest caveats are about a systematic **offset**, not about precision
collapsing — and the young end is in fact the *most* accurate region by MAE.

* Shown **under 12** → runs ~3 years high, and the model never outputs below
  about 3, so for an infant the number is a floor rather than a measurement.
* Shown **over 65** → runs low, increasingly so with age. The model compresses
  the top of its range: a face in its mid-eighties typically displays as ~70.
  The threshold is 65 rather than 70 precisely because of that compression — at
  70 the note would miss the very people it is for.

`tailCaveat()` in `src/overlay.ts` adds a short flag to the box label and a
fuller explanation to the face card, both folded into the box's `aria-label` so
the warning is not purely visual.

The estimate is **not corrected**. Quietly subtracting the bias would bake a
dataset artefact into the served answer and hide the model's real behaviour.

The worst band by MAE is actually **40–70** (7.0–8.2), not the extremes. It is
deliberately not caveated — flagging most of the range would dilute the signal,
and the confidence bar already varies there.

For context on why a single number is shown at all: the predicted distribution's
standard deviation averages **12.7 years** on the test set, far wider than the
5.5-year MAE implies. The product decision is to lead with one number; the range
remains in the API response and under the advanced panel.

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
| Age at either extreme | A caveat under the estimate when the shown age is < 12 or > 65 |

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
