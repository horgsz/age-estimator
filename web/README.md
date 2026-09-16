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
* **Results** are drawn as boxes over the analysed frame. The headline number is
  the range `low–high`, with the point estimate shown smaller underneath —
  a single figure would imply precision the model does not have. Confidence
  drives the box's colour (red → green), its opacity, and a small bar.

### UI states

| State | What you see |
| ----- | ------------ |
| Camera off / permission denied | Overlay on the preview explaining why, with a retry button; upload still works |
| Insecure origin | Explains that `getUserMedia` needs `localhost` or https |
| Analysing | Busy status, controls disabled |
| 1+ faces | Boxes + per-face cards |
| 0 faces | `{"faces": []}` is a success: a hint about lighting/distance, not an error |
| Server error / unreachable | The server's `detail` message, or a "start the API" hint |
| Stub model | A banner from `GET /health` warning that the ages are fake |

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
