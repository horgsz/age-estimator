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
| typical error vs real age | ~6.3 yr | ~8.5 yr |
| under-18s shown as 18+ | 29.6% | 40.3% |
| band caveat | none | shown 40+ reads ~6 yr high |

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
