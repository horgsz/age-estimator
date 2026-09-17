# Parity results

Measured with [`make parity`](README.md#running-it). Regenerate this file's
numbers by re-running it; do not edit them by hand.

- **Commit:** `e89cccf` (first run with the harness in place)
- **Date:** 2026-09-17
- **Machine:** macOS, Apple silicon, Chromium 153 (Playwright headless shell)
- **Python:** 3.13, OpenCV 4.14.0, PyTorch 2.14.0
- **Browser:** onnxruntime-web 1.22.0, WASM execution provider, **1 thread**

## Verdict

**The two paths agree exactly.** Every detected box is identical, every
224×224×3 input tensor is bit-identical — not close, identical: zero differing
float32 elements out of 150,528 per face, and matching sha256 digests of the
tensor bytes — and every predicted age, interval bound and confidence figure is
the same.

The only non-zero difference anywhere is in the model outputs, at 3.3×10⁻⁶ to
2.5×10⁻⁵ of a logit. That is the ONNX export versus PyTorch, and it is an order
of magnitude below anything that could move a decode: the point estimate is an
integer bin index, and the interval bounds are CDF crossings.

## Per-case

| case | faces | box Δ | tensor max Δ | tensor mean Δ | differing elements | logit max Δ | age Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 1.31e-05 | 0.00 |
| `large` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 1.32e-05 | 0.00 |
| `small` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 2.48e-05 | 0.00 |
| `edge` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 1.14e-05 | 0.00 |
| `clipped` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 1.43e-05 | 0.00 |
| `two-faces` | 2 | 0 | 0.000e+00 | 0.000e+00 | 0 / 301056 | 1.55e-05 | 0.00 |
| `full-apparent` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 5.96e-06 | 0.00 |
| `strict-no-resize` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 1.30e-05 | 0.00 |
| `margin-0.25` | 1 | 0 | 0.000e+00 | 0.000e+00 | 0 / 150528 | 3.34e-06 | 0.00 |

Tensor differences are on the normalised float32 input, where one
least-significant bit of an 8-bit pixel is ≈1.7×10⁻². Zero means the crops are
byte-for-byte the same image.

What each case exercises:

| case | path it covers |
| --- | --- |
| `full` | plain detection at native size |
| `large` | 880×1100, past `MAX_DETECT_SIDE`, so detection downscales and the box is mapped back |
| `small` | face under 224 px, so `resize_crop` takes `INTER_LINEAR` instead of `INTER_AREA` |
| `edge` | face against the frame edge: the square fits and is slid back inside |
| `clipped` | face filling the frame: the square does not fit, so edge replication pads it |
| `two-faces` | NMS has real work to do, and the batch has more than one row |
| `full-apparent` | the other model, so the toggle is covered |
| `strict-no-resize` | crop side exactly 224, so there is **no resampling at all** |
| `margin-0.25` | a non-default crop margin |

## Per-face output

| case | face | python | browser |
| --- | ---: | --- | --- |
| `full` | 0 | 48 (42–52), conf 0.5455, bbox [148, 36, 97, 133] | 48 (42–52), conf 0.5455, bbox [148, 36, 97, 133] |
| `large` | 0 | 48 (42–52), conf 0.5455, bbox [326, 73, 213, 300] | 48 (42–52), conf 0.5455, bbox [326, 73, 213, 300] |
| `small` | 0 | 47 (41–53), conf 0.5000, bbox [48, 12, 34, 43] | 47 (41–53), conf 0.5000, bbox [48, 12, 34, 43] |
| `edge` | 0 | 46 (40–50), conf 0.5455, bbox [29, 39, 96, 128] | 46 (40–50), conf 0.5455, bbox [29, 39, 96, 128] |
| `clipped` | 0 | 46 (41–51), conf 0.5455, bbox [27, 9, 98, 128] | 46 (41–51), conf 0.5455, bbox [27, 9, 98, 128] |
| `two-faces` | 0 | 48 (43–52), conf 0.5714, bbox [556, 36, 94, 133] | 48 (43–52), conf 0.5714, bbox [556, 36, 94, 133] |
| `two-faces` | 1 | 48 (42–52), conf 0.5455, bbox [148, 36, 97, 133] | 48 (42–52), conf 0.5455, bbox [148, 36, 97, 133] |
| `full-apparent` | 0 | 56 (48–65), conf 0.4138, bbox [148, 36, 97, 133] | 56 (48–65), conf 0.4138, bbox [148, 36, 97, 133] |
| `strict-no-resize` | 0 | 65 (51–72), conf 0.3636, bbox [80, 90, 224, 224] | 65 (51–72), conf 0.3636, bbox [80, 90, 224, 224] |
| `margin-0.25` | 0 | 64 (56–71), conf 0.4444, bbox [148, 36, 97, 133] | 64 (56–71), conf 0.4444, bbox [148, 36, 97, 133] |

The fixture is a portrait of a man in his early fifties, so ~48 from the
real-age model and ~56 from the apparent-age one are both in the plausible
range. The point of the table is the **agreement**, not the accuracy — accuracy
is measured properly by `make eval`, on a held-out split, not on one face.

`margin-0.25` predicting 64 against `full`'s 48 on the same face is not an
error: it is the crop-margin sensitivity `server/config.py` documents, visible
directly. Widening the crop by a quarter of the box moves this face by 16 years.
That is the size of the hazard the margin constant is guarding against.

## Environment

| | |
| --- | --- |
| `crossOriginIsolated` | `false` |
| `SharedArrayBuffer` | unavailable |
| `ort.env.wasm.numThreads` | 1 |

Asserted rather than assumed. The harness's static file server deliberately
omits `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` because
GitHub Pages cannot send them, so the measurement comes from the same
single-threaded WASM configuration the deployment runs. It is not silently
falling back and it is not erroring — it is running, in one thread, at the
speeds below.

## Timings

Per-case, warm (model already loaded), on the machine above:

| case | detect | preprocess | infer |
| --- | ---: | ---: | ---: |
| `full` (400×500) | 20.7 ms | 3.4 ms | 7.5 ms |
| `large` (880×1100) | 67.7 ms | 2.0 ms | 6.2 ms |
| `small` (133×166) | 2.1 ms | 1.6 ms | 6.3 ms |
| `edge` (280×310) | 5.4 ms | 1.5 ms | 5.6 ms |
| `clipped` (168×181) | 2.2 ms | 0.7 ms | 5.9 ms |
| `two-faces` (800×500) | 22.9 ms | 1.8 ms | 13.4 ms (2 faces) |

Age inference is ~6 ms per face and does not depend on the frame size; detection
does, because YuNet runs on the whole frame. A 1280×720 webcam frame lands
between the `full` and `large` rows.

## Cold load

Measured in a fresh browser context — empty HTTP cache — from navigation to a
rendered age, which is what a visitor actually waits for. `DOMContentLoaded`
happens far earlier and is not the number that matters.

| | localhost |
| --- | ---: |
| navigation to interactive | 21 ms |
| first result (detector + model + WASM + inference) | 187 ms |
| bytes transferred | 17.20 MB |

Breakdown of the 17.20 MB:

| asset | bytes |
| --- | ---: |
| `ort-wasm-simd-threaded.wasm` | 11,210,254 |
| `age_model_realgt.onnx` + `face_detection_yunet_2023mar.dynamic.onnx` | 6,729,552 |
| app JavaScript | 64,710 |
| `ort-wasm-simd-threaded.mjs` | 20,856 |
| `models.json` | 5,073 |
| HTML | 1,208 |

**These are uncompressed, over loopback, and are the wrong number for a real
visitor twice over.** GitHub Pages serves `.wasm` and `.onnx` compressed, which
takes the WASM runtime from 11.2 MB to ≈2.8 MB on the wire, and the transfer
itself is then bounded by the network rather than by anything measured here. See
the deployed figures in the section below.

The second visit is effectively free: both files are fetched with
`cache: 'force-cache'` under stable names, so they come from the HTTP cache.

## Known residual differences

None in this harness. Two are known to exist outside it and are recorded so they
are not rediscovered as surprises:

1. **`cv-resize.ts` INTER_AREA on pure noise.** Differential testing against
   `cv2.resize` over 30 synthetic cases found 10 differing pixels in total, all
   in random-noise images, and all where the exact area average lands within
   2×10⁻⁵ of a `.5` rounding tie — close enough that float32 accumulation order
   decides which way it goes. On smooth images (photographs) there is no
   difference at all, which is why every case above reports exactly zero.

2. **JPEG decoding.** The browser's JPEG decoder and OpenCV's disagree by a
   least-significant bit or two on some coefficients. The fixtures are PNG
   precisely so this is not folded into the measurement. It is a real
   difference in production for *uploaded* JPEGs, but it is bounded by one bit
   of an 8-bit pixel and it is not something either path can fix.
