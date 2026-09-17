/**
 * Face cropping + normalisation, in the browser.
 *
 * A LINE-FOR-LINE PORT OF ``server/preprocessing.py``. That module is the
 * single source of truth for how a face is cropped; this file is a second
 * implementation of it, which is a liability, not a feature. It exists only
 * because GitHub Pages cannot run Python.
 *
 * The rule that keeps it honest is `parity/`: a harness that runs the same
 * images through both implementations and compares the 224x224 tensors
 * element-wise. If you change the geometry here without changing
 * ``server/preprocessing.py`` (or vice versa), that harness fails. Do not
 * "fix" a divergence by widening its tolerance -- the project has already had
 * one silent preprocessing mismatch that cost 2.4x accuracy with a green test
 * suite, and a loosened tolerance is how the second one ships.
 *
 * The margin lives in ``server/config.py`` as ``CROP_MARGIN`` and is 0.0;
 * `parity/` asserts that the value baked into the static build still matches
 * the server's, so the two cannot drift apart unnoticed. Read the reasoning in
 * the Python module before changing it: the curve is strongly asymmetric and
 * erring wide is the dangerous direction.
 */

import type { Mat8U } from './cv-resize';
import { copyMakeBorderReplicate, INTER_AREA, INTER_LINEAR, resize8U, roi } from './cv-resize';

/** `(x, y, w, h)` in original-image pixel coords. */
export type BBox = [number, number, number, number];

export const IMAGENET_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
export const IMAGENET_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];
export const INPUT_SIZE = 224;

/**
 * Where the square crop lands, split into in-image region + padding.
 *
 * By construction `padLeft + w + padRight === side === padTop + h + padBottom`.
 */
export interface CropGeometry {
  x: number;
  y: number;
  w: number;
  h: number;
  side: number;
  padLeft: number;
  padTop: number;
  padRight: number;
  padBottom: number;
}

export function needsPadding(g: CropGeometry): boolean {
  return Boolean(g.padLeft || g.padTop || g.padRight || g.padBottom);
}

/**
 * Python's `round()` is banker's rounding; `numpy`/`int(round(x))` in the
 * source module inherits it. JavaScript's `Math.round` rounds halves away from
 * zero, which would place the crop one pixel differently whenever a box centre
 * lands on a half-pixel -- which, with integer boxes of even width, it does
 * constantly.
 */
export function pyRound(v: number): number {
  const floor = Math.floor(v);
  const frac = v - floor;
  if (frac === 0.5) return floor % 2 === 0 ? floor : floor + 1;
  return Math.round(v);
}

/** Port of `compute_crop_geometry`. */
export function computeCropGeometry(
  bbox: BBox,
  imageHeight: number,
  imageWidth: number,
  margin: number,
): CropGeometry {
  const imgH = Math.trunc(imageHeight);
  const imgW = Math.trunc(imageWidth);
  const [x, y, w, h] = bbox;
  const cx = x + w / 2;
  const cy = y + h / 2;

  const side = Math.max(1, Math.trunc(pyRound(Math.max(w, h) * (1 + 2 * margin))));

  let x0 = pyRound(cx - side / 2);
  let y0 = pyRound(cy - side / 2);

  // Slide back inside the frame only when the square actually fits; otherwise
  // stay centred on the face and let the padding below make up the difference.
  if (side <= imgW) x0 = Math.min(Math.max(x0, 0), imgW - side);
  if (side <= imgH) y0 = Math.min(Math.max(y0, 0), imgH - side);

  let ax0 = Math.max(0, x0);
  let ay0 = Math.max(0, y0);
  let ax1 = Math.min(imgW, x0 + side);
  let ay1 = Math.min(imgH, y0 + side);

  // Degenerate guard, matching the Python: a box whose centre sits outside the
  // image would otherwise give an empty slice.
  if (ax1 <= ax0) {
    ax0 = Math.min(Math.max(ax0, 0), Math.max(0, imgW - 1));
    ax1 = Math.min(imgW, ax0 + 1);
  }
  if (ay1 <= ay0) {
    ay0 = Math.min(Math.max(ay0, 0), Math.max(0, imgH - 1));
    ay1 = Math.min(imgH, ay0 + 1);
  }

  return {
    x: ax0,
    y: ay0,
    w: ax1 - ax0,
    h: ay1 - ay0,
    side,
    padLeft: ax0 - x0,
    padTop: ay0 - y0,
    padRight: x0 + side - ax1,
    padBottom: y0 + side - ay1,
  };
}

/** Port of `crop_face`: the square BGR crop, at source pixel scale. */
export function cropFace(imageBgr: Mat8U, bbox: BBox, margin: number): Mat8U {
  const g = computeCropGeometry(bbox, imageBgr.height, imageBgr.width, margin);
  let crop = roi(imageBgr, g.x, g.y, g.w, g.h);
  if (needsPadding(g)) {
    crop = copyMakeBorderReplicate(crop, g.padTop, g.padBottom, g.padLeft, g.padRight);
  }
  return crop;
}

/** Port of `resize_crop`, including the INTER_AREA / INTER_LINEAR choice. */
export function resizeCrop(cropBgr: Mat8U, size: number = INPUT_SIZE): Mat8U {
  const interp = cropBgr.height > size ? INTER_AREA : INTER_LINEAR;
  return resize8U(cropBgr, size, size, interp);
}

/**
 * Port of `normalize`: BGR uint8 HWC -> ImageNet-normalised float32 CHW (RGB).
 *
 * The channel swap is where a browser port is most likely to go quietly wrong,
 * because canvas pixel data is RGBA and OpenCV is BGR. The crop is carried as
 * BGR throughout this module precisely so this function stays a transcription
 * of the Python rather than a re-derivation.
 *
 * Every intermediate is rounded to float32 at the same points numpy rounds:
 * `v/255`, then `- mean`, then `/ std`. Folding those into one float64
 * expression drifts by ~1e-8 per element, which is harmless but would make the
 * parity harness's "bit-identical" assertion into an "approximately equal" one,
 * and approximate is exactly the standard this project is trying not to accept.
 */
const MEAN_F32 = Float32Array.from(IMAGENET_MEAN);
const STD_F32 = Float32Array.from(IMAGENET_STD);

export function normalize(cropBgr: Mat8U, out?: Float32Array): Float32Array {
  const size = cropBgr.width;
  const plane = size * cropBgr.height;
  const cn = cropBgr.channels;
  const dst = out ?? new Float32Array(3 * plane);
  // BGR source index -> RGB destination plane.
  const srcChannelForPlane = [2, 1, 0];
  for (let p = 0; p < 3; p++) {
    const sc = srcChannelForPlane[p]!;
    const mean = MEAN_F32[p]!;
    const std = STD_F32[p]!;
    const base = p * plane;
    for (let i = 0; i < plane; i++) {
      const v = Math.fround(cropBgr.data[i * cn + sc]! / 255);
      dst[base + i] = Math.fround(Math.fround(v - mean) / std);
    }
  }
  return dst;
}

/** Port of `preprocess_face`: detector bbox -> model-ready float32 CHW. */
export function preprocessFace(
  imageBgr: Mat8U,
  bbox: BBox,
  margin: number,
  size: number = INPUT_SIZE,
  out?: Float32Array,
): Float32Array {
  return normalize(resizeCrop(cropFace(imageBgr, bbox, margin), size), out);
}
