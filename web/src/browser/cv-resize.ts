/**
 * A bit-exact port of `cv2.resize` for 8-bit interleaved images.
 *
 * WHY THIS EXISTS AT ALL
 * ----------------------
 * The obvious way to resize an image in a browser is
 * `ctx.drawImage(source, 0, 0, 224, 224)`. It is one line, it is hardware
 * accelerated, and it is the wrong thing to do here. The canvas downscaling
 * filter is unspecified -- it differs between Chromium, Firefox and WebKit, and
 * between GPU and software paths on the same browser -- so the 224x224 tensor
 * the model sees would differ from the one `server/preprocessing.py` produces,
 * by an amount nobody could bound and nobody would notice.
 *
 * That is precisely the failure mode this project has already had once: a wrong
 * crop margin that cost 2.4x accuracy while every test stayed green. A resize
 * that is "close enough" is the same bug wearing different clothes. So instead
 * of measuring a tolerance and hoping, the browser runs the *same algorithm*:
 * this file is a line-for-line port of the relevant paths in OpenCV's
 * `modules/imgproc/src/resize.cpp` (4.x), including its fixed-point arithmetic
 * and its round-half-to-even. The parity harness then asserts equality rather
 * than closeness, which means any future divergence is detectable instead of
 * being absorbed into a tolerance.
 *
 * WHAT IS PORTED
 * --------------
 * Only what `server/preprocessing.py` and `server/detector.py` actually reach:
 *
 *   INTER_AREA, scale >= 1   -- downscaling. Both OpenCV variants:
 *                               `resizeAreaFast_` (integer scale factors) and
 *                               the generic `resizeArea_` decimation tables.
 *   INTER_LINEAR             -- the CV_8U fixed-point path (`HResizeLinear` +
 *                               `VResizeLinear<uchar, int, short, ...>`), used
 *                               when the crop is smaller than the model input.
 *
 * Anything else throws rather than silently picking a lookalike.
 *
 * NUMERICS
 * --------
 * OpenCV accumulates INTER_AREA in float32 and INTER_LINEAR in 32-bit fixed
 * point with 11 fractional bits. JavaScript numbers are float64, so every
 * intermediate that OpenCV holds in a float is passed through `Math.fround`,
 * and every integer intermediate uses `| 0` / `>>` semantics that match C's
 * 32-bit int. `cvRound` is round-half-to-even, not JavaScript's round-half-up.
 */

/** OpenCV's `cvFloor`. */
function cvFloor(v: number): number {
  return Math.floor(v);
}

/**
 * OpenCV's `cvRound`: round to nearest, ties to even.
 *
 * `Math.round` rounds ties away from zero (`Math.round(2.5) === 3`,
 * `Math.round(-2.5) === -2`), which disagrees with OpenCV on exactly the
 * half-integers that INTER_AREA produces constantly on flat image regions.
 */
export function cvRound(v: number): number {
  const floor = Math.floor(v);
  if (v - floor === 0.5) return floor % 2 === 0 ? floor : floor + 1;
  return Math.round(v);
}

/** `saturate_cast<uchar>` from a float. */
function saturateCastU8(v: number): number {
  const r = cvRound(v);
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

/** `saturate_cast<short>` from a float. */
function saturateCastS16(v: number): number {
  const r = cvRound(v);
  return r < -32768 ? -32768 : r > 32767 ? 32767 : r;
}

/** OpenCV's `clip(x, a, b)`: clamps into `[a, b - 1]`. */
function clip(x: number, a: number, b: number): number {
  return x >= a ? (x < b ? x : b - 1) : a;
}

export const INTER_LINEAR = 1;
export const INTER_AREA = 3;

/** 11 fractional bits; `INTER_RESIZE_COEF_SCALE` is `1 << 11`. */
const INTER_RESIZE_COEF_BITS = 11;
const INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

export interface Mat8U {
  /** Interleaved, row-major, `height * width * channels` bytes. */
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
}

export function mat8u(width: number, height: number, channels: number): Mat8U {
  return { data: new Uint8Array(width * height * channels), width, height, channels };
}

/**
 * `resizeAreaFast_<uchar, int, ...>` -- the integer-scale-factor INTER_AREA.
 *
 * Reached whenever `src/dst` is an exact integer in both axes (a 448 -> 224
 * crop, for instance, which is common). Sums are integers; only the final
 * `sum * scale` is float.
 */
function resizeAreaFast(
  src: Mat8U,
  dst: Mat8U,
  iscaleX: number,
  iscaleY: number,
): void {
  const cn = src.channels;
  const area = iscaleX * iscaleY;
  const scale = Math.fround(1 / area);
  const fastMode = iscaleX === 2 && iscaleY === 2 && (cn === 1 || cn === 3 || cn === 4);
  const srcStep = src.width * cn;
  const dstStep = dst.width * cn;
  const dwidth1 = Math.floor(src.width / iscaleX) * cn;
  const dsizeWidth = dst.width * cn;
  const ssizeWidth = src.width * cn;

  // `ofs`: the flat offsets of the block being averaged. `xofs`: where each
  // destination column's block starts.
  const ofs = new Int32Array(area);
  for (let sy = 0, k = 0; sy < iscaleY; sy++) {
    for (let sx = 0; sx < iscaleX; sx++) ofs[k++] = sy * srcStep + sx * cn;
  }
  const xofs = new Int32Array(dsizeWidth);
  for (let dx = 0; dx < dst.width; dx++) {
    const j = dx * cn;
    const sx = iscaleX * j;
    for (let k = 0; k < cn; k++) xofs[j + k] = sx + k;
  }

  for (let dy = 0; dy < dst.height; dy++) {
    const D = dy * dstStep;
    const sy0 = dy * iscaleY;
    const w = sy0 + iscaleY <= src.height ? dwidth1 : 0;

    if (sy0 >= src.height) {
      for (let dx = 0; dx < dsizeWidth; dx++) dst.data[D + dx] = 0;
      continue;
    }

    let dx = 0;
    const rowBase = sy0 * srcStep;
    if (fastMode) {
      // `ResizeAreaFastVec`: for an exact 2x2 decimation with 1, 3 or 4
      // channels OpenCV does NOT average in float. It uses integer
      // `(a + b + c + d + 2) >> 2`, which rounds halves UP where the float
      // path's `cvRound` rounds them to even. On random pixel data the two
      // disagree on ~12% of samples, so this is not a detail.
      const next = rowBase + srcStep;
      for (; dx < w; dx++) {
        const c = dx % cn;
        const base = (dx - c) * 2 + c;
        dst.data[D + dx] =
          (src.data[rowBase + base]! +
            src.data[rowBase + base + cn]! +
            src.data[next + base]! +
            src.data[next + base + cn]! +
            2) >>
          2;
      }
    }
    for (; dx < w; dx++) {
      const S = rowBase + xofs[dx]!;
      let sum = 0;
      for (let k = 0; k < area; k++) sum += src.data[S + ofs[k]!]!;
      dst.data[D + dx] = saturateCastU8(Math.fround(sum * scale));
    }

    // Ragged right/bottom edge: average only the pixels that exist.
    for (; dx < dsizeWidth; dx++) {
      let sum = 0;
      let count = 0;
      const sx0 = xofs[dx]!;
      if (sx0 >= ssizeWidth) {
        dst.data[D + dx] = 0;
      }
      for (let sy = 0; sy < iscaleY; sy++) {
        if (sy0 + sy >= src.height) break;
        const S = (sy0 + sy) * srcStep + sx0;
        for (let sx = 0; sx < iscaleX * cn; sx += cn) {
          if (sx0 + sx >= ssizeWidth) break;
          sum += src.data[S + sx]!;
          count++;
        }
      }
      dst.data[D + dx] = saturateCastU8(Math.fround(sum / count));
    }
  }
}

interface DecimateAlpha {
  si: number;
  di: number;
  alpha: number;
}

/** `computeResizeAreaTab` verbatim. */
function computeResizeAreaTab(
  ssize: number,
  dsize: number,
  cn: number,
  scale: number,
): DecimateAlpha[] {
  const tab: DecimateAlpha[] = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);

    let sx1 = Math.ceil(fsx1);
    let sx2 = cvFloor(fsx2);

    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);

    if (sx1 - fsx1 > 1e-3) {
      tab.push({ di: dx * cn, si: (sx1 - 1) * cn, alpha: Math.fround((sx1 - fsx1) / cellWidth) });
    }
    for (let sx = sx1; sx < sx2; sx++) {
      tab.push({ di: dx * cn, si: sx * cn, alpha: Math.fround(1.0 / cellWidth) });
    }
    if (fsx2 - sx2 > 1e-3) {
      tab.push({
        di: dx * cn,
        si: sx2 * cn,
        alpha: Math.fround(Math.min(Math.min(fsx2 - sx2, 1.0), cellWidth) / cellWidth),
      });
    }
  }
  return tab;
}

/**
 * `resizeArea_<uchar, float>` -- generic INTER_AREA via decimation tables.
 *
 * The horizontal pass accumulates into `buf` per source row, the vertical pass
 * accumulates `beta * buf` into `sum`, and a destination row is emitted when
 * the table's `di` moves on. Accumulation is float32 throughout, hence the
 * `Math.fround` on every partial.
 */
function resizeAreaGeneric(
  src: Mat8U,
  dst: Mat8U,
  scaleX: number,
  scaleY: number,
): void {
  const cn = src.channels;
  const dwidth = dst.width * cn;
  const srcStep = src.width * cn;
  const dstStep = dst.width * cn;

  const xtab = computeResizeAreaTab(src.width, dst.width, cn, scaleX);
  const ytab = computeResizeAreaTab(src.height, dst.height, 1, scaleY);

  const buf = new Float32Array(dwidth);
  const sum = new Float32Array(dwidth);

  let prevDy = ytab.length > 0 ? ytab[0]!.di : 0;

  for (let j = 0; j < ytab.length; j++) {
    const beta = ytab[j]!.alpha;
    const dy = ytab[j]!.di;
    const sy = ytab[j]!.si;

    const S = sy * srcStep;
    buf.fill(0);
    for (let k = 0; k < xtab.length; k++) {
      const sxn = xtab[k]!.si;
      const dxn = xtab[k]!.di;
      const alpha = xtab[k]!.alpha;
      for (let c = 0; c < cn; c++) {
        // Deliberately NOT `fround(product)` then add: compilers contract
        // `buf + S*alpha` into a fused multiply-add, and OpenCV's SIMD path
        // uses `v_muladd` explicitly. Computing the product in float64 and
        // rounding once reproduces FMA exactly for float32 inputs.
        buf[dxn + c] = Math.fround(buf[dxn + c]! + src.data[S + sxn + c]! * alpha);
      }
    }

    if (dy !== prevDy) {
      const D = prevDy * dstStep;
      for (let dx = 0; dx < dwidth; dx++) dst.data[D + dx] = saturateCastU8(sum[dx]!);
      for (let dx = 0; dx < dwidth; dx++) sum[dx] = Math.fround(beta * buf[dx]!);
      prevDy = dy;
    } else {
      for (let dx = 0; dx < dwidth; dx++) {
        // `inter_area::muladd`, i.e. `sum += beta * buf` -- fused, see above.
        sum[dx] = Math.fround(sum[dx]! + beta * buf[dx]!);
      }
    }
  }

  const D = prevDy * dstStep;
  for (let dx = 0; dx < dwidth; dx++) dst.data[D + dx] = saturateCastU8(sum[dx]!);
}

/**
 * `resizeGeneric_<HResizeLinear<uchar,int,short,2048>, VResizeLinear<uchar,...>>`.
 *
 * The CV_8U bilinear path is fixed point, not float: weights are 11-bit
 * fractions and the vertical pass finishes with the exact shift sequence
 * OpenCV uses. Reproducing the shifts rather than the intent is the whole
 * point -- a float reimplementation lands within a least-significant bit,
 * which is close enough to hide a real bug behind.
 *
 * `areaMode` reproduces OpenCV's INTER_AREA-when-upscaling fallback, which
 * uses the same machinery with differently derived coefficients.
 */
function resizeLinearU8(
  src: Mat8U,
  dst: Mat8U,
  scaleX: number,
  scaleY: number,
  areaMode: boolean,
): void {
  const cn = src.channels;
  const invScaleX = 1 / scaleX;
  const invScaleY = 1 / scaleY;
  const dsizeWidth = dst.width * cn;
  const srcStep = src.width * cn;
  const dstStep = dst.width * cn;

  const ksize = 2;
  const ksize2 = 1;

  const xofs = new Int32Array(dsizeWidth);
  const ialpha = new Int16Array(dsizeWidth * ksize);
  const yofs = new Int32Array(dst.height);
  const ibeta = new Int16Array(dst.height * ksize);

  let xmin = 0;
  let xmax = dst.width;

  for (let dx = 0; dx < dst.width; dx++) {
    let sx: number;
    let fx: number;
    if (!areaMode) {
      fx = Math.fround((dx + 0.5) * scaleX - 0.5);
      sx = cvFloor(fx);
      fx = Math.fround(fx - sx);
    } else {
      sx = cvFloor(dx * scaleX);
      fx = Math.fround(dx + 1 - (sx + 1) * invScaleX);
      fx = fx <= 0 ? 0 : Math.fround(fx - cvFloor(fx));
    }

    if (sx < ksize2 - 1) {
      xmin = dx + 1;
      if (sx < 0) {
        fx = 0;
        sx = 0;
      }
    }
    if (sx + ksize2 >= src.width) {
      xmax = Math.min(xmax, dx);
      if (sx >= src.width - 1) {
        fx = 0;
        sx = src.width - 1;
      }
    }

    sx *= cn;
    for (let k = 0; k < cn; k++) xofs[dx * cn + k] = sx + k;

    const cbuf0 = Math.fround(1 - fx);
    const cbuf1 = fx;
    const a0 = saturateCastS16(Math.fround(cbuf0 * INTER_RESIZE_COEF_SCALE));
    const a1 = saturateCastS16(Math.fround(cbuf1 * INTER_RESIZE_COEF_SCALE));
    for (let k = 0; k < cn; k++) {
      ialpha[dx * cn * ksize + k * ksize] = a0;
      ialpha[dx * cn * ksize + k * ksize + 1] = a1;
    }
  }

  for (let dy = 0; dy < dst.height; dy++) {
    let sy: number;
    let fy: number;
    if (!areaMode) {
      fy = Math.fround((dy + 0.5) * scaleY - 0.5);
      sy = cvFloor(fy);
      fy = Math.fround(fy - sy);
    } else {
      sy = cvFloor(dy * scaleY);
      fy = Math.fround(dy + 1 - (sy + 1) * invScaleY);
      fy = fy <= 0 ? 0 : Math.fround(fy - cvFloor(fy));
    }
    yofs[dy] = sy;
    ibeta[dy * ksize] = saturateCastS16(Math.fround(Math.fround(1 - fy) * INTER_RESIZE_COEF_SCALE));
    ibeta[dy * ksize + 1] = saturateCastS16(Math.fround(fy * INTER_RESIZE_COEF_SCALE));
  }

  // `xmin` exists in OpenCV only to bound its SIMD prologue; the scalar loop
  // below covers the same range, so only `xmax` is needed here.
  void xmin;
  const xmaxC = xmax * cn;

  // Two row buffers, reused across destination rows exactly as OpenCV does:
  // consecutive output rows usually share a source row, and recomputing it
  // would be wasted work (and, with fixed point, identical anyway).
  const rows: Int32Array[] = [new Int32Array(dsizeWidth), new Int32Array(dsizeWidth)];
  const prevSy = [-1, -1];

  const hresize = (sy: number, into: Int32Array): void => {
    const S = sy * srcStep;
    let dx = 0;
    for (; dx < xmaxC; dx++) {
      const sx = xofs[dx]!;
      into[dx] = src.data[S + sx]! * ialpha[dx * 2]! + src.data[S + sx + cn]! * ialpha[dx * 2 + 1]!;
    }
    for (; dx < dsizeWidth; dx++) {
      into[dx] = src.data[S + xofs[dx]!]! * INTER_RESIZE_COEF_SCALE;
    }
  };

  for (let dy = 0; dy < dst.height; dy++) {
    const sy0 = yofs[dy]!;
    const want = [clip(sy0, 0, src.height), clip(sy0 + 1, 0, src.height)];

    // Reuse rule from `resizeGeneric_Invoker`: if row k's source row is already
    // in slot k1 >= k, copy it across rather than recomputing.
    for (let k = 0; k < 2; k++) {
      let reused = false;
      for (let k1 = k; k1 < 2; k1++) {
        if (want[k] === prevSy[k1]) {
          if (k1 > k) rows[k]!.set(rows[k1]!);
          reused = true;
          break;
        }
      }
      if (!reused) hresize(want[k]!, rows[k]!);
      prevSy[k] = want[k]!;
    }

    const b0 = ibeta[dy * 2]!;
    const b1 = ibeta[dy * 2 + 1]!;
    const S0 = rows[0]!;
    const S1 = rows[1]!;
    const D = dy * dstStep;
    for (let x = 0; x < dsizeWidth; x++) {
      // `uchar(( ((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2) >> 2)`.
      // The C cast to uchar truncates rather than saturating; `& 0xff` is the
      // same thing for the values this can produce.
      const v = ((((b0 * (S0[x]! >> 4)) >> 16) + ((b1 * (S1[x]! >> 4)) >> 16) + 2) >> 2) & 0xff;
      dst.data[D + x] = v;
    }
  }
}

/**
 * `cv2.resize(src, (dstWidth, dstHeight), interpolation=...)` for CV_8U.
 *
 * Mirrors the dispatch in `cv::hal::resize`, including the two cases where
 * OpenCV silently substitutes one interpolation for another.
 */
export function resize8U(
  src: Mat8U,
  dstWidth: number,
  dstHeight: number,
  interpolation: number,
): Mat8U {
  if (dstWidth <= 0 || dstHeight <= 0) throw new Error('resize8U: empty destination');

  const dst = mat8u(dstWidth, dstHeight, src.channels);

  const invScaleX = dstWidth / src.width;
  const invScaleY = dstHeight / src.height;
  const scaleX = src.width / dstWidth;
  const scaleY = src.height / dstHeight;

  const iscaleX = cvRound(scaleX);
  const iscaleY = cvRound(scaleY);
  const isAreaFast =
    Math.abs(scaleX - iscaleX) < Number.EPSILON && Math.abs(scaleY - iscaleY) < Number.EPSILON;

  let interp = interpolation;
  // "in case of scale_x && scale_y is equal to 2 INTER_AREA (fast) also is
  // equal to INTER_LINEAR" -- OpenCV substitutes, so we must too.
  if (interp === INTER_LINEAR && isAreaFast && iscaleX === 2 && iscaleY === 2) {
    interp = INTER_AREA;
  }

  if (interp === INTER_AREA && scaleX >= 1 && scaleY >= 1) {
    if (isAreaFast) resizeAreaFast(src, dst, iscaleX, iscaleY);
    else resizeAreaGeneric(src, dst, scaleX, scaleY);
    return dst;
  }

  if (interp === INTER_LINEAR || interp === INTER_AREA) {
    resizeLinearU8(src, dst, scaleX, scaleY, interp === INTER_AREA);
    return dst;
  }

  void invScaleX;
  void invScaleY;
  throw new Error(`resize8U: unsupported interpolation ${interpolation}`);
}

/** `cv2.copyMakeBorder(..., cv2.BORDER_REPLICATE)`. */
export function copyMakeBorderReplicate(
  src: Mat8U,
  top: number,
  bottom: number,
  left: number,
  right: number,
): Mat8U {
  if (top === 0 && bottom === 0 && left === 0 && right === 0) return src;
  const cn = src.channels;
  const dst = mat8u(src.width + left + right, src.height + top + bottom, cn);
  for (let y = 0; y < dst.height; y++) {
    const sy = Math.min(Math.max(y - top, 0), src.height - 1);
    for (let x = 0; x < dst.width; x++) {
      const sx = Math.min(Math.max(x - left, 0), src.width - 1);
      const s = (sy * src.width + sx) * cn;
      const d = (y * dst.width + x) * cn;
      for (let c = 0; c < cn; c++) dst.data[d + c] = src.data[s + c]!;
    }
  }
  return dst;
}

/** The `[x, y, w, h]` sub-rectangle of `src`, copied. Mirrors numpy slicing. */
export function roi(src: Mat8U, x: number, y: number, w: number, h: number): Mat8U {
  const cn = src.channels;
  const dst = mat8u(w, h, cn);
  for (let r = 0; r < h; r++) {
    const s = ((y + r) * src.width + x) * cn;
    dst.data.set(src.data.subarray(s, s + w * cn), r * w * cn);
  }
  return dst;
}
