/**
 * YuNet face detection in the browser.
 *
 * WHY YUNET AND NOT MEDIAPIPE
 * ---------------------------
 * MediaPipe Face Detector is the easier option: it is a packaged WASM task with
 * its own model and needs no post-processing. It was rejected because it is a
 * *different detector*, and the crop geometry this app depends on is calibrated
 * to YuNet's box convention.
 *
 * `server/config.py` records that `CROP_MARGIN` was chosen by sweeping
 * end-to-end MAE, that the optimum is 0.0, and that the error curve is strongly
 * asymmetric: -0.05 costs 0.11 years, +0.2 costs 0.71, +0.4 costs 2.97. A
 * detector whose boxes are systematically, say, 15% wider than YuNet's does not
 * announce itself -- it silently shifts the effective margin into the expensive
 * side of that curve. There is no way to detect that from inside the browser
 * build, and no test would fail.
 *
 * Running the same ONNX weights OpenCV runs removes the question entirely
 * rather than answering it: the boxes are not "close to" YuNet's, they are
 * YuNet's. The measured result is in `parity/RESULTS.md`. The price is that the
 * post-processing below -- which `cv2.FaceDetectorYN` does in C++ -- has to be
 * reimplemented here, so it is a direct port of
 * `opencv/modules/objdetect/src/face_detect.cpp` (4.x), down to the truncation
 * to integer boxes before NMS and OpenCV's ties-and-ordering behaviour.
 *
 * THE INPUT IS BGR, 0..255, UNNORMALISED
 * --------------------------------------
 * `cv2.FaceDetectorYN::detect` calls `dnn::blobFromImage(pad_image)` with all
 * defaults, which means scalefactor 1, no mean subtraction and `swapRB=false`.
 * The network therefore sees raw BGR bytes as floats. Feeding it RGB, or
 * anything scaled to 0..1, produces plausible-looking but subtly wrong boxes.
 */

import type { Mat8U } from './cv-resize';
import { INTER_AREA, mat8u, resize8U } from './cv-resize';
import type { BBox } from './preprocess';
import { pyRound } from './preprocess';
import type { InferenceSession, Tensor } from 'onnxruntime-web/wasm';
import { createSession, ortTensor } from './ort';

/** Mirrors `server/detector.py`: detection runs on a downscaled copy. */
export const MAX_DETECT_SIDE = 1024;

const STRIDES = [8, 16, 32] as const;
const DIVISOR = 32;

export interface DetectorOptions {
  scoreThreshold: number;
  nmsThreshold: number;
  topK: number;
}

/** Defaults from `server/config.py`. */
export const DEFAULT_DETECTOR_OPTIONS: DetectorOptions = {
  scoreThreshold: 0.7,
  nmsThreshold: 0.3,
  topK: 50,
};

interface Candidate {
  /** Float box, as decoded. */
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
}

/**
 * `dnn::NMSBoxes(std::vector<Rect2i>, ...)` with `eta = 1`.
 *
 * Two OpenCV details that a naive NMS gets wrong and that change which box
 * survives when two detections overlap:
 *
 * 1. The boxes are truncated to `int` *before* IoU is computed
 *    (`Rect2i(int(x), int(y), int(w), int(h))`), so the overlap is measured on
 *    integer rectangles, not the float ones that get returned.
 * 2. `getMaxScoreIndex` sorts descending with `std::stable_sort`, so equal
 *    scores keep their original order, and `top_k` truncates after sorting.
 */
function nmsBoxes(
  candidates: Candidate[],
  scoreThreshold: number,
  nmsThreshold: number,
  topK: number,
): number[] {
  const indexed = candidates
    .map((c, i) => ({ c, i }))
    .filter((e) => e.c.score > scoreThreshold);
  // Stable descending sort: JS `Array.prototype.sort` is required to be stable.
  indexed.sort((a, b) => b.c.score - a.c.score);
  const pool = topK > 0 && topK < indexed.length ? indexed.slice(0, topK) : indexed;

  const rects = pool.map((e) => ({
    x: Math.trunc(e.c.x),
    y: Math.trunc(e.c.y),
    w: Math.trunc(e.c.w),
    h: Math.trunc(e.c.h),
  }));

  const keep: number[] = [];
  for (let i = 0; i < pool.length; i++) {
    let ok = true;
    for (let k = 0; k < keep.length && ok; k++) {
      const a = rects[i]!;
      const b = rects[keep[k]!]!;
      const ix = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const iy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      const inter = ix > 0 && iy > 0 ? ix * iy : 0;
      const areaA = a.w * a.h;
      const areaB = b.w * b.h;
      const union = areaA + areaB - inter;
      const overlap = union <= 0 ? 1 : inter / union;
      if (overlap > nmsThreshold) ok = false;
    }
    if (ok) keep.push(i);
  }
  return keep.map((i) => pool[i]!.i);
}

export class YuNetDetector {
  private constructor(
    private readonly session: InferenceSession,
    private readonly options: DetectorOptions,
  ) {}

  static async create(
    modelBytes: Uint8Array,
    options: DetectorOptions = DEFAULT_DETECTOR_OPTIONS,
  ): Promise<YuNetDetector> {
    return new YuNetDetector(await createSession(modelBytes), options);
  }

  /**
   * Detect faces, returning integer boxes in ORIGINAL image coordinates.
   *
   * Mirrors `server/detector.py::FaceDetector.detect` exactly, including the
   * downscale-for-speed step, the inverse scaling of the boxes, the clamp into
   * frame and the `>= 2` degenerate-box filter.
   */
  async detect(imageBgr: Mat8U): Promise<BBox[]> {
    if (imageBgr.width === 0 || imageBgr.height === 0) return [];

    const imgH = imageBgr.height;
    const imgW = imageBgr.width;
    const scale = Math.min(1, MAX_DETECT_SIDE / Math.max(imgH, imgW));

    let detImg = imageBgr;
    let detW = imgW;
    let detH = imgH;
    if (scale < 1) {
      detW = Math.max(1, pyRound(imgW * scale));
      detH = Math.max(1, pyRound(imgH * scale));
      detImg = resize8U(imageBgr, detW, detH, INTER_AREA);
    }

    const faces = await this.detectRaw(detImg);

    const invX = imgW / detW;
    const invY = imgH / detH;
    const boxes: BBox[] = [];
    for (const f of faces) {
      // `int(round(...))` in the Python: banker's rounding, not JavaScript's.
      let bx = pyRound(f.x * invX);
      let by = pyRound(f.y * invY);
      let bw = pyRound(f.w * invX);
      let bh = pyRound(f.h * invY);

      bx = Math.min(Math.max(bx, 0), Math.max(0, imgW - 1));
      by = Math.min(Math.max(by, 0), Math.max(0, imgH - 1));
      bw = Math.min(bw, imgW - bx);
      bh = Math.min(bh, imgH - by);
      if (bw >= 2 && bh >= 2) boxes.push([bx, by, bw, bh]);
    }
    return boxes;
  }

  /** Detection in the input image's own coordinate space, before rescaling. */
  private async detectRaw(img: Mat8U): Promise<Candidate[]> {
    const padW = (Math.trunc((img.width - 1) / DIVISOR) + 1) * DIVISOR;
    const padH = (Math.trunc((img.height - 1) / DIVISOR) + 1) * DIVISOR;

    // `padWithDivisor`: BORDER_CONSTANT 0 on the bottom and right only.
    const padded = mat8u(padW, padH, img.channels);
    for (let y = 0; y < img.height; y++) {
      padded.data.set(
        img.data.subarray(y * img.width * img.channels, (y + 1) * img.width * img.channels),
        y * padW * img.channels,
      );
    }

    // `blobFromImage` with defaults: NCHW, BGR order preserved, values 0..255.
    const plane = padW * padH;
    const blob = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      blob[i] = padded.data[i * 3]!;
      blob[plane + i] = padded.data[i * 3 + 1]!;
      blob[2 * plane + i] = padded.data[i * 3 + 2]!;
    }

    const inputName = this.session.inputNames[0]!;
    const feeds: Record<string, Tensor> = {
      [inputName]: ortTensor(blob, [1, 3, padH, padW]),
    };
    const out = await this.session.run(feeds);

    const candidates: Candidate[] = [];
    for (let i = 0; i < STRIDES.length; i++) {
      const stride = STRIDES[i]!;
      const cols = Math.trunc(padW / stride);
      const rows = Math.trunc(padH / stride);

      const cls = out[`cls_${stride}`]!.data as Float32Array;
      const obj = out[`obj_${stride}`]!.data as Float32Array;
      const bbox = out[`bbox_${stride}`]!.data as Float32Array;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const idx = r * cols + c;
          const clsScore = Math.min(Math.max(cls[idx]!, 0), 1);
          const objScore = Math.min(Math.max(obj[idx]!, 0), 1);
          const score = Math.sqrt(clsScore * objScore);
          if (score < this.options.scoreThreshold) continue;

          const cx = (c + bbox[idx * 4]!) * stride;
          const cy = (r + bbox[idx * 4 + 1]!) * stride;
          const w = Math.exp(bbox[idx * 4 + 2]!) * stride;
          const h = Math.exp(bbox[idx * 4 + 3]!) * stride;
          candidates.push({ x: cx - w / 2, y: cy - h / 2, w, h, score });
        }
      }
    }

    // OpenCV only runs NMS when more than one candidate survived thresholding.
    if (candidates.length <= 1) return candidates;
    const keep = nmsBoxes(
      candidates,
      this.options.scoreThreshold,
      this.options.nmsThreshold,
      this.options.topK,
    );
    return keep.map((i) => candidates[i]!);
  }
}
