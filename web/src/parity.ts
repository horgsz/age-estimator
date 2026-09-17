/**
 * The browser half of the parity harness.
 *
 * This page has no UI worth the name. It exists so a headless browser can run
 * the *deployed build's* inference path -- the real modules, the real
 * onnxruntime-web session, the real WASM binaries -- over a fixed set of images
 * and hand back every intermediate for comparison against `server/`.
 *
 * It is shipped with the site rather than kept as a test-only entry point, and
 * that is deliberate: a harness that runs against a specially-built bundle
 * proves things about that bundle. Being able to point the comparison at
 * `https://horgsz.github.io/age-estimator/parity.html` means the thing measured
 * is the thing served.
 *
 * `window.__parity` is the whole contract; see `parity/run_browser.mjs`.
 */

import { BrowserEngine } from './browser/engine';
import { buildResult, decodeLogits } from './browser/decode';
import type { BBox } from './browser/preprocess';
import { INPUT_SIZE, preprocessFace } from './browser/preprocess';
import { ortTensor } from './browser/ort';

export interface ParityRequest {
  /** A data: URL for a PNG. Lossless, so both decoders must agree exactly. */
  dataUrl: string;
  model: string;
  /** Crop margin to use. Omitted means the registry's value. */
  cropMargin?: number;
  /**
   * Skip detection and preprocess these boxes instead.
   *
   * The strict half of the harness uses this to pin a crop whose side is
   * exactly 224, so the resize is an identity and any tensor difference is
   * unambiguously a cropping or colour bug rather than a resampling one.
   */
  boxes?: BBox[];
}

export interface ParityFaceResult {
  bbox: BBox;
  age: number;
  low: number;
  high: number;
  confidence: number;
  median: number;
  expectation: number;
  std: number;
  /** Raw logits, so the comparison can separate model from decode. */
  logits: number[];
  /** sha256 of the float32 tensor bytes, for a cheap identity check. */
  tensorSha256: string;
}

export interface ParityResponse {
  width: number;
  height: number;
  cropMargin: number;
  boxes: BBox[];
  faces: ParityFaceResult[];
  /** Base64 of the concatenated float32 CHW tensors, little-endian. */
  tensorsBase64: string;
  timings: { detectMs: number; preprocessMs: number; inferMs: number };
}

async function decodeToCanvas(dataUrl: string): Promise<HTMLCanvasElement> {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  // `imageOrientation: 'none'` matters: the fixtures are PNGs with no EXIF, but
  // asking for 'from-image' would make this path differ from `cv2.imdecode` the
  // moment someone adds a JPEG fixture.
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'none' });
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('no 2d context');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvas;
}

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as unknown as ArrayBuffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

const engine = new BrowserEngine();

async function run(request: ParityRequest): Promise<ParityResponse> {
  const registry = await engine.loadRegistry();
  const canvas = await decodeToCanvas(request.dataUrl);
  const image = BrowserEngine.toBgr(canvas);
  const margin = request.cropMargin ?? registry.preprocessing.crop_margin;

  const session = await engine.sessionFor(request.model);
  const detector = await engine.detectorFor(request.model);

  const t0 = performance.now();
  const boxes = request.boxes ?? (await detector.detect(image));
  const t1 = performance.now();

  if (boxes.length === 0) {
    return {
      width: canvas.width,
      height: canvas.height,
      cropMargin: margin,
      boxes: [],
      faces: [],
      tensorsBase64: '',
      timings: { detectMs: t1 - t0, preprocessMs: 0, inferMs: 0 },
    };
  }

  const per = 3 * INPUT_SIZE * INPUT_SIZE;
  const batch = new Float32Array(boxes.length * per);
  boxes.forEach((box, i) => batch.set(preprocessFace(image, box, margin), i * per));
  const t2 = performance.now();

  const outputs = await session.run({
    [session.inputNames[0]!]: ortTensor(batch, [boxes.length, 3, INPUT_SIZE, INPUT_SIZE]),
  });
  const t3 = performance.now();

  const logits = outputs[session.outputNames[0]!]!.data as Float32Array;
  const bins = logits.length / boxes.length;

  const faces: ParityFaceResult[] = [];
  for (let i = 0; i < boxes.length; i++) {
    const row = logits.subarray(i * bins, (i + 1) * bins);
    const stats = decodeLogits(row);
    const tensor = batch.subarray(i * per, (i + 1) * per);
    faces.push({
      bbox: boxes[i]!,
      ...buildResult(stats.median, stats.low, stats.high),
      median: stats.median,
      expectation: stats.expectation,
      std: stats.std,
      logits: Array.from(row),
      tensorSha256: await sha256Hex(new Uint8Array(tensor.buffer, tensor.byteOffset, per * 4)),
    });
  }

  return {
    width: canvas.width,
    height: canvas.height,
    cropMargin: margin,
    boxes,
    faces,
    tensorsBase64: toBase64(new Uint8Array(batch.buffer, batch.byteOffset, batch.byteLength)),
    timings: { detectMs: t1 - t0, preprocessMs: t2 - t1, inferMs: t3 - t2 },
  };
}

declare global {
  interface Window {
    __parity: (request: ParityRequest) => Promise<ParityResponse>;
    __parityReady: boolean;
  }
}

window.__parity = run;
window.__parityReady = true;

const status = document.getElementById('status');
if (status) status.textContent = 'ready';
