/**
 * The whole inference path, in the browser.
 *
 * Canvas pixels -> YuNet -> the crop from `server/preprocessing.py` -> the age
 * model -> the median decode from `server/predictor.py`. Nothing leaves the
 * tab: there is no upload, and on GitHub Pages there is nowhere to upload to.
 *
 * MODEL LOADING IS LAZY AND PER MODEL
 * -----------------------------------
 * The two age models are 6.2 MB each. Fetching both up front would cost every
 * visitor 12.4 MB to use one of them, and most visitors never touch the toggle.
 * So each is fetched the first time it is actually selected, and the detector
 * (230 KB) is fetched once alongside the first of them.
 *
 * The browser's HTTP cache makes the second visit free, which is why the
 * fetches use `cache: 'force-cache'` and why the files are served under
 * content-independent names -- a build that renamed them per deploy would throw
 * that away.
 */

import type { AnalyseOptions, AnalyseResult, Engine, LoadProgress } from '../engine';
import type { FaceResult, ModelsInfo, StaticModelInfo, StaticRegistry } from '../types';
import { decode } from './decode';
import { createSession, fetchWithProgress, ortTensor } from './ort';
import type { BBox } from './preprocess';
import { INPUT_SIZE, preprocessFace } from './preprocess';
import type { Mat8U } from './cv-resize';
import { DEFAULT_DETECTOR_OPTIONS, YuNetDetector } from './yunet';
import type { InferenceSession } from 'onnxruntime-web/wasm';

const ASSET_BASE = `${import.meta.env.BASE_URL}models/`;
// The shape-metadata-corrected copy; see `stageDynamicYunet` in
// `web/scripts/stage-assets.mjs` for why the published file cannot be used
// directly and why the weights are nonetheless identical.
const YUNET_ASSET = 'face_detection_yunet_2023mar.dynamic.onnx';

/**
 * Canvas RGBA -> the BGR byte layout OpenCV (and therefore every port in this
 * directory) works in.
 *
 * Dropping alpha rather than compositing is correct here: the frames come from
 * a video element or a decoded image drawn onto an opaque canvas, so alpha is
 * always 255. It is also what `cv2.imdecode(..., IMREAD_COLOR)` does.
 */
function canvasToBgr(canvas: HTMLCanvasElement): Mat8U {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Could not read pixels from the canvas.');
  const { width, height } = canvas;
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const data = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
    data[i] = rgba[j + 2]!;
    data[i + 1] = rgba[j + 1]!;
    data[i + 2] = rgba[j]!;
  }
  return { data, width, height, channels: 3 };
}

interface LoadedModel {
  session: InferenceSession;
  info: StaticModelInfo;
}

export class BrowserEngine implements Engine {
  readonly kind = 'browser' as const;
  readonly privacyNote =
    'Everything runs in your browser. The photo and the video never leave this ' +
    'device — there is no server to send them to.';

  private registry: StaticRegistry | null = null;
  private detector: YuNetDetector | null = null;
  private readonly models = new Map<string, LoadedModel>();
  private readonly inFlight = new Map<string, Promise<LoadedModel>>();

  async describe(): Promise<ModelsInfo> {
    const registry = await this.loadRegistry();
    return {
      default: registry.default,
      available: registry.available,
      models: registry.models.map((m) => ({
        key: m.key,
        label: m.label,
        question: m.question,
        explanation: m.explanation,
        available: m.available,
        identity_verified: m.identity_verified,
        ...(m.unavailable_reason === undefined
          ? {}
          : { unavailable_reason: m.unavailable_reason }),
        stub: false,
        checkpoint: m.checkpoint,
      })),
    };
  }

  /** The static stand-in for `GET /health`, plus the figures it carries. */
  async loadRegistry(): Promise<StaticRegistry> {
    if (this.registry) return this.registry;
    const response = await fetch(`${ASSET_BASE}models.json`, { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(
        `Could not load the model list (HTTP ${response.status}). The site may be ` +
          'mid-deploy; try again in a moment.',
      );
    }
    this.registry = (await response.json()) as StaticRegistry;
    return this.registry;
  }

  async prepare(model: string | null, onProgress: (p: LoadProgress) => void): Promise<void> {
    await this.load(model, onProgress);
  }

  private async load(
    model: string | null,
    onProgress: (p: LoadProgress) => void,
  ): Promise<LoadedModel> {
    const registry = await this.loadRegistry();
    const key = model ?? registry.default;
    const cached = this.models.get(key);
    if (cached) {
      onProgress({ label: '', loaded: 0, total: null, done: true });
      return cached;
    }

    // Two rapid model-toggle clicks must not start two 6.2 MB downloads.
    const pending = this.inFlight.get(key);
    if (pending) return pending;

    const started = this.loadUncached(key, registry, onProgress).finally(() => {
      this.inFlight.delete(key);
    });
    this.inFlight.set(key, started);
    return started;
  }

  private async loadUncached(
    key: string,
    registry: StaticRegistry,
    onProgress: (p: LoadProgress) => void,
  ): Promise<LoadedModel> {
    const info = registry.models.find((m) => m.key === key);
    if (!info) throw new Error(`Unknown model "${key}".`);
    if (!info.available || !info.asset) {
      throw new Error(info.unavailable_reason ?? `Model "${key}" is not available.`);
    }

    if (!this.detector) {
      onProgress({ label: 'Loading the face detector', loaded: 0, total: null, done: false });
      const bytes = await fetchWithProgress(`${ASSET_BASE}${YUNET_ASSET}`, (p) =>
        onProgress({
          label: 'Loading the face detector',
          loaded: p.loaded,
          total: p.total,
          done: false,
        }),
      );
      this.detector = await YuNetDetector.create(bytes, {
        ...DEFAULT_DETECTOR_OPTIONS,
        scoreThreshold: registry.detector.score_threshold,
        nmsThreshold: registry.detector.nms_threshold,
        topK: registry.detector.top_k,
      });
    }

    const label = `Loading the ${info.label.toLowerCase()} model`;
    onProgress({ label, loaded: 0, total: info.bytes, done: false });
    const modelBytes = await fetchWithProgress(`${ASSET_BASE}${info.asset}`, (p) =>
      onProgress({ label, loaded: p.loaded, total: p.total ?? info.bytes, done: false }),
    );

    onProgress({ label: 'Starting the model', loaded: 1, total: 1, done: false });
    const session = await createSession(modelBytes);
    const loaded: LoadedModel = { session, info };
    this.models.set(key, loaded);
    onProgress({ label: '', loaded: 1, total: 1, done: true });
    return loaded;
  }

  async analyse(canvas: HTMLCanvasElement, options: AnalyseOptions): Promise<AnalyseResult> {
    const registry = await this.loadRegistry();
    const { session } = await this.load(options.model ?? null, () => {});
    if (!this.detector) throw new Error('The face detector is not loaded.');

    const margin = options.cropMargin ?? registry.preprocessing.crop_margin;
    const image = canvasToBgr(canvas);
    const boxes = await this.detector.detect(image);
    if (boxes.length === 0) return { faces: [], cropMargin: margin };

    const faces = await this.estimateBoxes(session, image, boxes, margin);
    return { faces, cropMargin: margin };
  }

  /**
   * The crop + model half, split out for the same reason
   * `AgePredictor.predict_boxes` is: the parity harness needs to feed in boxes
   * it chose rather than boxes the detector found, so that a preprocessing
   * comparison is not also a detection comparison.
   */
  async estimateBoxes(
    session: InferenceSession,
    image: Mat8U,
    boxes: BBox[],
    margin: number,
  ): Promise<FaceResult[]> {
    const per = 3 * INPUT_SIZE * INPUT_SIZE;
    const batch = new Float32Array(boxes.length * per);
    boxes.forEach((box, i) => {
      batch.set(preprocessFace(image, box, margin), i * per);
    });

    const inputName = session.inputNames[0]!;
    const outputs = await session.run({
      [inputName]: ortTensor(batch, [boxes.length, 3, INPUT_SIZE, INPUT_SIZE]),
    });
    const logits = outputs[session.outputNames[0]!]!.data as Float32Array;
    const bins = logits.length / boxes.length;

    return boxes.map((box, i) => {
      const decoded = decode(logits.subarray(i * bins, (i + 1) * bins));
      return { bbox: box, ...decoded };
    });
  }

  /** The loaded session for `key`, for the parity harness. */
  async sessionFor(key: string | null): Promise<InferenceSession> {
    return (await this.load(key, () => {})).session;
  }

  async detectorFor(key: string | null): Promise<YuNetDetector> {
    await this.load(key, () => {});
    if (!this.detector) throw new Error('The face detector is not loaded.');
    return this.detector;
  }

  /** Canvas -> BGR, exposed so the harness shares one conversion. */
  static toBgr(canvas: HTMLCanvasElement): Mat8U {
    return canvasToBgr(canvas);
  }
}
