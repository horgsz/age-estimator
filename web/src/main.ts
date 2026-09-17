import './styles.css';

import { API_BASE, ApiError } from './api';
import { Camera, CameraError } from './camera';
import type { Engine, LoadProgress } from './engine';
import { createEngine, selectedEngineKind } from './engine';
import type { CaveatSpec } from './overlay';
import { clearCanvas, drawToCanvas, renderBoxes, renderFaceList } from './overlay';
import type { FaceResult, ModelsInfo, UserFacing } from './types';

/** Longest side of an analysed frame. Keeps uploads small and inference quick. */
const MAX_ANALYSED_SIDE = 1600;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

const ui = {
  banner: el<HTMLDivElement>('model-banner'),
  video: el<HTMLVideoElement>('preview'),
  cameraMessage: el<HTMLDivElement>('camera-message'),
  cameraMessageText: el<HTMLParagraphElement>('camera-message-text'),
  enableCamera: el<HTMLButtonElement>('enable-camera'),
  deviceSelect: el<HTMLSelectElement>('device-select'),
  mirrorToggle: el<HTMLInputElement>('mirror-toggle'),
  capture: el<HTMLButtonElement>('capture'),
  chooseFile: el<HTMLButtonElement>('choose-file'),
  fileInput: el<HTMLInputElement>('file-input'),
  dropzone: el<HTMLDivElement>('dropzone'),
  status: el<HTMLDivElement>('status'),
  resultStage: el<HTMLDivElement>('result-stage'),
  resultCanvas: el<HTMLCanvasElement>('result-canvas'),
  overlay: el<HTMLDivElement>('overlay'),
  faceList: el<HTMLUListElement>('face-list'),
  cropMargin: el<HTMLInputElement>('crop-margin'),
  reanalyse: el<HTMLButtonElement>('reanalyse'),
  tuningNote: el<HTMLParagraphElement>('tuning-note'),
  tuning: el<HTMLDetailsElement>('tuning'),
  modelPicker: el<HTMLDivElement>('model-picker'),
  modelNote: el<HTMLParagraphElement>('model-note'),
  modelGroup: el<HTMLFieldSetElement>('model-picker-group'),
  appSub: el<HTMLParagraphElement>('app-sub'),
  appWarn: el<HTMLParagraphElement>('app-warn'),
  privacyNote: el<HTMLParagraphElement>('privacy-note'),
  loader: el<HTMLDivElement>('load-progress'),
  loadLabel: el<HTMLSpanElement>('load-label'),
  loadDetail: el<HTMLSpanElement>('load-detail'),
  loadBar: el<HTMLDivElement>('load-bar'),
};

/**
 * Where inference happens.
 *
 * `null` until `boot()` resolves it. The two engines -- the FastAPI server and
 * the in-browser onnxruntime-web path -- are behind one interface precisely so
 * nothing below this line has to know which one is live.
 */
let engine: Engine | null = null;
const engineKind = selectedEngineKind();

// Opening the advanced panel also reveals the per-face debug details (the
// low–high range and the unrounded estimate). Keying this off a body class
// rather than touching each card means it survives every re-render.
function syncDebugDetails(): void {
  document.body.classList.toggle('debug', ui.tuning.open);
}
ui.tuning.addEventListener('toggle', syncDebugDetails);
syncDebugDetails();

const camera = new Camera(ui.video);
let busy = false;

/**
 * The exact pixels of the last analysed frame.
 *
 * Kept so the same frame can be re-sent at a different crop margin. A/B'ing
 * margins is only meaningful if the input is byte-identical between runs —
 * re-capturing from the webcam would change the pose and the lighting too.
 */
let lastCanvas: HTMLCanvasElement | null = null;

/**
 * Which model answers the next request, and what /health said about them.
 *
 * `null` means "whatever the server defaults to" — the UI does not assume it
 * knows, because the default is server-configured.
 */
let selectedModel: string | null = null;
let models: ModelsInfo | null = null;

/**
 * Every derived figure shown to the user is a property of specific weights.
 *
 * The typical-error number, the age-gating percentage and the caveat band all
 * differ between the two models, and the caveat in particular is true of one
 * and FALSE of the other. So none of them are constants here: they are served
 * per model by /health, keyed by the checkpoint's own content digest, and read
 * back out for whichever model is active.
 *
 * An unmeasured checkpoint supplies none of them. The copy then falls back to
 * unquantified wording rather than borrowing another model's numbers -- the
 * safe direction, and the same rule /health applies to accuracy.
 */
function activeUserFacing(): UserFacing | null {
  const active = models?.models.find((m) => m.key === selectedModel);
  return active?.checkpoint?.user_facing ?? null;
}

function activeCaveat(): CaveatSpec | null {
  return activeUserFacing()?.caveat ?? null;
}

function syncAccuracyCopy(): void {
  const uf = activeUserFacing();

  ui.appSub.textContent = uf
    ? 'Point a camera at a face, or drop in a photo. Estimates are rough: ' +
      `they are off by about ${Math.round(uf.typical_error_years)} years on ` +
      `average ${uf.typical_error_basis}, and individual faces can be much ` +
      'further out.'
    : 'Point a camera at a face, or drop in a photo. Estimates are rough, and ' +
      'individual faces can be a long way out.';

  // The age-verification warning is never suppressed, only de-quantified. It
  // is a safety disclaimer, and the model with no measured figure is not
  // thereby safer -- it is merely unmeasured.
  const pct = uf ? `about ${Math.round(uf.gating_under18_shown_adult_pct)}%` : 'a large share';
  ui.appWarn.innerHTML =
    '<strong>Not usable for age verification.</strong> In testing, ' +
    `<strong>${pct} of people under 18 were shown as 18 or over</strong>. ` +
    'Do not use this to decide whether someone meets an age limit.';
}

function renderModelPicker(info: ModelsInfo): void {
  models = info;
  selectedModel = selectedModel ?? info.default;
  ui.modelPicker.replaceChildren();

  // One model (or none) is not a choice; hide the control rather than showing
  // a radio group with a single option.
  if (info.models.filter((m) => m.available).length < 2) {
    ui.modelGroup.hidden = true;
    syncAccuracyCopy();
    return;
  }
  ui.modelGroup.hidden = false;

  for (const model of info.models) {
    const label = document.createElement('label');
    label.className = 'control control--check';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'model';
    input.value = model.key;
    input.checked = model.key === selectedModel;
    input.disabled = !model.available;
    input.addEventListener('change', () => {
      if (!input.checked) return;
      selectedModel = model.key;
      syncAccuracyCopy();
      showModelNote();
      // Re-analyse the retained pixels rather than re-capturing: comparing two
      // models is only meaningful on byte-identical input.
      if (lastCanvas) void analyseCanvas(lastCanvas);
    });
    const text = document.createElement('span');
    text.textContent = model.available
      ? model.question
      : `${model.question} (unavailable)`;
    label.append(input, text);
    if (!model.available && model.unavailable_reason) {
      label.title = model.unavailable_reason;
    }
    ui.modelPicker.append(label);
  }
  showModelNote();
  syncAccuracyCopy();
}

function showModelNote(): void {
  const active = models?.models.find((m) => m.key === selectedModel);
  ui.modelNote.textContent = active
    ? `${active.label}: ${active.explanation} The two models are not ranked — ` +
      'they answer different questions, so their error figures are not comparable.'
    : '';
}

// ---------------------------------------------------------------------------
// status / UI states
// ---------------------------------------------------------------------------

type StatusKind = 'idle' | 'busy' | 'ok' | 'empty' | 'error';

function setStatus(kind: StatusKind, message: string): void {
  ui.status.className = `status status--${kind}`;
  ui.status.textContent = message;
}

function setCameraMessage(text: string | null, buttonLabel = 'Enable camera'): void {
  if (text === null) {
    ui.cameraMessage.hidden = true;
    return;
  }
  ui.cameraMessage.hidden = false;
  ui.cameraMessageText.textContent = text;
  ui.enableCamera.textContent = buttonLabel;
}

function setBusy(value: boolean): void {
  busy = value;
  ui.capture.disabled = value || !camera.isActive;
  ui.chooseFile.disabled = value;
  ui.deviceSelect.disabled = value || ui.deviceSelect.options.length < 2;
  ui.cropMargin.disabled = value;
  ui.reanalyse.disabled = value || lastCanvas === null;
  ui.dropzone.classList.toggle('dropzone--disabled', value);
}

/** The margin box, as a number — or null for "let the server decide". */
function requestedMargin(): number | null {
  const raw = ui.cropMargin.value.trim();
  if (raw === '') return null;
  const value = Number.parseFloat(raw);
  return Number.isFinite(value) ? value : null;
}

/** Render a margin unambiguously: 0 becomes "0.0", not a bare "0" before a full stop. */
function formatMargin(value: number): string {
  return Number.isInteger(value) ? value.toFixed(1) : String(value);
}

function showMarginNote(used: number | null): void {
  if (used === null) {
    ui.tuningNote.hidden = true;
    return;
  }
  ui.tuningNote.hidden = false;
  ui.tuningNote.textContent =
    requestedMargin() === null
      ? `Default margin: ${formatMargin(used)}`
      : `Analysed with margin ${formatMargin(used)}`;
}

// ---------------------------------------------------------------------------
// model download progress
// ---------------------------------------------------------------------------

function formatBytes(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.round(bytes / 1024)} KB`;
}

/**
 * Show real progress for the model fetch.
 *
 * The browser engine downloads 6.2 MB the first time a model is used, then the
 * HTTP cache serves it instantly forever after. A percentage is available from
 * `Content-Length`, so it is shown; when it is not (a proxy stripping the
 * header, say), the bar goes indeterminate and the byte count is shown instead
 * of a made-up percentage.
 */
function showLoadProgress(p: LoadProgress): void {
  if (p.done || !p.label) {
    hideLoadProgress();
    return;
  }
  ui.loader.hidden = false;
  ui.loadLabel.textContent = p.label;
  if (p.total && p.total > 0) {
    const pct = Math.min(100, Math.round((p.loaded / p.total) * 100));
    ui.loadBar.style.width = `${pct}%`;
    ui.loadBar.classList.remove('loader__bar--indeterminate');
    ui.loadDetail.textContent = `${formatBytes(p.loaded)} of ${formatBytes(p.total)}`;
  } else {
    ui.loadBar.style.width = '100%';
    ui.loadBar.classList.add('loader__bar--indeterminate');
    ui.loadDetail.textContent = p.loaded > 0 ? formatBytes(p.loaded) : '';
  }
}

function hideLoadProgress(): void {
  ui.loader.hidden = true;
  ui.loadBar.classList.remove('loader__bar--indeterminate');
  ui.loadBar.style.width = '0%';
}

function clearResult(): void {
  ui.overlay.replaceChildren();
  ui.faceList.replaceChildren();
  clearCanvas(ui.resultCanvas);
  ui.resultStage.classList.add('stage--empty');
}

function showResult(faces: FaceResult[], width: number, height: number): void {
  ui.resultStage.classList.remove('stage--empty');
  renderBoxes(ui.overlay, faces, width, height, activeCaveat());
  renderFaceList(ui.faceList, faces, activeCaveat());

  if (faces.length === 0) {
    setStatus(
      'empty',
      'No face found in that frame. Try better lighting, move closer, or face the camera more directly.',
    );
  } else {
    setStatus(
      'ok',
      faces.length === 1
        ? 'Found 1 face. The range is the model’s uncertainty, not a rounding.'
        : `Found ${faces.length} faces. Each range is that face’s uncertainty.`,
    );
  }
}

// ---------------------------------------------------------------------------
// analysis
// ---------------------------------------------------------------------------

/**
 * Analyse a canvas.
 *
 * The canvas is the single coordinate space: we re-encode exactly these pixels
 * as the JPEG we upload, so the bboxes that come back line up with what we draw
 * without any mirroring or scaling correction.
 */
async function analyseCanvas(canvas: HTMLCanvasElement): Promise<void> {
  if (busy) return;
  if (!engine) return;
  lastCanvas = canvas;
  setBusy(true);
  clearResult();
  setStatus('busy', 'Analysing…');

  try {
    drawToCanvas(ui.resultCanvas, canvas, canvas.width, canvas.height);

    // The model download happens here, not at boot: most visitors only ever use
    // the default model, so fetching both up front would cost everyone 12.4 MB
    // to use 6.2 MB of it.
    await engine.prepare(selectedModel, showLoadProgress);

    const { faces, cropMargin } = await engine.analyse(canvas, {
      cropMargin: requestedMargin(),
      model: selectedModel,
    });
    showResult(faces, canvas.width, canvas.height);
    showMarginNote(cropMargin);
  } catch (err) {
    clearResult();
    if (err instanceof ApiError) {
      setStatus('error', err.message);
    } else {
      setStatus('error', err instanceof Error ? err.message : 'Something went wrong.');
    }
  } finally {
    hideLoadProgress();
    setBusy(false);
  }
}

function scaledSize(width: number, height: number): [number, number] {
  const scale = Math.min(1, MAX_ANALYSED_SIDE / Math.max(width, height));
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

/**
 * Decode an uploaded file onto a canvas, honouring EXIF orientation.
 *
 * This matters: browsers rotate EXIF-tagged photos when displaying them but
 * OpenCV's `imdecode` on the server does not. Baking the orientation in here
 * and re-encoding keeps the uploaded pixels identical to the displayed ones.
 */
async function fileToCanvas(file: File): Promise<HTMLCanvasElement> {
  const canvas = document.createElement('canvas');

  if ('createImageBitmap' in window) {
    const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    const [w, h] = scaledSize(bitmap.width, bitmap.height);
    drawToCanvas(canvas, bitmap, w, h);
    bitmap.close();
    return canvas;
  }

  const url = URL.createObjectURL(file);
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('That file could not be decoded as an image.'));
      img.src = url;
    });
    const [w, h] = scaledSize(image.naturalWidth, image.naturalHeight);
    drawToCanvas(canvas, image, w, h);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function analyseFile(file: File): Promise<void> {
  if (busy) return;
  if (!file.type.startsWith('image/')) {
    setStatus('error', `“${file.name}” is not an image.`);
    return;
  }
  try {
    const canvas = await fileToCanvas(file);
    await analyseCanvas(canvas);
  } catch (err) {
    setStatus('error', err instanceof Error ? err.message : 'That file could not be read.');
  }
}

// ---------------------------------------------------------------------------
// camera wiring
// ---------------------------------------------------------------------------

async function refreshDevices(): Promise<void> {
  const devices = await camera.listDevices();
  const previous = ui.deviceSelect.value;

  ui.deviceSelect.replaceChildren();
  if (devices.length === 0) {
    const option = new Option('No cameras found', '');
    ui.deviceSelect.add(option);
    ui.deviceSelect.disabled = true;
    return;
  }

  devices.forEach((device, index) => {
    ui.deviceSelect.add(new Option(device.label || `Camera ${index + 1}`, device.deviceId));
  });

  if (previous && devices.some((d) => d.deviceId === previous)) {
    ui.deviceSelect.value = previous;
  }
  // Only useful when there is an actual choice to make.
  ui.deviceSelect.disabled = devices.length < 2 || busy;
}

async function startCamera(deviceId?: string): Promise<void> {
  setCameraMessage('Starting camera…', 'Enable camera');
  ui.enableCamera.disabled = true;
  try {
    await camera.start(deviceId);
    setCameraMessage(null);
    ui.capture.disabled = busy;
    await refreshDevices();
    if (!deviceId) {
      const track = ui.video.srcObject instanceof MediaStream
        ? ui.video.srcObject.getVideoTracks()[0]
        : undefined;
      const active = track?.getSettings().deviceId;
      if (active) ui.deviceSelect.value = active;
    }
  } catch (err) {
    ui.capture.disabled = true;
    const message =
      err instanceof CameraError ? err.message : 'The camera could not be started.';
    const retryLabel =
      err instanceof CameraError && err.kind === 'denied' ? 'Try again' : 'Enable camera';
    setCameraMessage(message, retryLabel);
  } finally {
    ui.enableCamera.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------

ui.enableCamera.addEventListener('click', () => {
  void startCamera(ui.deviceSelect.value || undefined);
});

ui.deviceSelect.addEventListener('change', () => {
  if (camera.isActive) void startCamera(ui.deviceSelect.value || undefined);
});

ui.mirrorToggle.addEventListener('change', () => {
  camera.setMirrored(ui.mirrorToggle.checked);
});

ui.capture.addEventListener('click', () => {
  try {
    void analyseCanvas(camera.captureFrame());
  } catch (err) {
    setStatus('error', err instanceof Error ? err.message : 'Could not capture a frame.');
  }
});

ui.chooseFile.addEventListener('click', () => ui.fileInput.click());

ui.reanalyse.addEventListener('click', () => {
  if (lastCanvas) void analyseCanvas(lastCanvas);
});

ui.cropMargin.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && lastCanvas) {
    event.preventDefault();
    void analyseCanvas(lastCanvas);
  }
});

ui.fileInput.addEventListener('change', () => {
  const file = ui.fileInput.files?.[0];
  if (file) void analyseFile(file);
  ui.fileInput.value = '';
});

ui.dropzone.addEventListener('click', () => ui.fileInput.click());
ui.dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    ui.fileInput.click();
  }
});

for (const type of ['dragenter', 'dragover'] as const) {
  ui.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    ui.dropzone.classList.add('dropzone--over');
  });
}
for (const type of ['dragleave', 'dragend'] as const) {
  ui.dropzone.addEventListener(type, () => ui.dropzone.classList.remove('dropzone--over'));
}
ui.dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  ui.dropzone.classList.remove('dropzone--over');
  const file = event.dataTransfer?.files?.[0];
  if (file) void analyseFile(file);
});

// Dropping anywhere else must not navigate away from the app.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

navigator.mediaDevices?.addEventListener?.('devicechange', () => {
  void refreshDevices();
});

window.addEventListener('pagehide', () => camera.stop());

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

async function showModelBanner(): Promise<void> {
  if (!engine) return;
  try {
    const models = await engine.describe();
    const active = models.models.find((m) => m.key === models.default);
    if (active?.stub) {
      ui.banner.className = 'banner banner--warn';
      ui.banner.textContent =
        'Running the STUB predictor — ages are fake placeholder values, not a real estimate.';
    } else {
      ui.banner.className = 'banner banner--ok';
      ui.banner.textContent =
        engine.kind === 'browser'
          ? 'Runs in your browser. The model is downloaded once and cached.'
          : `Model loaded: ${active?.label ?? 'unknown'}`;
    }
    renderModelPicker(models);
  } catch (err) {
    ui.banner.className = 'banner banner--error';
    ui.banner.textContent =
      engine.kind === 'browser'
        ? err instanceof Error
          ? err.message
          : 'Could not load the model list.'
        : `Cannot reach the API at ${API_BASE}. Start it with: make api`;
    ui.modelGroup.hidden = true;
  }
}

async function boot(): Promise<void> {
  camera.setMirrored(ui.mirrorToggle.checked);
  clearResult();
  setStatus('idle', 'Nothing analysed yet.');

  if (!Camera.isSupported) {
    setCameraMessage(
      window.isSecureContext
        ? 'This browser does not support camera capture. You can still upload a photo.'
        : 'Camera access needs a secure context — open the app on http://localhost or over https. You can still upload a photo.',
    );
    ui.enableCamera.disabled = true;
  } else {
    setCameraMessage('Camera is off.');
  }

  void refreshDevices();

  engine = await createEngine(engineKind);
  // A genuine privacy property of the browser build, and only of it: there is
  // no upload because there is nowhere to upload to. Stated by the engine
  // rather than by this file, so the server build cannot accidentally claim it.
  if (engine.privacyNote) {
    ui.privacyNote.hidden = false;
    ui.privacyNote.textContent = engine.privacyNote;
  }
  await showModelBanner();
}

void boot();
