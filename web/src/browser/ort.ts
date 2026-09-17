/**
 * onnxruntime-web setup, and model fetching with real progress.
 *
 * SINGLE-THREADED WASM, DELIBERATELY
 * ----------------------------------
 * Multi-threaded WASM needs `SharedArrayBuffer`, which needs cross-origin
 * isolation, which needs `Cross-Origin-Opener-Policy` and
 * `Cross-Origin-Embedder-Policy` response headers. GitHub Pages serves static
 * files with a fixed header set and no way to add either. So threads are not
 * merely unused here, they are unavailable.
 *
 * `numThreads = 1` is therefore set explicitly rather than left to
 * autodetection. Left to itself, onnxruntime-web probes for
 * `SharedArrayBuffer`, and the failure mode when the probe is wrong is a
 * spawned worker that never initialises -- which surfaces as a hang, not an
 * error. Asking for one thread means the question is never asked.
 *
 * `simd` is left on: it needs no special headers and is widely supported. If a
 * browser lacks it, onnxruntime-web falls back on its own.
 */

import * as ort from 'onnxruntime-web/wasm';

/**
 * Where the `.wasm` binaries live, relative to the deployed base path.
 *
 * This is a project site (`/age-estimator/`), so a bare `/ort/` would 404
 * against the user site root. `import.meta.env.BASE_URL` carries Vite's `base`,
 * which the Pages build sets.
 */
const ORT_WASM_BASE = `${import.meta.env.BASE_URL}ort/`;

let configured = false;

export function configureOrt(): void {
  if (configured) return;
  ort.env.wasm.wasmPaths = ORT_WASM_BASE;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';
  configured = true;
}

export async function createSession(modelBytes: Uint8Array): Promise<ort.InferenceSession> {
  configureOrt();
  return ort.InferenceSession.create(modelBytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    // 3 = error. The published YuNet declares fixed output shapes for a
    // 640x640 input (see `stageDynamicYunet` in web/scripts/stage-assets.mjs);
    // running it at any other size makes onnxruntime warn once per output per
    // inference. The warnings are correct and harmless -- the returned shapes
    // are the real ones -- but 12 of them per frame would bury anything worth
    // reading in the console.
    logSeverityLevel: 3,
  });
}

export function ortTensor(data: Float32Array, dims: readonly number[]): ort.Tensor {
  return new ort.Tensor('float32', data, dims as number[]);
}

export interface FetchProgress {
  /** Bytes received so far. */
  loaded: number;
  /** Total bytes, or null when the server sends no usable `Content-Length`. */
  total: number | null;
}

/**
 * Fetch a binary asset, reporting real byte-level progress.
 *
 * The age models are 6.2 MB each. An indeterminate spinner for a
 * multi-megabyte download over a slow link is indistinguishable from a hang,
 * and the user has no way to tell whether waiting is worthwhile. A byte count
 * is available here for free, so it is used.
 *
 * `total` is null rather than guessed when `Content-Length` is missing (which
 * happens under `Content-Encoding`); the caller then shows bytes-so-far instead
 * of inventing a percentage.
 */
export async function fetchWithProgress(
  url: string,
  onProgress: (p: FetchProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const init: RequestInit = { cache: 'force-cache' };
  if (signal) init.signal = signal;
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`Could not download ${url} (HTTP ${response.status}).`);
  }

  const header = response.headers.get('content-length');
  const total = header === null ? null : Number.parseInt(header, 10);
  const knownTotal = total !== null && Number.isFinite(total) && total > 0 ? total : null;

  if (!response.body) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    onProgress({ loaded: buffer.byteLength, total: knownTotal ?? buffer.byteLength });
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  onProgress({ loaded: 0, total: knownTotal });
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      onProgress({ loaded, total: knownTotal });
    }
  }

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}
