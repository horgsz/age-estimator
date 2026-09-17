/**
 * What the UI talks to, so it does not have to know where inference happens.
 *
 * Two implementations exist and both are supported:
 *
 *   server   POST the frame to the FastAPI app (`server/`). This is the local
 *            `make dev` workflow and is unchanged.
 *   browser  Run YuNet and the age model in this tab with onnxruntime-web.
 *            This is what GitHub Pages serves, because Pages cannot run Python.
 *
 * The build picks a default (`VITE_ENGINE`), and `?engine=` overrides it at
 * runtime so either path can be exercised against the same page -- which is how
 * the parity harness compares them.
 */

import type { FaceResult, ModelsInfo } from './types';

export interface AnalyseOptions {
  cropMargin?: number | null;
  model?: string | null;
}

export interface AnalyseResult {
  faces: FaceResult[];
  /** The crop margin actually used, for the advanced panel. */
  cropMargin: number | null;
}

/** Progress of a one-off preparation step, for the loading UI. */
export interface LoadProgress {
  /** What is happening, in words a user can read. */
  label: string;
  loaded: number;
  total: number | null;
  done: boolean;
}

export interface Engine {
  readonly kind: 'server' | 'browser';
  /** Model catalog + measured figures, in `GET /health`'s shape. */
  describe(): Promise<ModelsInfo>;
  /**
   * Make `model` ready to run, reporting progress.
   *
   * The server engine resolves immediately. The browser engine downloads a
   * 6.2 MB model here, so this is where the progress bar comes from.
   */
  prepare(model: string | null, onProgress: (p: LoadProgress) => void): Promise<void>;
  analyse(canvas: HTMLCanvasElement, options: AnalyseOptions): Promise<AnalyseResult>;
  /** Extra copy for this engine, shown under the header. Empty for none. */
  readonly privacyNote: string;
}

export type EngineKind = 'server' | 'browser';

/** Build default, overridable per visit with `?engine=server|browser`. */
export function selectedEngineKind(): EngineKind {
  const requested = new URLSearchParams(window.location.search).get('engine');
  if (requested === 'server' || requested === 'browser') return requested;
  const configured = import.meta.env.VITE_ENGINE;
  return configured === 'browser' ? 'browser' : 'server';
}

export async function createEngine(kind: EngineKind = selectedEngineKind()): Promise<Engine> {
  if (kind === 'browser') {
    // Dynamic so the server build never pulls onnxruntime-web (a few hundred
    // KB of WASM glue) into its bundle for a path it will not take.
    const { BrowserEngine } = await import('./browser/engine');
    return new BrowserEngine();
  }
  const { ServerEngine } = await import('./engine-server');
  return new ServerEngine();
}
