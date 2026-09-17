/**
 * The original path: hand the frame to the FastAPI server and let it answer.
 *
 * A thin wrapper over `api.ts`, which is untouched. Everything the server
 * engine does, it did before the browser engine existed; this file only
 * re-expresses it behind the shared `Engine` interface so `main.ts` has one
 * code path instead of two.
 */

import { API_BASE, ApiError, estimate, fetchHealth } from './api';
import type { AnalyseOptions, AnalyseResult, Engine, LoadProgress } from './engine';
import { canvasToJpeg } from './camera';
import type { ModelsInfo } from './types';

const JPEG_QUALITY = 0.9;

export class ServerEngine implements Engine {
  readonly kind = 'server' as const;
  readonly privacyNote = '';

  async describe(): Promise<ModelsInfo> {
    const health = await fetchHealth();
    if (health.models) return health.models;
    // A server with a single predictor installed (the test fake, or a build
    // predating the registry) reports no catalog. Present it as one unnamed
    // model rather than pretending to offer a choice.
    return {
      default: 'default',
      available: ['default'],
      models: [
        {
          key: 'default',
          label: health.model,
          question: 'How old this person is',
          explanation: '',
          available: true,
          identity_verified: false,
          stub: health.stub,
          checkpoint: null,
        },
      ],
    };
  }

  async prepare(_model: string | null, onProgress: (p: LoadProgress) => void): Promise<void> {
    onProgress({ label: '', loaded: 0, total: null, done: true });
  }

  async analyse(canvas: HTMLCanvasElement, options: AnalyseOptions): Promise<AnalyseResult> {
    const blob = await canvasToJpeg(canvas, JPEG_QUALITY);
    const result = await estimate(
      blob,
      'frame.jpg',
      options.cropMargin ?? null,
      options.model ?? null,
    );
    return { faces: result.faces, cropMargin: result.cropMargin };
  }
}

export { ApiError, API_BASE };
