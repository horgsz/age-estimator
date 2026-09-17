import type { EstimateResponse, HealthResponse } from './types';

/** Base URL of the FastAPI server. Override with `VITE_API_BASE`. */
export const API_BASE: string = (
  import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
).replace(/\/$/, '');

const REQUEST_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === 'string') return detail;
      if (Array.isArray(detail)) {
        // FastAPI validation errors: [{ loc: [...], msg: "..." }]. Naming the
        // offending field matters most for crop_margin, which the user can type.
        const parts = detail
          .map((item) => {
            if (!item || typeof item !== 'object') return null;
            const { loc, msg } = item as { loc?: unknown; msg?: unknown };
            if (typeof msg !== 'string') return null;
            const field = Array.isArray(loc) ? loc[loc.length - 1] : undefined;
            return typeof field === 'string' && field !== 'body'
              ? `${field}: ${msg}`
              : msg;
          })
          .filter((part): part is string => part !== null);
        if (parts.length > 0) return parts.join('; ');
        return 'The server rejected the upload.';
      }
    }
  } catch {
    /* non-JSON error body */
  }
  return `Request failed (HTTP ${response.status}).`;
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError('The server took too long to respond.');
    }
    throw new ApiError(
      `Could not reach the API at ${API_BASE}. Is the server running?`,
    );
  } finally {
    window.clearTimeout(timer);
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await request('/health');
  if (!response.ok) throw new ApiError(await readErrorDetail(response), response.status);
  return (await response.json()) as HealthResponse;
}

/** Result of an `/estimate` call, plus the crop margin the server actually used. */
export interface EstimateResult extends EstimateResponse {
  /** Parsed from the `X-Crop-Margin` response header; null if absent. */
  cropMargin: number | null;
  /** Parsed from the `X-Model` response header; null if absent. Echoed so a
   *  displayed number can always be traced to the model that produced it. */
  model: string | null;
}

/**
 * POST an image to `/estimate`. Zero faces is a success, not an error.
 *
 * `cropMargin` overrides the server's CROP_MARGIN for this request only, so
 * margins can be A/B'd against real webcam photos without a server restart.
 */
export async function estimate(
  blob: Blob,
  filename = 'frame.jpg',
  cropMargin?: number | null,
  model?: string | null,
): Promise<EstimateResult> {
  const form = new FormData();
  form.append('image', blob, filename);
  if (cropMargin != null && Number.isFinite(cropMargin)) {
    form.append('crop_margin', String(cropMargin));
  }
  if (model) form.append('model', model);

  const response = await request('/estimate', { method: 'POST', body: form });
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status);
  }

  const body = (await response.json()) as EstimateResponse;
  if (!Array.isArray(body?.faces)) {
    throw new ApiError('The server returned an unexpected response.');
  }

  const header = response.headers.get('X-Crop-Margin');
  const parsed = header === null ? Number.NaN : Number.parseFloat(header);
  return {
    ...body,
    cropMargin: Number.isFinite(parsed) ? parsed : null,
    model: response.headers.get('X-Model'),
  };
}
