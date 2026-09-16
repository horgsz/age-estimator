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
      if (Array.isArray(detail)) return 'The server rejected the upload.';
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

/** POST an image to `/estimate`. Zero faces is a success, not an error. */
export async function estimate(blob: Blob, filename = 'frame.jpg'): Promise<EstimateResponse> {
  const form = new FormData();
  form.append('image', blob, filename);

  const response = await request('/estimate', { method: 'POST', body: form });
  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status);
  }

  const body = (await response.json()) as EstimateResponse;
  if (!Array.isArray(body?.faces)) {
    throw new ApiError('The server returned an unexpected response.');
  }
  return body;
}
