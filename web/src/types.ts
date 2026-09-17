/** Shapes returned by the age-estimation API. */

/** `[x, y, w, h]`, integer pixels in the uploaded image's coordinate space. */
export type BBox = [number, number, number, number];

export interface FaceResult {
  bbox: BBox;
  /** Point estimate, years. */
  age: number;
  /** Lower bound of the uncertainty interval (`age - std`), years. */
  low: number;
  /** Upper bound of the uncertainty interval (`age + std`), years. */
  high: number;
  /** 0..1, higher means a tighter predicted distribution. */
  confidence: number;
}

export interface EstimateResponse {
  faces: FaceResult[];
}

/** One selectable model, as described by `GET /health`. */
export interface ModelInfo {
  key: string;
  label: string;
  /** What this model predicts, in plain words. The two models answer
   *  different questions; this is the distinction the UI must not blur. */
  question: string;
  explanation: string;
  available: boolean;
  identity_verified: boolean;
  unavailable_reason?: string;
  stub: boolean | null;
}

export interface ModelsInfo {
  default: string;
  available: string[];
  models: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  model: string;
  stub: boolean;
  models?: ModelsInfo;
}
