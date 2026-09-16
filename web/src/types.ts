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

export interface HealthResponse {
  status: string;
  model: string;
  stub: boolean;
}
