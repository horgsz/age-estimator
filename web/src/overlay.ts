import type { FaceResult } from './types';

/**
 * Rendering of the analysed frame and its bounding boxes.
 *
 * Boxes arrive in the pixel coordinate space of the image that was uploaded,
 * and that same image is what we draw here (see the mirroring note in
 * `camera.ts`), so positioning is a straight pixel -> percentage conversion
 * with no flipping. Using percentages means the overlay tracks the canvas at
 * any CSS display size.
 */

export function drawToCanvas(
  canvas: HTMLCanvasElement,
  source: CanvasImageSource,
  width: number,
  height: number,
): void {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get a 2D canvas context.');
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);
}

export function clearCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  ctx?.clearRect(0, 0, canvas.width, canvas.height);
}

function formatYears(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/**
 * The headline figure, rounded to a whole year.
 *
 * A decimal on a single number ("31.4") implies a precision the model does not
 * have — the predicted distribution's standard deviation averages ~12.6 years.
 * The full-precision value stays in the API response and is shown in the
 * advanced panel.
 */
export function formatAge(age: number): string {
  return String(Math.round(age));
}

export function formatRange(face: FaceResult): string {
  return `${formatYears(face.low)}–${formatYears(face.high)}`;
}

/**
 * Ages where the displayed number is least reliable, and the app says so.
 *
 * Thresholds and wording are conditioned on the *predicted* age, because that
 * is the only thing the UI knows. Binning by true age answers a question the UI
 * cannot ask.
 *
 * These numbers were re-derived after the decode changed from soft-expectation
 * to the distribution median, and the change inverted the previous conclusion.
 * Under the old decode a label-smoothing pedestal dragged every estimate toward
 * 50, so the extremes were badly offset (a real 85-year-old displayed as ~68)
 * and both tails needed a warning. The median ignores that pedestal, and the
 * offsets largely vanished.
 *
 * Measured end to end over 1,184 UTKFace test images at the shipped margin:
 *
 *   shown     n     MAE    bias
 *   0–5     109    0.98   −0.71
 *   5–10     49    2.33   −0.82
 *   10–12    13    3.54   −2.31
 *   20–30   386    3.52   −0.49
 *   30–40   226    5.90   −0.02
 *   40–50    85    6.61   +2.71
 *   50–60   138    7.18   +2.09
 *   60–70    81    7.98   +0.80
 *   70–75    16    7.81   +2.56
 *   75+      34    4.82   +0.59
 *
 * Two consequences, both of which reverse earlier behaviour:
 *
 * 1. The young caveat is GONE. Ages shown under 12 are now the most accurate
 *    region the model has (MAE 1.76, bias −0.99) and the output reaches down to
 *    1, so there is no floor to warn about. The old "likely younger than shown"
 *    note would now be both unnecessary and pointing the wrong way.
 *
 * 2. The old-age caveat is GONE TOO, and this one is subtle. Binned by *true*
 *    age the top still compresses (bias −7.8 at 80+), which is what an offline
 *    eval sees and it is tempting to warn about. But binned by *displayed* age,
 *    everything shown at 80+ has bias +1.10 — a ">= 80, reads low" rule would
 *    fire on a population it is not actually wrong about. There is no threshold
 *    where a directional old-age warning is supportable.
 *
 * What is left is a genuine precision story in mid-to-late adulthood: 40–75 is
 * 27% of cases at MAE 7.27, against 3.83 everywhere else. That is nearly a 2x
 * difference and it is worth telling the user about, so it is the only caveat.
 */
const MID_LOW = 40;
const MID_HIGH = 75;

export interface TailCaveat {
  kind: 'mid';
  short: string;
  long: string;
}

export function tailCaveat(age: number): TailCaveat | null {
  if (age >= MID_LOW && age < MID_HIGH) {
    return {
      kind: 'mid',
      short: 'less precise in this range',
      long:
        'Middle and later adulthood is this model’s weakest range: estimates ' +
        'here are typically off by about 7 years, against roughly 4 elsewhere, ' +
        'and they tend to read slightly old. Treat the number as a broad ' +
        'bracket rather than a reading.',
    };
  }
  return null;
}

/** Hue from red (uncertain) to green (confident). */
function confidenceColor(confidence: number): string {
  const hue = Math.round(Math.max(0, Math.min(1, confidence)) * 120);
  return `hsl(${hue} 85% 55%)`;
}

export function renderBoxes(
  overlay: HTMLElement,
  faces: FaceResult[],
  imageWidth: number,
  imageHeight: number,
): void {
  overlay.replaceChildren();
  if (!imageWidth || !imageHeight) return;

  faces.forEach((face, index) => {
    const [x, y, w, h] = face.bbox;
    const color = confidenceColor(face.confidence);

    const box = document.createElement('div');
    box.className = 'face-box';
    box.style.left = `${(x / imageWidth) * 100}%`;
    box.style.top = `${(y / imageHeight) * 100}%`;
    box.style.width = `${(w / imageWidth) * 100}%`;
    box.style.height = `${(h / imageHeight) * 100}%`;
    box.style.borderColor = color;
    // Confidence is also legible at a glance from how solid the box looks.
    box.style.opacity = String(0.45 + 0.55 * face.confidence);

    const label = document.createElement('div');
    label.className = 'face-box__label';
    // Boxes near the top of the frame get their label underneath instead.
    if (y / imageHeight < 0.14) label.classList.add('face-box__label--below');
    label.style.borderColor = color;

    const age = document.createElement('span');
    age.className = 'face-box__age';
    age.textContent = formatAge(face.age);

    const unit = document.createElement('span');
    unit.className = 'face-box__unit';
    unit.textContent = 'Age Estimate';

    const bar = document.createElement('span');
    bar.className = 'face-box__bar';
    const fill = document.createElement('span');
    fill.className = 'face-box__bar-fill';
    fill.style.width = `${Math.round(face.confidence * 100)}%`;
    fill.style.background = color;
    bar.appendChild(fill);

    // Kept for tuning: the underlying range is still in the response, it is
    // just not what the user leads with. Revealed with the advanced panel.
    const detail = document.createElement('span');
    detail.className = 'face-box__detail';
    detail.textContent = `${formatRange(face)} yrs · ≈ ${formatYears(face.age)}`;

    label.append(unit, age, bar, detail);

    const caveat = tailCaveat(face.age);
    if (caveat) {
      const flag = document.createElement('span');
      flag.className = `face-box__caveat face-box__caveat--${caveat.kind}`;
      flag.textContent = `⚠ ${caveat.short}`;
      label.append(flag);
    }

    box.append(label);

    box.setAttribute(
      'aria-label',
      `Face ${index + 1}: age estimate ${formatAge(face.age)} years, ` +
        `confidence ${Math.round(face.confidence * 100)} percent` +
        (caveat ? `. ${caveat.long}` : ''),
    );

    overlay.appendChild(box);
  });
}

export function renderFaceList(list: HTMLElement, faces: FaceResult[]): void {
  list.replaceChildren();

  faces.forEach((face, index) => {
    const color = confidenceColor(face.confidence);

    const item = document.createElement('li');
    item.className = 'faces__item';

    const heading = document.createElement('div');
    heading.className = 'faces__heading';
    heading.innerHTML = `<span class="faces__index">Face ${index + 1}</span>`;

    const caption = document.createElement('div');
    caption.className = 'faces__caption';
    caption.textContent = 'Age Estimate';

    const age = document.createElement('div');
    age.className = 'faces__age';
    age.textContent = formatAge(face.age);

    const bar = document.createElement('div');
    bar.className = 'faces__bar';
    const fill = document.createElement('div');
    fill.className = 'faces__bar-fill';
    fill.style.width = `${Math.round(face.confidence * 100)}%`;
    fill.style.background = color;
    bar.appendChild(fill);

    const confidence = document.createElement('div');
    confidence.className = 'faces__confidence';
    confidence.textContent = `Confidence ${Math.round(face.confidence * 100)}%`;

    item.append(heading, caption, age, bar, confidence);

    const caveat = tailCaveat(face.age);
    if (caveat) {
      const note = document.createElement('p');
      note.className = `faces__caveat faces__caveat--${caveat.kind}`;
      note.textContent = caveat.long;
      item.append(note);
    }

    // The range and the unrounded value are still returned by the API; the
    // advanced panel keeps them reachable for tuning without putting false
    // precision in front of the user.
    const detail = document.createElement('div');
    detail.className = 'faces__detail';
    detail.textContent =
      `Range ${formatRange(face)} · point ${formatYears(face.age)} · ` +
      `box ${face.bbox.join(', ')}`;
    item.append(detail);

    list.appendChild(item);
  });
}
