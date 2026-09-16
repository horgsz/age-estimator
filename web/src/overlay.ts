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

export function formatRange(face: FaceResult): string {
  return `${formatYears(face.low)}–${formatYears(face.high)}`;
}

/**
 * Ages where the model is measurably biased, and the app should say so.
 *
 * Measured on the UTKFace test split: bias runs +5.6 yrs at ages 0–9 and
 * −9.4 yrs at 80+. At 0–9 the MAE *equals* the bias, meaning the model never
 * under-predicts a child — infants read as roughly 6–11. That is a floor, not
 * a measurement, and someone photographing a toddler deserves to be told.
 *
 * We surface it rather than silently correcting it: a correction would bake a
 * dataset artefact into the served answer and hide it from the user.
 */
const YOUNG_TAIL = 12;
const OLD_TAIL = 70;

export interface TailCaveat {
  kind: 'young' | 'old';
  short: string;
  long: string;
}

export function tailCaveat(age: number): TailCaveat | null {
  if (age < YOUNG_TAIL) {
    return {
      kind: 'young',
      short: 'reads high for children',
      long:
        'Accuracy degrades at this end of the range. The model reads high for ' +
        'children — it effectively never guesses below about 6, so infants and ' +
        'toddlers come out far too old. Treat this as the model’s floor, not a ' +
        'measurement.',
    };
  }
  if (age > OLD_TAIL) {
    return {
      kind: 'old',
      short: 'reads low for older faces',
      long:
        'Accuracy degrades at this end of the range. The model reads low for ' +
        'older faces, by roughly 9 years past 80, so the true age is likely ' +
        'higher than shown.',
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

    const range = document.createElement('span');
    range.className = 'face-box__range';
    range.textContent = `${formatRange(face)} yrs`;

    const point = document.createElement('span');
    point.className = 'face-box__point';
    point.textContent = `≈ ${formatYears(face.age)}`;

    const bar = document.createElement('span');
    bar.className = 'face-box__bar';
    const fill = document.createElement('span');
    fill.className = 'face-box__bar-fill';
    fill.style.width = `${Math.round(face.confidence * 100)}%`;
    fill.style.background = color;
    bar.appendChild(fill);

    label.append(range, point, bar);

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
      `Face ${index + 1}: estimated ${formatRange(face)} years, ` +
        `point estimate ${formatYears(face.age)}, ` +
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

    const range = document.createElement('div');
    range.className = 'faces__range';
    range.textContent = `${formatRange(face)} years`;

    const point = document.createElement('div');
    point.className = 'faces__point';
    point.textContent = `Point estimate ${formatYears(face.age)} · box ${face.bbox.join(', ')}`;

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

    item.append(heading, range, point, bar, confidence);

    const caveat = tailCaveat(face.age);
    if (caveat) {
      const note = document.createElement('p');
      note.className = `faces__caveat faces__caveat--${caveat.kind}`;
      note.textContent = caveat.long;
      item.append(note);
    }

    list.appendChild(item);
  });
}
