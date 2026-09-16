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
 * Ages where the displayed number is systematically offset, and the app says so.
 *
 * Thresholds and wording are conditioned on the *predicted* age, because that
 * is the only thing the UI knows. That distinction matters: binned by true age
 * the model looks catastrophic at the top (MAE 10.8 at 80+), but binned by what
 * it actually displays the picture is different, because it rarely commits to
 * an extreme number and is roughly right when it does.
 *
 * Measured end to end over 1,184 UTKFace test images at the shipped margin:
 *
 *   shown    n     MAE    bias
 *   0–5      25    3.09   +1.54
 *   5–10    103    3.98   +3.55
 *   10–12    23    4.63   +3.49
 *   40–50   121    7.60   +2.83
 *   65–70    34    8.19   −0.20
 *   70–75    18    6.56   −0.69
 *   75+      22    5.83   −3.48
 *
 * So the honest caveats are about a systematic *offset*, not about precision
 * collapsing. A shown age under 12 runs about 3 years high (and the model never
 * outputs below ~3, so it has a floor). A shown age over 65 runs low, and more
 * so the higher it goes — the model compresses the top of the range, which is
 * why a genuinely 85-year-old face typically displays around 70.
 *
 * We surface this rather than silently correcting it: a correction would bake a
 * dataset artefact into the served answer and hide it from the user.
 *
 * Note the worst band by MAE is actually 40–70 (7.0–8.2), not the extremes. It
 * is deliberately not caveated — flagging most of the range would dilute the
 * signal, and the confidence bar already varies there.
 */
const YOUNG_TAIL = 12;
// 65 rather than 70: tail compression means an 85-year-old typically displays
// around 70, so a threshold at 70 misses the very cases the note is for.
const OLD_TAIL = 65;

export interface TailCaveat {
  kind: 'young' | 'old';
  short: string;
  long: string;
}

export function tailCaveat(age: number): TailCaveat | null {
  if (age < YOUNG_TAIL) {
    return {
      kind: 'young',
      short: 'likely younger than shown',
      long:
        'The model reads high for young faces — by about 3 years in this range, ' +
        'and it never outputs an age below roughly 3. A child is probably ' +
        'younger than this says, and for an infant the number is the model’s ' +
        'floor rather than a measurement.',
    };
  }
  if (age > OLD_TAIL) {
    return {
      kind: 'old',
      short: 'likely older than shown',
      long:
        'The model compresses the top of its range, so older faces read low and ' +
        'increasingly so with age — a face in its mid-eighties typically shows ' +
        'as about 70. The true age is likely higher than this says.',
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
