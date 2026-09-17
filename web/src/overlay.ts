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
 * cannot ask. This distinction has now inverted a conclusion twice, so it is
 * the first thing to check before changing anything here.
 *
 * These numbers come from APPA-REAL (7,534 images, real chronological ages,
 * YuNet on full original scenes) rather than from UTKFace. UTKFace's labels are
 * DEX-algorithm estimates, so our in-corpus MAE of 4.76 is substantially
 * agreement with the labelling method rather than accuracy against real age.
 * Against real ages the MAE is 8.52. Caveats must be derived from the latter --
 * the in-corpus numbers understate the error a user actually experiences.
 *
 * Binned by DISPLAYED age (n = 7,534, margin 0.0, median decode):
 *
 *   shown     n     MAE    bias
 *   0–10    617    3.34   −1.01
 *   10–20  1126    6.77   −4.43
 *   20–30  1897    6.34   +0.00
 *   30–40  1377    8.04   +3.04
 *   40–50   651   10.27   +6.78
 *   50–60  1198   13.72  +10.94
 *   60–70   467   12.48   +7.76
 *   70+     201   12.22   +7.30
 *
 * One caveat ships, for ages shown 40 and over: MAE 12.48 against 6.53 below
 * 40, with bias +8.98. So the number is both about twice as imprecise and
 * systematically high -- someone displayed as 55 averages 46. It fires on 33%
 * of faces, which is a lot, but the effect is large, monotonic in displayed age
 * and stable across APPA-REAL's splits (+8.75 / +9.14 / +9.07).
 *
 * Two warnings that are NOT shipped, both because re-conditioning on displayed
 * age reversed or dissolved them:
 *
 * 1. No teenager caveat. Binned by *true* age, 10–19 is our worst region
 *    relative to human raters: bias +7.42, i.e. we read teenagers as much
 *    older than they are. But binned by *displayed* age the sign flips --
 *    those shown as 10–20 average bias −4.43, i.e. they are older than shown.
 *    A warning saying "teenagers read old" would fire on a population for whom
 *    the opposite is true. The band is also more accurate than average
 *    (MAE 6.77 vs 8.52 overall), so there is no precision case either, and the
 *    dip is non-monotonic against its neighbours (−1.01, −4.43, +0.00), which
 *    is the signature of a local artefact rather than a stable effect.
 *
 * 2. No old-age caveat. Previously removed on UTKFace evidence; APPA-REAL
 *    confirms it. At true 70–79 our MAE is 10.08 against a single human
 *    rater's 10.01, and our bias (−7.12) is smaller than the human crowd's own
 *    (−7.99). Those faces genuinely read young to people; the labels encode
 *    that and we reproduce it. It is not a defect to warn about.
 *
 * Independently corroborated on FG-NET (998 images, real ages, never used to
 * derive any of the above): shown ≥ 40 has MAE 17.28 and bias +16.55 against
 * 5.57 / +4.24 below 40 — same direction, same shape. Its magnitude is not
 * quoted anywhere, because FG-NET's median true age is 13 and only 7% of it is
 * genuinely over 40, so a high prediction there is near-certainly wrong and the
 * effect is inflated. It is corroboration of direction, not of size. Every
 * number the UI states comes from APPA-REAL, which is 27% over-40.
 *
 * The estimate is never silently corrected. Subtracting the bias would bury a
 * known, measured limitation inside a number that looks authoritative.
 */
const MID_LOW = 40;

export interface TailCaveat {
  kind: 'mid';
  short: string;
  long: string;
}

export function tailCaveat(age: number): TailCaveat | null {
  if (age >= MID_LOW) {
    return {
      kind: 'mid',
      short: 'often reads high — likely younger',
      long:
        'From about 40 upwards this model tends to overestimate, by roughly 9 ' +
        'years on average, and it is about twice as imprecise here as it is ' +
        'with younger faces. The person is more likely younger than the number ' +
        'suggests than older. Treat it as a broad bracket, not a reading.',
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
