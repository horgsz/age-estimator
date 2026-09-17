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
 * Ages where the displayed number is least reliable.
 *
 * THE CAVEAT IS A PROPERTY OF THE CHECKPOINT, NOT OF THIS APP.
 *
 * The two selectable models have genuinely different error profiles, so the
 * caveat set is served per model by /health (keyed by the checkpoint's own
 * content digest) and passed in here. This file decides only how to *say* it.
 *
 * Hardcoding a band here would mean one model's measured limitation being
 * asserted about the other's predictions -- which is exactly wrong in this
 * case, since the ">= 40 reads high" band is true of the apparent-age model
 * and false of the real-age one. An unrecognised checkpoint supplies no
 * caveat at all, which is the safe direction for a directional claim.
 *
 * Thresholds and wording must be conditioned on the *predicted* age, because
 * that is the only thing the UI knows. Binning by true age answers a question
 * the UI cannot ask. This distinction has inverted a conclusion three times in
 * this project, so it is the first thing to check before changing anything here.
 *
 * WHY THE OLD ">= 40 reads high" CAVEAT WAS REMOVED
 *
 * It was correct, and well corroborated, for the previous model (trained on
 * UTKFace's DEX-estimated labels). It is not correct for the model now served,
 * which is trained on real chronological ages. Re-measured end to end through
 * this server's own path (YuNet -> our crop -> model -> median decode) over the
 * real-ground-truth held-out test split (AgeDB + APPA-REAL + FG-NET, n = 3,807):
 *
 *   shown     n     MAE    bias    CS@5
 *    0-10   225    2.15   -0.34   95.1%
 *   10-18   159    4.72   -1.90   75.5%
 *   18-25   353    5.66   -1.72   61.2%
 *   25-40  1527    6.20   -0.15   55.7%
 *   40-50   595    7.30   +1.14   51.4%
 *   50-60   448    7.70   -0.16   47.5%
 *   60-70   280    7.33   -0.51   48.6%
 *     70+   220    7.35   -0.23   46.8%
 *
 * Bias is near zero in every band (-2.56 to +1.14 across finer bins). The
 * previous model showed +5.70 above 40 against -0.55 below it -- a 6.25-year
 * split. The largest equivalent split now available at ANY threshold is 1.52
 * years, at 30, and only 0.80 at 40. There is no direction left to warn about,
 * and a directional warning derived from the old weights would now fire on
 * predictions that are approximately unbiased.
 *
 * WHY NO PRECISION CAVEAT REPLACED IT
 *
 * Precision does still degrade with displayed age (MAE 5.60 below 40 vs 7.43
 * above, a 1.33x ratio). That was the obvious candidate replacement, and it was
 * rejected on measurement: the confidence bar already carries this signal, per
 * face, and carries it better than a band threshold could.
 *
 *   confidence quartile:   Q1 8.86   Q2 6.46   Q3 5.84   Q4 4.21   (MAE)
 *   Pearson r(confidence, |error|) = -0.343
 *
 * and it already tracks the displayed-age drop that a band caveat would state:
 *
 *   shown  0-25  mean confidence 0.596   MAE 4.37
 *   shown 25-40  mean confidence 0.514   MAE 6.20
 *   shown 40-60  mean confidence 0.485   MAE 7.47
 *   shown  60+   mean confidence 0.507   MAE 7.34
 *
 * A band caveat would restate a continuous per-face signal as a step function,
 * and would be wrong for the many high-confidence faces above 40. Shipping it
 * would add noise, not information.
 *
 * WHAT IS STILL TRUE AND IS STATED ELSEWHERE
 *
 * Overall error is still large in absolute terms (~6.3 years against real age),
 * and the model still cannot be used for age verification -- about 30% of true
 * under-18s display as 18 or over. Both are in the page header, because they
 * apply to every face rather than to a band.
 *
 * WHY THIS IS KEPT RATHER THAN DELETED
 *
 * The correct caveat has now changed three times, twice reversing direction,
 * and each time the trap was re-deriving from true-age bins. Keeping the
 * derivation and the negative result means the next person starts from the
 * evidence rather than from scratch -- and the standard to clear is stated: a
 * caveat ships only if it is supportable on DISPLAYED age, on the weights
 * actually served, and is not already better conveyed by the confidence bar.
 *
 * The estimate is never silently corrected. Subtracting a bias would bury a
 * known, measured limitation inside a number that looks authoritative.
 */
export interface TailCaveat {
  kind: 'mid';
  short: string;
  long: string;
}

/** A model's caveat band, as served by `/health`. */
export interface CaveatSpec {
  min_age: number;
  short: string;
  long: string;
}

export function tailCaveat(age: number, spec?: CaveatSpec | null): TailCaveat | null {
  if (!spec || !Number.isFinite(age) || age < spec.min_age) return null;
  return { kind: 'mid', short: spec.short, long: spec.long };
}

/** Confidence ramp: red (uncertain) to green (confident). */
function confidenceColor(confidence: number): string {
  const hue = Math.round(Math.max(0, Math.min(1, confidence)) * 120);
  return `hsl(${hue} 85% 55%)`;
}

export function renderBoxes(
  overlay: HTMLElement,
  faces: FaceResult[],
  imageWidth: number,
  imageHeight: number,
  caveatSpec?: CaveatSpec | null,
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

    const caveat = tailCaveat(face.age, caveatSpec);
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

export function renderFaceList(
  list: HTMLElement,
  faces: FaceResult[],
  caveatSpec?: CaveatSpec | null,
): void {
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

    const caveat = tailCaveat(face.age, caveatSpec);
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
