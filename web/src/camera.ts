/**
 * Camera access, device selection, and frame capture.
 *
 * ## Mirroring and coordinate spaces (read before changing anything)
 *
 * A selfie preview is mirrored, otherwise moving left appears to move right and
 * it feels broken. That mirroring is display-only on the `<video>` element,
 * which makes it very easy to end up with bounding boxes drawn on the wrong side
 * of the face: the server sees un-mirrored pixels, the user sees mirrored ones.
 *
 * We avoid that class of bug by construction: {@link Camera.captureFrame} bakes
 * the mirror into the captured canvas when the preview is mirrored. So the
 * bytes uploaded to `/estimate` are *exactly* the pixels the user was looking
 * at, the returned bboxes are already in that same coordinate space, and the
 * overlay needs no flip. Uploaded files are never mirrored and follow the same
 * single-space rule.
 *
 * Invariant: **the image POSTed to the server is always the image rendered in
 * the result view.** Keep it that way.
 */

export type CameraErrorKind =
  | 'unsupported'
  | 'insecure'
  | 'denied'
  | 'notfound'
  | 'inuse'
  | 'other';

export class CameraError extends Error {
  readonly kind: CameraErrorKind;

  constructor(kind: CameraErrorKind, message: string) {
    super(message);
    this.name = 'CameraError';
    this.kind = kind;
  }
}

function classify(err: unknown): CameraError {
  const name = err instanceof DOMException ? err.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return new CameraError(
        'denied',
        'Camera permission was denied. Allow camera access in your browser’s site settings, then try again. You can still upload a photo instead.',
      );
    case 'NotFoundError':
    case 'OverconstrainedError':
      return new CameraError('notfound', 'No camera matched the request.');
    case 'NotReadableError':
    case 'AbortError':
      return new CameraError(
        'inuse',
        'The camera could not be started — another app may be using it.',
      );
    default:
      return new CameraError(
        'other',
        err instanceof Error ? err.message : 'The camera could not be started.',
      );
  }
}

export class Camera {
  private stream: MediaStream | null = null;
  private mirrored = true;

  constructor(private readonly video: HTMLVideoElement) {
    this.applyMirror();
  }

  get isActive(): boolean {
    return this.stream !== null;
  }

  get isMirrored(): boolean {
    return this.mirrored;
  }

  static get isSupported(): boolean {
    return Boolean(navigator.mediaDevices?.getUserMedia);
  }

  setMirrored(on: boolean): void {
    this.mirrored = on;
    this.applyMirror();
  }

  private applyMirror(): void {
    this.video.style.transform = this.mirrored ? 'scaleX(-1)' : 'none';
  }

  /** Start (or restart) the preview, optionally on a specific device. */
  async start(deviceId?: string): Promise<void> {
    if (!Camera.isSupported) {
      // getUserMedia is only exposed on secure origins (https or localhost).
      throw new CameraError(
        window.isSecureContext ? 'unsupported' : 'insecure',
        window.isSecureContext
          ? 'This browser does not support camera capture.'
          : 'Camera access needs a secure context — use http://localhost or https.',
      );
    }

    this.stop();

    const video: MediaTrackConstraints = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } };

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (err) {
      throw classify(err);
    }

    this.stream = stream;
    this.video.srcObject = stream;

    try {
      await this.video.play();
    } catch {
      /* autoplay policies; the element is muted so this rarely fires */
    }
    await this.waitForDimensions();
  }

  private waitForDimensions(): Promise<void> {
    if (this.video.videoWidth > 0) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        this.video.removeEventListener('loadedmetadata', done);
        resolve();
      };
      this.video.addEventListener('loadedmetadata', done, { once: true });
      window.setTimeout(done, 3000);
    });
  }

  stop(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  /** Video input devices. Labels are only populated after permission is granted. */
  async listDevices(): Promise<MediaDeviceInfo[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter((d) => d.kind === 'videoinput');
    } catch {
      return [];
    }
  }

  /**
   * Draw the current frame to a canvas, applying the mirror if the preview is
   * mirrored, so the canvas matches what the user saw.
   */
  captureFrame(): HTMLCanvasElement {
    const width = this.video.videoWidth;
    const height = this.video.videoHeight;
    if (!width || !height) {
      throw new CameraError('other', 'The camera has not produced a frame yet.');
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new CameraError('other', 'Could not get a 2D canvas context.');

    if (this.mirrored) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(this.video, 0, 0, width, height);
    return canvas;
  }
}

/** Export a canvas as a JPEG blob. */
export function canvasToJpeg(canvas: HTMLCanvasElement, quality = 0.9): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the frame as JPEG.'))),
      'image/jpeg',
      quality,
    );
  });
}
