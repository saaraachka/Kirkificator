// Front-camera access and frame capture.

const MAX_CAPTURE_SIZE = 960; // longest side of the captured selfie, in px

let stream = null;

export class CameraError extends Error {
  constructor(kind, cause) {
    super(kind);
    this.kind = kind; // 'insecure' | 'denied' | 'missing' | 'busy' | 'unknown'
    this.cause = cause;
  }
}

export async function startCamera(video) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new CameraError('insecure');
  }
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 1280 } },
    });
  } catch (err) {
    const kind = {
      NotAllowedError: 'denied',
      SecurityError: 'denied',
      NotFoundError: 'missing',
      OverconstrainedError: 'missing',
      NotReadableError: 'busy',
      AbortError: 'busy',
    }[err.name] || 'unknown';
    throw new CameraError(kind, err);
  }
  video.srcObject = stream;
  // iOS Safari needs playsinline + muted (set in the markup) and an explicit play().
  await video.play().catch(() => {});
  if (!video.videoWidth) {
    await new Promise((resolve) => video.addEventListener('loadedmetadata', resolve, { once: true }));
  }
}

export function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

/**
 * Capture exactly what the user sees: the object-fit: cover crop of the video,
 * mirrored like the preview, scaled so the longest side is at most MAX_CAPTURE_SIZE.
 */
export function captureFrame(video) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const ew = video.clientWidth || vw;
  const eh = video.clientHeight || vh;

  const scale = Math.max(ew / vw, eh / vh);
  const sw = ew / scale;
  const sh = eh / scale;
  const sx = (vw - sw) / 2;
  const sy = (vh - sh) / 2;

  const out = Math.min(1, MAX_CAPTURE_SIZE / Math.max(sw, sh));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(sw * out);
  canvas.height = Math.round(sh * out);
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  return canvas;
}
