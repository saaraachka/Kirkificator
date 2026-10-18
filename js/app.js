// Screen flow: intro -> camera (live preview, tap to capture, then use it or retake)
// -> processing lines -> result (or an error).

import { startCamera, stopCamera, captureFrame, CameraError } from './camera.js';
import { initFaceSwap, hasFace, swapFace, swapFaceRemote, FaceSwapError } from './faceswap.js';

const PROCESSING_LINES = [
  'Measuring your skull for the collection...',
  'The ringmaster has seen your photo. He has notes.',
  'Removing your soul. Please do not move.',
  "Consulting the bearded lady. She's not convinced.",
  "This is taking longer than usual. That's not a good sign.",
  "We're so sorry about what happens next.",
];
const LINE_MS = 2600; // how long each line stays fully visible
const FADE_MS = 700; // matches the .processing-line opacity transition
const MIN_LINES = 3; // always shown, however fast the swap is
const OMINOUS_FROM = 4; // index of the first line where the bulbs start to sputter

const FACE_CHECK_MS = 250; // how often the live preview is checked for a face
const PROBE_WIDTH = 256; // the preview is downscaled to this width for detection

const ERROR_TITLE = 'INTERMISSION';

const $ = (id) => document.getElementById(id);
const video = $('video');
const cameraScreen = $('screen-camera');

let resultBlob = null;
let resultUrl = null;
let selfie = null; // the captured frame awaiting "USE IT"
let watchId = 0; // bumped to stop the live face check
let opening = false; // guards against double taps opening the camera twice
let processing = false; // guards against double taps on "USE IT"

// Start loading the model and analysing the source face straight away, in the background.
initFaceSwap().catch((err) => console.error('Face swap init failed:', err));

function show(name) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === `screen-${name}`));
  document.body.dataset.screen = name;
}

function showError(message, button = 'TRY AGAIN') {
  stopWatching();
  stopCamera();
  document.body.classList.remove('ominous');
  $('error-title').textContent = ERROR_TITLE;
  $('error-message').textContent = message;
  $('btn-retry').textContent = button;
  show('error');
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Camera: live preview with a face check, tap to capture, then use it or retake
// ---------------------------------------------------------------------------

async function openCamera() {
  if (opening || processing) return;
  opening = true;
  try {
    selfie = null;
    cameraScreen.classList.remove('reviewing', 'snap');
    setFaceFound(false);
    show('camera');
    try {
      await startCamera(video);
    } catch (err) {
      console.error(err);
      const kind = err instanceof CameraError ? err.kind : 'unknown';
      const messages = {
        insecure: 'This only works over a secure connection. Open the page with https:// and try again.',
        denied: "We need your camera. We're mostly harmless.",
        missing: 'No front camera found. The exhibit needs a face, and something to see it with.',
        busy: 'Something else is using your camera. Close it and come back.',
        unknown: "The camera jammed. It happens. Not usually this early.",
      };
      showError(messages[kind], kind === 'denied' ? 'LET US IN' : 'TRY AGAIN');
      return;
    }
    startWatching();
  } finally {
    opening = false;
  }
}

function setFaceFound(found) {
  $('face-guide').classList.toggle('found', found);
  $('btn-take').disabled = !found;
}

/** Check the live preview for a face a few times a second; the shutter only works while one is found. */
async function startWatching() {
  const id = ++watchId;
  const probe = document.createElement('canvas');
  while (id === watchId) {
    if (video.videoWidth) {
      probe.width = PROBE_WIDTH;
      probe.height = Math.round((PROBE_WIDTH * video.videoHeight) / video.videoWidth);
      probe.getContext('2d').drawImage(video, 0, 0, probe.width, probe.height);
      let found;
      try {
        found = await hasFace(probe);
      } catch (err) {
        console.error(err);
        if (id === watchId) showError("We couldn't load the face detector. Check your connection and try again.");
        return;
      }
      if (id !== watchId) return;
      setFaceFound(found);
    }
    await wait(FACE_CHECK_MS);
  }
}

function stopWatching() {
  watchId++;
}

function takePhoto() {
  if ($('btn-take').disabled || !video.videoWidth) return;
  stopWatching();
  selfie = captureFrame(video);
  const review = $('review');
  review.width = selfie.width;
  review.height = selfie.height;
  review.getContext('2d').drawImage(selfie, 0, 0);
  cameraScreen.classList.add('reviewing');
  const flash = $('flash');
  for (const [el, cls] of [[flash, 'go'], [cameraScreen, 'snap']]) {
    el.classList.remove(cls);
    void el.offsetWidth; // restart the animation
    el.classList.add(cls);
  }
}

function retake() {
  selfie = null;
  cameraScreen.classList.remove('reviewing', 'snap');
  setFaceFound(false);
  startWatching();
}

async function usePhoto() {
  if (!selfie || processing) return;
  processing = true;
  const shot = selfie;
  selfie = null;
  stopWatching();
  stopCamera();
  try {
    await processPhoto(shot);
  } finally {
    processing = false;
  }
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------

/**
 * Show each line for at least LINE_MS with a slow fade. Once `until` settles (and at least
 * MIN_LINES have been shown) the current line finishes and the sequence stops. If the
 * swap outlasts the list, the last line stays up with a pulsing ellipsis.
 */
async function playProcessingLines(until) {
  let settled = false;
  until.finally(() => { settled = true; });
  const line = $('processing-line');
  for (let i = 0; i < PROCESSING_LINES.length; i++) {
    const last = i === PROCESSING_LINES.length - 1;
    line.textContent = PROCESSING_LINES[i];
    document.body.classList.toggle('ominous', i >= OMINOUS_FROM);
    line.classList.add('show');
    await wait(FADE_MS + LINE_MS);
    if (last && !settled) {
      line.textContent = line.textContent.replace(/\.+$/, ''); // the ellipsis replaces the full stop
      line.append(ellipsis());
      await until;
    }
    line.classList.remove('show');
    await wait(FADE_MS);
    if (settled && i + 1 >= MIN_LINES) break;
  }
  line.textContent = '';
}

function ellipsis() {
  const span = document.createElement('span');
  span.className = 'ellipsis';
  span.setAttribute('aria-hidden', 'true');
  span.innerHTML = '<i>.</i><i>.</i><i>.</i>';
  return span;
}

/**
 * Check for a face on-device, then try the AI swap, falling back to the on-device swap
 * if it fails or times out. Resolves to null if there is no face in the selfie.
 */
async function kirkify(selfie) {
  if (!(await hasFace(selfie))) return null; // nothing is uploaded without a face
  try {
    return await swapFaceRemote(selfie);
  } catch (err) {
    console.warn('AI face swap failed, using the on-device swap instead:', err.cause || err);
  }
  return swapFace(selfie);
}

async function processPhoto(selfie) {
  show('processing');
  const swap = kirkify(selfie).then(
    (canvas) => ({ canvas }),
    (error) => ({ error }),
  );
  const [{ canvas, error }] = await Promise.all([swap, playProcessingLines(swap)]);
  document.body.classList.remove('ominous');

  if (error) {
    console.error(error);
    const kind = error instanceof FaceSwapError ? error.kind : 'unknown';
    const messages = {
      model: "We couldn't load the face detector. Check your connection and try again.",
      'source-missing': 'The exhibit is missing its star. (Missing assets/source-face.jpg.)',
      'source-no-face': "The exhibit's star has no face. (No face found in assets/source-face.jpg.)",
      unknown: 'Something broke backstage. Try again.',
    };
    showError(messages[kind]);
    return;
  }
  if (!canvas) {
    showError("The camera saw nothing. That's either a lighting problem or a you problem.");
    return;
  }

  await stampNoRefunds(canvas);
  resultBlob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.92));
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(resultBlob);
  $('result-img').src = resultUrl;
  show('result');
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/** Bake a red rubber ticket stamp, "NO REFUNDS", into the image so it survives sharing. */
async function stampNoRefunds(canvas) {
  const font = '"Rye", Georgia, serif';
  try {
    await document.fonts.load(`400 40px ${font}`);
  } catch { /* fall back to Georgia */ }

  const W = canvas.width;
  const H = canvas.height;
  const text = 'NO REFUNDS';
  const size = Math.round(Math.min(W, H) * 0.1);
  const stamp = document.createElement('canvas');
  const sctx = stamp.getContext('2d');
  sctx.font = `400 ${size}px ${font}`;
  const textW = sctx.measureText(text).width;
  const padX = size * 0.95; // room for the notches
  const padY = size * 0.5;
  stamp.width = Math.ceil(textW + padX * 2);
  stamp.height = Math.ceil(size + padY * 2);

  const ink = '#c0141c';
  sctx.strokeStyle = ink;
  sctx.fillStyle = ink;
  const lw = Math.max(2, size * 0.09);
  const notch = stamp.height * 0.2;
  sctx.lineWidth = lw;
  ticketPath(sctx, lw, lw, stamp.width - lw * 2, stamp.height - lw * 2, notch);
  sctx.stroke();
  sctx.lineWidth = lw * 0.5;
  ticketPath(sctx, lw * 2.8, lw * 2.8, stamp.width - lw * 5.6, stamp.height - lw * 5.6, notch * 0.7);
  sctx.stroke();
  sctx.font = `400 ${size}px ${font}`;
  sctx.textAlign = 'center';
  sctx.textBaseline = 'middle';
  sctx.fillText(text, stamp.width / 2, stamp.height / 2 + size * 0.05);

  // Worn-ink speckle.
  sctx.globalCompositeOperation = 'destination-out';
  const specks = Math.round((stamp.width * stamp.height) / 90);
  for (let i = 0; i < specks; i++) {
    sctx.globalAlpha = Math.random() * 0.8;
    sctx.beginPath();
    sctx.arc(Math.random() * stamp.width, Math.random() * stamp.height, Math.random() * size * 0.05, 0, Math.PI * 2);
    sctx.fill();
  }

  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.translate(W * 0.5, H - stamp.height * 0.9 - H * 0.04);
  ctx.rotate(-0.14);
  ctx.globalAlpha = 0.88;
  ctx.drawImage(stamp, -stamp.width / 2, -stamp.height / 2);
  ctx.restore();
}

/** A ticket outline: a rectangle with a half-round notch bitten out of each short end. */
function ticketPath(ctx, x, y, w, h, r) {
  const cy = y + h / 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, cy - r);
  ctx.arc(x + w, cy, r, -Math.PI / 2, Math.PI / 2, true);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, cy + r);
  ctx.arc(x, cy, r, Math.PI / 2, -Math.PI / 2, true);
  ctx.closePath();
}

async function shareResult() {
  if (!resultBlob) return;
  const file = new File([resultBlob], 'the-new-attraction.jpg', { type: 'image/jpeg' });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: 'Presenting: the new attraction' });
      return;
    } catch (err) {
      if (err.name === 'AbortError') return; // user closed the share sheet
      console.warn('Share failed, downloading instead', err);
    }
  }
  const a = document.createElement('a');
  a.href = resultUrl;
  a.download = 'the-new-attraction.jpg';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

$('btn-begin').addEventListener('click', openCamera);
$('btn-take').addEventListener('click', takePhoto);
$('btn-retake').addEventListener('click', retake);
$('btn-use').addEventListener('click', usePhoto);
$('btn-again').addEventListener('click', openCamera);
$('btn-retry').addEventListener('click', openCamera);
$('btn-share').addEventListener('click', shareResult);

// Release the camera if the page is backgrounded mid-capture.
window.addEventListener('pagehide', () => {
  stopWatching();
  stopCamera();
});
