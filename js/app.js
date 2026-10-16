// Screen flow: intro -> camera -> processing -> result (or a themed error).

import { startCamera, stopCamera, captureFrame, CameraError } from './camera.js';
import { initFaceSwap, swapFace, FaceSwapError } from './faceswap.js';

const PROCESSING_LINES = [
  'Consulting the spirits...',
  'Gazing into the mists...',
  'Analyzing bone structure...',
  'Reading the lines of fate...',
  'Something is wrong...',
  'Oh no.',
];
const LINE_MS = 850;
const OMINOUS_FROM = 4; // index of the first line where the crystal ball starts to panic

const $ = (id) => document.getElementById(id);
const video = $('video');

let resultBlob = null;
let resultUrl = null;
let reading = false; // guards against double taps starting two readings

// Start loading the model and analysing the source face straight away, in the background.
initFaceSwap().catch((err) => console.error('Face swap init failed:', err));

function show(name) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === `screen-${name}`));
}

function showError(title, message) {
  stopCamera();
  $('error-title').textContent = title;
  $('error-message').textContent = message;
  show('error');
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function beginReading() {
  if (reading) return;
  reading = true;
  try {
    await runReading();
  } finally {
    reading = false;
  }
}

async function runReading() {
  show('camera');
  $('countdown').textContent = '';
  try {
    await startCamera(video);
  } catch (err) {
    console.error(err);
    const kind = err instanceof CameraError ? err.kind : 'unknown';
    const messages = {
      insecure: 'The spirits only speak over a secure connection. Open this page via https:// and try again.',
      denied: 'Madame Sarah cannot read a face she cannot see. Allow camera access in your browser settings, then try again.',
      missing: 'No looking glass was found. This reading needs a front-facing camera.',
      busy: 'Another spell holds your camera. Close other apps using it and try again.',
      unknown: 'The looking glass clouded over. Please try again.',
    };
    showError('The veil is closed', messages[kind]);
    return;
  }

  await wait(900); // a moment to settle into the oval
  for (const n of [3, 2, 1]) {
    const el = $('countdown');
    el.textContent = n;
    el.classList.remove('tick');
    void el.offsetWidth; // restart the animation
    el.classList.add('tick');
    await wait(900);
  }
  $('countdown').textContent = '';
  const flash = $('flash');
  flash.classList.remove('go');
  void flash.offsetWidth;
  flash.classList.add('go');

  const selfie = captureFrame(video);
  stopCamera();
  await wait(250);
  await processReading(selfie);
}

async function playProcessingLines() {
  const screen = $('screen-processing');
  const line = $('processing-line');
  screen.classList.remove('ominous');
  for (let i = 0; i < PROCESSING_LINES.length; i++) {
    line.textContent = PROCESSING_LINES[i];
    line.classList.remove('swap');
    void line.offsetWidth;
    line.classList.add('swap');
    screen.classList.toggle('ominous', i >= OMINOUS_FROM);
    await wait(LINE_MS);
  }
}

async function processReading(selfie) {
  show('processing');
  const swap = swapFace(selfie).then(
    (canvas) => ({ canvas }),
    (error) => ({ error }),
  );
  const [{ canvas, error }] = await Promise.all([swap, playProcessingLines()]);
  $('screen-processing').classList.remove('ominous');

  if (error) {
    console.error(error);
    const kind = error instanceof FaceSwapError ? error.kind : 'unknown';
    const messages = {
      model: 'The spirits are out of reach. Check your connection and try again.',
      'source-missing': 'The crystal ball is empty. (Missing assets/source-face.jpg.)',
      'source-no-face': 'The crystal ball is clouded. (No face found in assets/source-face.jpg.)',
      unknown: 'The reading went awry. Please try again.',
    };
    showError('The ritual failed', messages[kind]);
    return;
  }
  if (!canvas) {
    showError('The mists are too thick', 'The spirits cannot see you. Try better lighting, and keep your whole face within the oval.');
    return;
  }

  await stampDestiny(canvas);
  resultBlob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.92));
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(resultBlob);
  $('result-img').src = resultUrl;
  show('result');
}

/** Bake a rubber-stamp "YOUR DESTINY" into the image so it survives sharing. */
async function stampDestiny(canvas) {
  const font = '"Cinzel Decorative", Georgia, serif';
  try {
    await document.fonts.load(`900 40px ${font}`);
  } catch { /* fall back to Georgia */ }

  const W = canvas.width;
  const H = canvas.height;
  const size = Math.round(Math.min(W, H) * 0.085);
  const stamp = document.createElement('canvas');
  const sctx = stamp.getContext('2d');
  sctx.font = `900 ${size}px ${font}`;
  const textW = sctx.measureText('YOUR DESTINY').width;
  const padX = size * 0.6;
  const padY = size * 0.45;
  stamp.width = Math.ceil(textW + padX * 2);
  stamp.height = Math.ceil(size + padY * 2);

  const ink = '#c4122f';
  sctx.strokeStyle = ink;
  sctx.fillStyle = ink;
  const lw = Math.max(2, size * 0.09);
  sctx.lineWidth = lw;
  roundRect(sctx, lw, lw, stamp.width - lw * 2, stamp.height - lw * 2, size * 0.25);
  sctx.stroke();
  sctx.lineWidth = lw * 0.5;
  roundRect(sctx, lw * 2.6, lw * 2.6, stamp.width - lw * 5.2, stamp.height - lw * 5.2, size * 0.15);
  sctx.stroke();
  sctx.font = `900 ${size}px ${font}`;
  sctx.textAlign = 'center';
  sctx.textBaseline = 'middle';
  sctx.fillText('YOUR DESTINY', stamp.width / 2, stamp.height / 2 + size * 0.05);

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
  ctx.rotate(-0.16);
  ctx.globalAlpha = 0.88;
  ctx.drawImage(stamp, -stamp.width / 2, -stamp.height / 2);
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

async function shareResult() {
  if (!resultBlob) return;
  const file = new File([resultBlob], 'my-destiny.jpg', { type: 'image/jpeg' });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: 'My destiny, by Madame Sarah' });
      return;
    } catch (err) {
      if (err.name === 'AbortError') return; // user closed the share sheet
      console.warn('Share failed, downloading instead', err);
    }
  }
  const a = document.createElement('a');
  a.href = resultUrl;
  a.download = 'my-destiny.jpg';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

$('btn-begin').addEventListener('click', () => beginReading());
$('btn-again').addEventListener('click', () => beginReading());
$('btn-retry').addEventListener('click', () => beginReading());
$('btn-share').addEventListener('click', shareResult);

// Release the camera if the page is backgrounded mid-reading.
window.addEventListener('pagehide', stopCamera);
