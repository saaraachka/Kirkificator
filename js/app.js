// Screen flow: intro -> camera (live preview, tap to capture, then use it or retake)
// -> clayface (the loading screen, js/clay.js) -> result (or an error).

import { startCamera, stopCamera, captureFrame, setTorch, CameraError } from './camera.js';
import { initFaceSwap, hasFace, detectFace, swapFace, swapFaceRemote, FaceSwapError } from './faceswap.js';
import { createClayStage, MIN_KNEAD_MS, CLAY_IN_MS } from './clay.js';

// Shown one at a time over the kneading, then looped.
const KNEAD_LINES = [
  "Hold still. We're working with what we've got.",
  'The sculptor has asked for more clay.',
  'Removing your soul. Please do not move.',
  "This is taking longer than usual. That's not a good sign.",
  "Almost done. That's a lie.",
];
const LINE_MS = 2600; // how long each line stays fully visible
const FADE_MS = 900; // matches the .knead-line opacity transition

const FACE_CHECK_MS = 250; // how often the live preview is checked for a face
const PROBE_WIDTH = 256; // the preview is downscaled to this width for detection

// Screen flash for the front camera (like a phone's front "flash"): the screen turns
// warm white to light the face, then the frame is captured.
/** AUTO fires the flash when the preview's average brightness (0-255) is below this. */
const FLASH_AUTO_THRESHOLD = 60;
/** How long the screen stays lit before capturing, so the camera's auto-exposure can adjust. */
const FLASH_PRE_MS = 250;
/** The flash colour: a warm white, kinder to skin than pure white. */
const FLASH_COLOR = '#FFF6E8';
const FLASH_FADE_MS = 150; // the light fades out over this long after the capture
const FLASH_MODES = ['auto', 'on', 'off'];
const FLASH_DEFAULT = 'on'; // fires on every photo unless switched off
// Saved under a new key so choices saved before ON became the default are ignored once.
const FLASH_KEY = 'kirkificator.flash.v2';
const FLASH_OLD_KEYS = ['kirkificator.flash'];

const ERROR_TITLE = 'INTERMISSION';

const $ = (id) => document.getElementById(id);
const video = $('video');
const cameraScreen = $('screen-camera');

let resultBlob = null;
let resultUrl = null;
let videoBlob = null;
let videoUrl = null;
let selfie = null; // the captured frame awaiting "USE IT"
let watchId = 0; // bumped to stop the live face check
let opening = false; // guards against double taps opening the camera twice
let processing = false; // guards against double taps on "USE IT"
let capturing = false; // guards against double taps on "TAKE IT" while the flash is up
let previewLuma = 255; // average brightness of the live preview, 0-255
let flashMode = loadFlashMode();

// Start loading the model and analysing the source face straight away, in the background.
initFaceSwap().catch((err) => console.error('Face swap init failed:', err));

function show(name) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === `screen-${name}`));
  document.body.dataset.screen = name;
}

function showError(message, button = 'TRY AGAIN') {
  stopWatching();
  stopCamera();
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

// ---------------------------------------------------------------------------
// Screen flash
// ---------------------------------------------------------------------------

function loadFlashMode() {
  try {
    for (const key of FLASH_OLD_KEYS) localStorage.removeItem(key);
    const saved = localStorage.getItem(FLASH_KEY);
    if (FLASH_MODES.includes(saved)) return saved;
  } catch { /* storage unavailable */ }
  return FLASH_DEFAULT;
}

function saveFlashMode(mode) {
  try {
    localStorage.setItem(FLASH_KEY, mode);
  } catch { /* storage unavailable */ }
}

/** True if TAKE IT would fire the flash right now. */
function flashWillFire() {
  return flashMode === 'on' || (flashMode === 'auto' && previewLuma < FLASH_AUTO_THRESHOLD);
}

function updateFlashButton() {
  const btn = $('btn-flash');
  btn.dataset.mode = flashMode;
  btn.classList.toggle('armed', flashWillFire());
  btn.querySelector('.flash-mode').textContent = flashMode.toUpperCase();
  btn.setAttribute('aria-label', `Flash: ${flashMode}`);
}

function cycleFlashMode() {
  flashMode = FLASH_MODES[(FLASH_MODES.indexOf(flashMode) + 1) % FLASH_MODES.length];
  saveFlashMode(flashMode);
  updateFlashButton();
}

/** Average brightness (Rec. 601 luma) of the probe frame, sampling every 4th pixel. */
function measureLuma(probe) {
  const { data } = probe.getContext('2d').getImageData(0, 0, probe.width, probe.height);
  let sum = 0, n = 0;
  for (let i = 0; i < data.length; i += 16) {
    sum += data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    n++;
  }
  return n ? sum / n : 255;
}

/** Resolve once the camera has delivered a new frame (or after a short timeout). */
function nextVideoFrame() {
  return new Promise((resolve) => {
    const done = () => { clearTimeout(t); resolve(); };
    const t = setTimeout(done, 100);
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(done);
    else requestAnimationFrame(() => requestAnimationFrame(done));
  });
}

/**
 * Light the face with the screen: cover everything in warm white, give the camera's
 * auto-exposure FLASH_PRE_MS to adjust, capture from the video element (so the overlay
 * itself is never in the photo), then fade the light out. One flash, never repeated.
 */
async function captureWithFlash() {
  const light = $('screen-flash');
  const theme = document.querySelector('meta[name="theme-color"]');
  const themeWas = theme?.content;
  light.style.background = FLASH_COLOR;
  light.style.transition = 'none';
  light.classList.add('on');
  document.body.classList.add('flashing');
  if (theme) theme.content = FLASH_COLOR; // light the browser's own bars too
  setTorch(true);
  await wait(FLASH_PRE_MS);
  await nextVideoFrame();
  const shot = captureFrame(video);
  setTorch(false);
  light.style.transition = `opacity ${FLASH_FADE_MS}ms ease-out`;
  light.classList.remove('on');
  document.body.classList.remove('flashing');
  if (theme && themeWas) theme.content = themeWas;
  return shot;
}

/** Check the live preview for a face a few times a second; the shutter only works while one is found. */
async function startWatching() {
  const id = ++watchId;
  const probe = document.createElement('canvas');
  while (id === watchId) {
    if (video.videoWidth) {
      probe.width = PROBE_WIDTH;
      probe.height = Math.round((PROBE_WIDTH * video.videoHeight) / video.videoWidth);
      probe.getContext('2d', { willReadFrequently: true }).drawImage(video, 0, 0, probe.width, probe.height);
      previewLuma = measureLuma(probe);
      updateFlashButton();
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

async function takePhoto() {
  if ($('btn-take').disabled || !video.videoWidth || capturing) return;
  capturing = true;
  stopWatching();
  const flashFired = flashWillFire();
  try {
    selfie = flashFired ? await captureWithFlash() : captureFrame(video);
  } finally {
    capturing = false;
  }
  const review = $('review');
  review.width = selfie.width;
  review.height = selfie.height;
  review.getContext('2d').drawImage(selfie, 0, 0);
  cameraScreen.classList.add('reviewing');
  // The screen flash replaces the usual shutter flash; the photo still pops in.
  const anims = flashFired ? [[cameraScreen, 'snap']] : [[$('flash'), 'go'], [cameraScreen, 'snap']];
  for (const [el, cls] of anims) {
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
// Processing: the clayface
// ---------------------------------------------------------------------------

/**
 * One line at a time at the bottom of the kneading, each held for LINE_MS with a slow
 * fade, looping through KNEAD_LINES until stop() is called.
 */
function startKneadLines() {
  const line = $('knead-line');
  let stopped = false;
  (async () => {
    for (let i = 0; !stopped; i = (i + 1) % KNEAD_LINES.length) {
      line.textContent = KNEAD_LINES[i];
      line.classList.add('show');
      await wait(FADE_MS + LINE_MS);
      line.classList.remove('show');
      await wait(FADE_MS);
    }
  })();
  return () => {
    stopped = true;
    line.classList.remove('show');
  };
}

/**
 * Try the AI swap, falling back to the on-device swap if it fails or times out.
 * Only called once a face has been found, so nothing faceless is uploaded.
 */
async function kirkify(selfie) {
  try {
    return await swapFaceRemote(selfie);
  } catch (err) {
    console.warn('AI face swap failed, using the on-device swap instead:', err.cause || err);
  }
  return swapFace(selfie);
}

/** `image` redrawn at exactly w x h (the AI output can come back a few pixels off). */
function fitTo(image, w, h) {
  if (image.width === w && image.height === h) return image;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  c.getContext('2d').drawImage(image, 0, 0, w, h);
  return c;
}

function copyCanvas(image) {
  const c = document.createElement('canvas');
  c.width = image.width;
  c.height = image.height;
  c.getContext('2d').drawImage(image, 0, 0);
  return c;
}

/** A fresh full-screen canvas for the clay stage (a canvas can only ever hold one context type). */
function freshStage() {
  const host = $('clay-host');
  host.replaceChildren();
  const canvas = document.createElement('canvas');
  canvas.className = 'clay-stage';
  canvas.setAttribute('aria-label', 'Your face, being remodelled');
  host.append(canvas);
  return canvas;
}

async function processPhoto(selfie) {
  let landmarks;
  try {
    landmarks = await detectFace(selfie);
  } catch (err) {
    console.error(err);
    showError("We couldn't load the face detector. Check your connection and try again.");
    return;
  }
  if (!landmarks) {
    showError("The camera saw nothing. That's either a lighting problem or a you problem.");
    return;
  }

  show('processing');
  const startedAt = performance.now();
  let stopLines = null;
  const onPhase = (phase) => {
    document.body.dataset.clay = phase;
    if (phase === 'knead' && !stopLines) stopLines = startKneadLines();
    if (phase === 'become' && stopLines) stopLines();
  };
  let stage = null;
  try {
    stage = createClayStage(freshStage(), selfie, landmarks, { onPhase });
  } catch (err) {
    console.error('Clayface unavailable:', err);
    stopLines = startKneadLines();
  }

  const { canvas, error } = await kirkify(selfie).then((c) => ({ canvas: c }), (e) => ({ error: e }));
  if (error || !canvas) {
    stage?.destroy();
    stopLines?.();
    if (error) console.error(error);
    const kind = error instanceof FaceSwapError ? error.kind : error ? 'unknown' : 'none';
    const messages = {
      model: "We couldn't load the face detector. Check your connection and try again.",
      'source-missing': 'The exhibit is missing its star. (Missing assets/source-face.jpg.)',
      'source-no-face': "The exhibit's star has no face. (No face found in assets/source-face.jpg.)",
      unknown: 'Something broke backstage. Try again.',
      none: "The camera saw nothing. That's either a lighting problem or a you problem.",
    };
    showError(messages[kind]);
    return;
  }

  const result = fitTo(canvas, selfie.width, selfie.height);
  const stamped = copyCanvas(result);
  await stampKirked(stamped);
  resultBlob = await new Promise((r) => stamped.toBlob(r, 'image/jpeg', 0.92));

  let video = null;
  if (stage) {
    const resultLandmarks = await detectFace(result).catch(() => null);
    ({ video } = await stage.finish(result, resultLandmarks, stamped));
  } else {
    await wait(Math.max(0, startedAt + CLAY_IN_MS + MIN_KNEAD_MS - performance.now()));
  }
  stopLines?.();

  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(resultBlob);
  $('result-img').src = resultUrl;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoBlob = video;
  videoUrl = video ? URL.createObjectURL(video) : null;
  $('btn-share-video').hidden = !video;
  show('result');
  stage?.destroy();
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

const STAMP_LINES = ['YOU\u2019VE BEEN', 'KIRKED'];
const STAMP_ANGLE = -0.14;

/**
 * Bake a red rubber ticket stamp, "YOU'VE BEEN KIRKED", into the image so it survives
 * sharing. Two lines, sized so the whole stamp (rotated) fits in 88% of the image width,
 * with a faint cream backing so it stays readable over red clothes or backgrounds.
 */
async function stampKirked(canvas) {
  const font = '"Rye", Georgia, serif';
  try {
    await document.fonts.load(`400 40px ${font}`);
  } catch { /* fall back to Georgia */ }

  const W = canvas.width;
  const H = canvas.height;
  const stamp = document.createElement('canvas');
  const sctx = stamp.getContext('2d');
  const layout = (size) => {
    sctx.font = `400 ${size}px ${font}`;
    const textW = Math.max(...STAMP_LINES.map((l) => sctx.measureText(l).width));
    const lineH = size * 1.12;
    const w = Math.ceil(textW + size * 1.9); // room for the notches
    const h = Math.ceil(lineH * STAMP_LINES.length + size * 1.0);
    const c = Math.abs(Math.cos(STAMP_ANGLE)), s = Math.abs(Math.sin(STAMP_ANGLE));
    return { size, lineH, w, h, rotW: w * c + h * s, rotH: w * s + h * c };
  };
  let L = layout(Math.round(Math.min(W, H) * 0.095));
  while (L.rotW > W * 0.88 && L.size > 8) L = layout(L.size - 1);
  stamp.width = L.w;
  stamp.height = L.h;
  const { size } = L;

  const ink = '#c0141c';
  const lw = Math.max(2, size * 0.09);
  const notch = Math.min(stamp.height * 0.16, size * 0.42);
  sctx.fillStyle = 'rgba(239, 225, 192, 0.6)';
  ticketPath(sctx, lw, lw, stamp.width - lw * 2, stamp.height - lw * 2, notch);
  sctx.fill();
  sctx.strokeStyle = ink;
  sctx.fillStyle = ink;
  sctx.lineWidth = lw;
  sctx.stroke();
  sctx.lineWidth = lw * 0.5;
  ticketPath(sctx, lw * 2.8, lw * 2.8, stamp.width - lw * 5.6, stamp.height - lw * 5.6, notch * 0.7);
  sctx.stroke();
  sctx.font = `400 ${size}px ${font}`;
  sctx.textAlign = 'center';
  sctx.textBaseline = 'middle';
  const top = stamp.height / 2 - (L.lineH * (STAMP_LINES.length - 1)) / 2 + size * 0.05;
  STAMP_LINES.forEach((line, i) => sctx.fillText(line, stamp.width / 2, top + i * L.lineH));

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
  ctx.translate(W * 0.5, H - L.rotH / 2 - H * 0.035);
  ctx.rotate(STAMP_ANGLE);
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

/** Share a file with the Web Share API, or download it where files can't be shared. */
async function shareFile(blob, url, name, title) {
  const file = new File([blob], name, { type: blob.type });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title });
      return;
    } catch (err) {
      if (err.name === 'AbortError') return; // user closed the share sheet
      console.warn('Share failed, downloading instead', err);
    }
  }
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function shareResult() {
  if (resultBlob) shareFile(resultBlob, resultUrl, 'the-new-attraction.jpg', 'Presenting: the new attraction');
}

function shareVideo() {
  if (!videoBlob) return;
  const ext = videoBlob.type.includes('mp4') ? 'mp4' : 'webm';
  shareFile(videoBlob, videoUrl, `the-transformation.${ext}`, 'The transformation');
}

$('btn-begin').addEventListener('click', openCamera);
$('btn-take').addEventListener('click', takePhoto);
$('btn-flash').addEventListener('click', cycleFlashMode);
updateFlashButton();
$('btn-retake').addEventListener('click', retake);
$('btn-use').addEventListener('click', usePhoto);
$('btn-again').addEventListener('click', openCamera);
$('btn-retry').addEventListener('click', openCamera);
$('btn-share').addEventListener('click', shareResult);
$('btn-share-video').addEventListener('click', shareVideo);

// Release the camera if the page is backgrounded mid-capture.
window.addEventListener('pagehide', () => {
  stopWatching();
  stopCamera();
});
