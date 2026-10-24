// Face swap. The main path is an AI face-swap model on Replicate, reached through the
// site's Netlify Functions (swapFaceRemote). The fallback runs on-device (swapFace):
// MediaPipe Face Landmarker + piecewise-affine triangle warp, frequency separation
// (source detail on the selfie's skin and lighting) and multi-band blending.
// MediaPipe also checks there is a face in the selfie before anything is uploaded (hasFace).

import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { triangulate } from './delaunay.js';
import { boxBlur, multiBandBlend, rgbaToRgb, transferDetail } from './blend.js';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/** How strongly the source face replaces the selfie: 0 = untouched selfie, 1 = full swap. */
export const BLEND_STRENGTH = 1.0;

/**
 * How much the source face bends to fit the selfie's features.
 * 0 = the source is only rotated, scaled and moved onto the head, keeping its own
 * proportions (the "kirkified" look); 1 = every landmark is pulled onto the selfie's
 * landmarks (a faithful warp that ends up looking like the selfie person).
 */
const SHAPE_BLEND = 0.35;

/**
 * Frequency separation: the swapped face is the selfie's skin blurred by this radius
 * (fraction of face width) plus the source's detail above that scale. Larger keeps more
 * of the source's broad shading; smaller takes more lighting from the selfie.
 */
const LOWFREQ_RADIUS = 0.06;

/** Strength of the source's features and texture laid over the selfie's skin (1 = as-is). */
const DETAIL_STRENGTH = 1.0;

/**
 * Extra scale for the detail layer inside the eyebrows only: above 1 makes the brows
 * darker and stronger, below 1 softens them.
 */
const BROW_DARKEN = 1.0;

/** How far the pasted region extends above the top of the face oval, as a fraction of face height. */
const FOREHEAD_EXTENSION = 0.1;

/**
 * Distance over which the top of the mask fades out, as a fraction of face height.
 * The fade reaches zero at the top-of-forehead landmark (whichever of the source's or
 * the selfie's sits lower), so the edge dissolves on forehead skin, below either hairline.
 */
const TOP_FADE = 0.2;

/**
 * Width of the soft edge along the cheeks and jaw, as a fraction of face width.
 * The mask is eroded by this much and then blurred, so it reaches zero at the outline.
 */
const FEATHER = 0.08;

/** Maximum number of pyramid levels for multi-band blending. */
const PYRAMID_LEVELS = 6;

// ---------------------------------------------------------------------------

const SOURCE_FACE_URL = 'assets/source-face.jpg';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const MAX_SOURCE_SIZE = 1024;
const MESH_POINTS = 468; // the face mesh proper; the last 10 landmarks are irises

// Face outline, in order around the face (MediaPipe face-oval landmarks).
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const FOREHEAD = 10;
const CHIN = 152;
const LEFT_EYE = [33, 133, 159, 145];
const RIGHT_EYE = [362, 263, 386, 374];

const LEFT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
const RIGHT_EYE_CONTOUR = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];
const LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
const LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
// Upper lids: from the top of each eye up to the lower edge of its brow (catches eyeshadow).
const LEFT_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133, 55, 65, 52, 53, 46];
const RIGHT_LID = [263, 466, 388, 387, 386, 385, 384, 398, 362, 285, 295, 282, 283, 276];

/** How far the feature mask is grown beyond the landmark contours, as a fraction of face width. */
const FEATURE_DILATE = 0.015;

// Remote (AI) swap.
const SWAP_START_URL = '/.netlify/functions/swap-start';
const SWAP_STATUS_URL = '/.netlify/functions/swap-status';
const UPLOAD_MAX_SIZE = 1024; // longest side of the uploaded selfie, in px
const UPLOAD_QUALITY = 0.85;
// Replicate recommends data URLs only for small files; step the quality down if a selfie comes out larger.
const UPLOAD_TARGET_BYTES = 256 * 1024;
const POLL_MS = 1500;
const REMOTE_TIMEOUT_MS = 45000;

export class FaceSwapError extends Error {
  constructor(kind, cause) {
    super(kind);
    this.kind = kind; // 'model' | 'source-missing' | 'source-no-face' | 'remote'
    this.cause = cause;
  }
}

let landmarkerPromise = null;
let initPromise = null;
let landmarker = null;
let source = null; // { canvas, uvs, triangles }
let warper = null;

/** Load the face landmark model. Safe to call repeatedly. */
function initLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = createLandmarker().then((lm) => { landmarker = lm; });
    landmarkerPromise.catch(() => { landmarkerPromise = null; }); // allow a later retry
  }
  return landmarkerPromise;
}

/** Load the model and precompute the source face for the on-device swap. Safe to call repeatedly. */
export function initFaceSwap() {
  if (!initPromise) {
    initPromise = doInit();
    initPromise.catch(() => { initPromise = null; }); // allow a later retry
  }
  return initPromise;
}

async function doInit() {
  const [, img] = await Promise.all([initLandmarker(), loadSourceImage()]);

  const scale = Math.min(1, MAX_SOURCE_SIZE / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);

  const landmarks = detect(canvas);
  if (!landmarks) throw new FaceSwapError('source-no-face');

  const mesh = landmarks.map((p) => [p.x * canvas.width, p.y * canvas.height]);
  const { ring, maskOutline } = buildSourceExtras(mesh);
  const points = mesh.concat(ring);
  source = {
    canvas,
    points,
    uvs: points.map(([x, y]) => [x / canvas.width, y / canvas.height]),
    triangles: triangulate(points),
    maskOutline,
  };
  warper = createGLWarper(canvas) || create2DWarper(canvas);
}

/**
 * The face mesh stops at the top of the forehead, but the hairline is a big part of
 * what makes a face recognisable. Add an outer ring of points around the face oval,
 * pushed further out at the top, so the warp also carries the source's forehead and
 * hair; plus the matching extended outline used for the blend mask.
 */
function buildSourceExtras(mesh) {
  const top = mesh[FOREHEAD];
  const chin = mesh[CHIN];
  const h = Math.hypot(top[0] - chin[0], top[1] - chin[1]);
  const up = [(top[0] - chin[0]) / h, (top[1] - chin[1]) / h];
  const [cx, cy] = centroid(FACE_OVAL.map((i) => mesh[i]));

  const ring = [];
  const maskOutline = [];
  for (const i of FACE_OVAL) {
    const [x, y] = mesh[i];
    const len = Math.hypot(x - cx, y - cy) || 1;
    const out = [(x - cx) / len, (y - cy) / len];
    const upness = Math.max(0, out[0] * up[0] + out[1] * up[1]); // 1 at the top, 0 at the sides and below
    const lift = FOREHEAD_EXTENSION * h * upness;
    maskOutline.push([x + up[0] * lift, y + up[1] * lift]);
    // The ring sits a little beyond the mask outline so the warp covers the feathered edge.
    const pad = 0.12 * h;
    ring.push([x + up[0] * lift * 1.3 + out[0] * pad, y + up[1] * lift * 1.3 + out[1] * pad]);
  }
  return { ring, maskOutline };
}

function centroid(pts) {
  let x = 0, y = 0;
  for (const p of pts) { x += p[0]; y += p[1]; }
  return [x / pts.length, y / pts.length];
}

/** Eye centres plus forehead and chin: the anchors for the similarity transform. */
function anchors(pts) {
  return [centroid(LEFT_EYE.map((i) => pts[i])), centroid(RIGHT_EYE.map((i) => pts[i])), pts[FOREHEAD], pts[CHIN]];
}

/**
 * Least-squares similarity transform (rotation, uniform scale, translation) mapping
 * `from` onto `to`. Returns a function that applies it to a point.
 */
function fitSimilarity(from, to) {
  const [fx, fy] = centroid(from);
  const [tx, ty] = centroid(to);
  let num = 0, cross = 0, den = 0;
  for (let i = 0; i < from.length; i++) {
    const ax = from[i][0] - fx, ay = from[i][1] - fy;
    const bx = to[i][0] - tx, by = to[i][1] - ty;
    num += ax * bx + ay * by;
    cross += ax * by - ay * bx;
    den += ax * ax + ay * ay;
  }
  const a = num / den; // scale * cos(angle)
  const b = cross / den; // scale * sin(angle)
  return ([x, y]) => [a * (x - fx) - b * (y - fy) + tx, b * (x - fx) + a * (y - fy) + ty];
}

async function createLandmarker() {
  try {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    // CPU is the most reliable delegate on iOS Safari, and a single still image
    // runs in well under 100ms on mid-range phones.
    return await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      runningMode: 'IMAGE',
      numFaces: 1,
    });
  } catch (err) {
    throw new FaceSwapError('model', err);
  }
}

function loadSourceImage() {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(new FaceSwapError('source-missing', err));
    img.src = SOURCE_FACE_URL;
  });
}

function detect(canvas) {
  const result = landmarker.detect(canvas);
  const face = result.faceLandmarks?.[0];
  return face && face.length >= MESH_POINTS ? face.slice(0, MESH_POINTS) : null;
}

/** True if MediaPipe finds a face in the image. Runs before upload, so faceless photos cost nothing. */
export async function hasFace(canvas) {
  await initLandmarker();
  return detect(canvas) !== null;
}

/** The 468 face-mesh landmarks of the image in pixel coordinates, or null if there's no face. */
export async function detectFace(canvas) {
  await initLandmarker();
  const landmarks = detect(canvas);
  return landmarks ? landmarks.map((p) => [p.x * canvas.width, p.y * canvas.height]) : null;
}

/**
 * Swap the face with the AI model: upload the selfie, poll until the prediction is done
 * (or REMOTE_TIMEOUT_MS passes), then download the result into a canvas.
 * Throws FaceSwapError('remote') on any failure, so the caller can fall back to swapFace.
 * @param {HTMLCanvasElement} selfie
 * @returns {Promise<HTMLCanvasElement>}
 */
export async function swapFaceRemote(selfie) {
  const deadline = Date.now() + REMOTE_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('Timed out')), REMOTE_TIMEOUT_MS);
  const { signal } = controller;
  try {
    const start = await fetchJson(SWAP_START_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: encodeForUpload(selfie) }),
      signal,
    });
    if (typeof start.id !== 'string') throw new Error('No prediction id');

    let outputUrl = null;
    while (!outputUrl) {
      await sleep(POLL_MS, signal);
      if (Date.now() > deadline) throw new Error('Timed out');
      const { status, outputUrl: url } = await fetchJson(`${SWAP_STATUS_URL}?id=${encodeURIComponent(start.id)}`, { signal });
      if (status === 'succeeded' && url) outputUrl = url;
      else if (status === 'failed' || status === 'canceled' || status === 'succeeded') throw new Error(`Prediction ${status}`);
    }

    // Fetch into a blob rather than pointing an <img> at the URL, so the canvas isn't
    // cross-origin tainted and Save/Share gets a real file.
    const res = await fetch(outputUrl, { signal });
    if (!res.ok) throw new Error(`Output download failed (${res.status})`);
    const bitmap = await createImageBitmap(await res.blob());
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close?.();
    return canvas;
  } catch (err) {
    throw new FaceSwapError('remote', err);
  } finally {
    clearTimeout(timer);
  }
}

/** The selfie as a JPEG data URL, at most UPLOAD_MAX_SIZE px on its long side. */
function encodeForUpload(selfie) {
  const scale = Math.min(1, UPLOAD_MAX_SIZE / Math.max(selfie.width, selfie.height));
  let canvas = selfie;
  if (scale < 1) {
    canvas = document.createElement('canvas');
    canvas.width = Math.round(selfie.width * scale);
    canvas.height = Math.round(selfie.height * scale);
    canvas.getContext('2d').drawImage(selfie, 0, 0, canvas.width, canvas.height);
  }
  let url = canvas.toDataURL('image/jpeg', UPLOAD_QUALITY);
  for (let q = UPLOAD_QUALITY - 0.1; q >= 0.55 && url.length * 0.75 > UPLOAD_TARGET_BYTES; q -= 0.1) {
    url = canvas.toDataURL('image/jpeg', q);
  }
  return url;
}

async function fetchJson(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`${url} responded ${res.status}`);
  return res.json();
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    const t = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(t); reject(signal.reason); }, { once: true });
  });
}

/**
 * Swap the source face onto the selfie on-device (the fallback when the AI swap fails).
 * @param {HTMLCanvasElement} selfie
 * @returns {Promise<HTMLCanvasElement|null>} the composited image, or null if no face was found
 */
export async function swapFace(selfie) {
  await initFaceSwap();
  const W = selfie.width;
  const H = selfie.height;

  const landmarks = detect(selfie);
  if (!landmarks) return null;
  const dst = landmarks.map((p) => [p.x * W, p.y * H]);
  const src = source.points;

  // Place the source face on the head with one rotation + uniform scale + translation,
  // then let each landmark drift SHAPE_BLEND of the way toward the selfie's landmark.
  const sim = fitSimilarity(anchors(src), anchors(dst));
  const targets = new Array(src.length);
  for (let i = 0; i < MESH_POINTS; i++) {
    const [sx, sy] = sim(src[i]);
    targets[i] = [sx + (dst[i][0] - sx) * SHAPE_BLEND, sy + (dst[i][1] - sy) * SHAPE_BLEND];
  }
  // Ring and mask points have no selfie counterpart: they follow their oval landmark.
  const follow = (p, k) => {
    const i = FACE_OVAL[k];
    const [px, py] = sim(p);
    const [ox, oy] = sim(src[i]);
    return [px + targets[i][0] - ox, py + targets[i][1] - oy];
  };
  FACE_OVAL.forEach((_, k) => { targets[MESH_POINTS + k] = follow(src[MESH_POINTS + k], k); });
  const outline = source.maskOutline.map(follow);
  const oval = FACE_OVAL.map((i) => targets[i]);

  // Work only inside a box around the pasted region to keep per-pixel work small.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of outline) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const faceW = maxX - minX;
  // Coarse pyramid levels spread colour well beyond the mask, so leave room for them.
  const pad = faceW * 0.3;
  const bx = Math.max(0, Math.floor(minX - pad));
  const by = Math.max(0, Math.floor(minY - pad));
  const bw = Math.min(W, Math.ceil(maxX + pad)) - bx;
  const bh = Math.min(H, Math.ceil(maxY + pad)) - by;
  if (bw < 8 || bh < 8) return null;

  const warped = warper.render(targets, W, H);

  const out = document.createElement('canvas');
  out.width = W;
  out.height = H;
  const octx = out.getContext('2d', { willReadFrequently: true });
  octx.drawImage(selfie, 0, 0);
  const selfieData = octx.getImageData(bx, by, bw, bh);

  const crop = document.createElement('canvas');
  crop.width = bw;
  crop.height = bh;
  const cctx = crop.getContext('2d', { willReadFrequently: true });
  cctx.drawImage(warped, bx, by, bw, bh, 0, 0, bw, bh);
  const warpedData = cctx.getImageData(0, 0, bw, bh);

  const n = bw * bh;
  const coverage = new Float32Array(n);
  for (let p = 0; p < n; p++) coverage[p] = warpedData.data[p * 4 + 3] / 255;

  // Fade along the face's own vertical axis, reaching 0 at the lower of the two
  // top-of-forehead landmarks so neither the source's hair nor the selfie's hairline is covered.
  const fade = topFade(targets[FOREHEAD], dst[FOREHEAD], targets[CHIN], bx, by, bw, bh);

  // Frequency separation: the selfie's skin tone and lighting (low frequencies) plus the
  // source's features and texture (high frequencies). Both low-pass layers are masked,
  // normalised blurs, blur(image * (1 - features)) / blur(1 - features), restricted to the
  // face, so neither person's brows, eyes, lids or lips feed into them: skin colour is filled in
  // underneath, the source's features keep their full darkness in the detail layer, and the
  // selfie's own features can't ghost through.
  const dilate = Math.max(2, faceW * FEATURE_DILATE);
  const selfieFeatures = featureMask(dst, bx, by, bw, bh, dilate);
  const sourceFeatures = featureMask(targets, bx, by, bw, bh, dilate);
  const selfieSkin = fillPolygon(shrink(FACE_OVAL.map((i) => dst[i]), 0.92), bx, by, bw, bh);
  const sourceSkin = fillPolygon(shrink(oval, 0.92), bx, by, bw, bh);
  for (let p = 0; p < n; p++) {
    selfieSkin[p] *= 1 - selfieFeatures.all[p];
    sourceSkin[p] *= (1 - sourceFeatures.all[p]) * (coverage[p] === 1 ? 1 : 0);
  }
  const radius = Math.max(2, Math.round(faceW * LOWFREQ_RADIUS));
  // Inside the eye openings the source's own colours are kept, so the whites stay white
  // on any skin tone.
  const eyes = fillPolygons([LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR].map((c) => c.map((i) => targets[i])), bx, by, bw, bh);
  boxBlur(eyes, bw, bh, Math.max(1, Math.round(faceW * 0.006)));
  const face = transferDetail(selfieData.data, warpedData.data, {
    selfieSkin, sourceSkin, eyes, brows: sourceFeatures.brows, w: bw, h: bh, radius,
    detailStrength: DETAIL_STRENGTH, browDarken: BROW_DARKEN,
  });

  // The foreground image is the new face only inside the outline and the selfie elsewhere,
  // so the wide, coarse pyramid levels can't drag the source's hair or background in.
  const selfieRgb = rgbaToRgb(selfieData.data);
  const content = fillPolygon(outline, bx, by, bw, bh);
  for (let p = 0; p < n; p++) {
    const a = content[p] * fade[p] * coverage[p];
    for (let c = 0; c < 3; c++) {
      const i = p * 3 + c;
      face[i] = selfieRgb[i] + (face[i] - selfieRgb[i]) * a;
    }
  }

  const mask = buildMask(outline, bx, by, bw, bh, faceW);
  for (let p = 0; p < n; p++) mask[p] *= fade[p] * coverage[p] * BLEND_STRENGTH;

  const levels = Math.max(1, Math.min(PYRAMID_LEVELS, Math.floor(Math.log2(Math.min(bw, bh) / 8))));
  const blended = multiBandBlend(face, selfieRgb, mask, bw, bh, levels);

  // Ease any low-frequency spill back to the untouched selfie at crop edges inside the image.
  const ramp = Math.max(1, pad * 0.5);
  const rampL = bx > 0 ? ramp : 0, rampT = by > 0 ? ramp : 0;
  const rampR = bx + bw < W ? ramp : 0, rampB = by + bh < H ? ramp : 0;
  const edge = (d, r) => (r ? Math.min(1, d / r) : 1);
  const px = selfieData.data;
  for (let y = 0; y < bh; y++) {
    const ey = Math.min(edge(y, rampT), edge(bh - 1 - y, rampB));
    for (let x = 0; x < bw; x++) {
      const e = Math.min(ey, edge(x, rampL), edge(bw - 1 - x, rampR));
      const p = y * bw + x;
      for (let c = 0; c < 3; c++) {
        const i = p * 3 + c;
        px[p * 4 + c] = selfieRgb[i] + (blended[i] - selfieRgb[i]) * e; // Uint8Clamped clamps
      }
    }
  }

  octx.putImageData(selfieData, bx, by);
  return out;
}

/** Scale a polygon about its centroid. */
function shrink(poly, k) {
  const [cx, cy] = centroid(poly);
  return poly.map(([x, y]) => [cx + (x - cx) * k, cy + (y - cy) * k]);
}

/** Union of several polygons as a 0..1 coverage map over the crop box. */
function fillPolygons(polys, bx, by, bw, bh) {
  const c = document.createElement('canvas');
  c.width = bw;
  c.height = bh;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff';
  for (const poly of polys) {
    ctx.beginPath();
    poly.forEach(([x, y], i) => (i ? ctx.lineTo(x - bx, y - by) : ctx.moveTo(x - bx, y - by)));
    ctx.closePath();
    ctx.fill();
  }
  const rgba = ctx.getImageData(0, 0, bw, bh).data;
  const a = new Float32Array(bw * bh);
  for (let i = 0; i < a.length; i++) a[i] = rgba[i * 4 + 3] / 255;
  return a;
}

/**
 * Soft masks (0..1) over the crop box for the brows, eyes, upper lids and lips: each landmark contour
 * is filled and stroked with a round pen to grow it by `dilate` px, then feathered.
 * Returns the union of all features and the brows on their own.
 */
function featureMask(pts, bx, by, bw, bh, dilate) {
  const draw = (contours) => {
    const c = document.createElement('canvas');
    c.width = bw;
    c.height = bh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = ctx.strokeStyle = '#fff';
    ctx.lineWidth = dilate * 2;
    ctx.lineJoin = ctx.lineCap = 'round';
    for (const idx of contours) {
      ctx.beginPath();
      idx.forEach((i, k) => (k ? ctx.lineTo(pts[i][0] - bx, pts[i][1] - by) : ctx.moveTo(pts[i][0] - bx, pts[i][1] - by)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    const rgba = ctx.getImageData(0, 0, bw, bh).data;
    const a = new Float32Array(bw * bh);
    for (let i = 0; i < a.length; i++) a[i] = rgba[i * 4 + 3] / 255;
    return boxBlur(a, bw, bh, Math.max(1, Math.round(dilate / 2)), 2);
  };
  return {
    all: draw([LEFT_BROW, RIGHT_BROW, LEFT_LID, RIGHT_LID, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, LIPS]),
    brows: draw([LEFT_BROW, RIGHT_BROW]),
  };
}

/**
 * Per-pixel weight that is 1 well inside the face and eases to 0 at the lower of the
 * two forehead points, measured along the chin-to-forehead axis, over TOP_FADE of the
 * face height.
 */
function topFade(forehead, selfieForehead, chin, bx, by, bw, bh) {
  const h = Math.hypot(forehead[0] - chin[0], forehead[1] - chin[1]) || 1;
  const ux = (forehead[0] - chin[0]) / h, uy = (forehead[1] - chin[1]) / h;
  const height = (p) => (p[0] - chin[0]) * ux + (p[1] - chin[1]) * uy;
  const top = height(selfieForehead) < height(forehead) ? selfieForehead : forehead;
  const span = TOP_FADE * h;
  const out = new Float32Array(bw * bh);
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const below = -((x + bx - top[0]) * ux + (y + by - top[1]) * uy); // distance below the top
      const t = Math.min(1, Math.max(0, below / span));
      out[y * bw + x] = t * t * (3 - 2 * t);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Mask, colour matching and blending
// ---------------------------------------------------------------------------

/** Rasterise a polygon (image coordinates) into a 0..1 coverage map over the crop box. */
function fillPolygon(poly, bx, by, bw, bh) {
  const c = document.createElement('canvas');
  c.width = bw;
  c.height = bh;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  poly.forEach(([x, y], i) => (i ? ctx.lineTo(x - bx, y - by) : ctx.moveTo(x - bx, y - by)));
  ctx.closePath();
  ctx.fill();
  const rgba = ctx.getImageData(0, 0, bw, bh).data;
  const a = new Float32Array(bw * bh);
  for (let i = 0; i < a.length; i++) a[i] = rgba[i * 4 + 3] / 255;
  return a;
}

/**
 * Soft mask (0..1 per pixel) over the crop box: the outline is eroded by half the
 * feather width and then blurred, so the edge ramps from 1 inside the face to 0
 * exactly at the outline, with no hard line along the cheeks or jaw.
 */
function buildMask(outline, bx, by, bw, bh, faceW) {
  // Three box-blur passes of radius q spread an edge over +-3q (close to a Gaussian;
  // ctx.filter is unreliable on iOS), so erode by 3q.
  const q = Math.max(1, Math.round((faceW * FEATHER) / 6));
  const erode = 3 * q;

  const [cx, cy] = centroid(outline);
  const inset = outline.map(([x, y]) => {
    const dx = x - cx, dy = y - cy;
    const k = Math.max(0, 1 - erode / (Math.hypot(dx, dy) || 1));
    return [cx + dx * k, cy + dy * k];
  });
  const a = fillPolygon(inset, bx, by, bw, bh);

  return boxBlur(a, bw, bh, q, 3);
}

// ---------------------------------------------------------------------------
// Triangle-mesh warpers: draw each source triangle onto the matching selfie triangle.
// ---------------------------------------------------------------------------

const VERT = `
attribute vec2 aPos;
attribute vec2 aUv;
uniform vec2 uRes;
varying vec2 vUv;
void main() {
  vec2 clip = aPos / uRes * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vUv = aUv;
}`;

const FRAG = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D uTex;
varying vec2 vUv;
void main() { gl_FragColor = texture2D(uTex, vUv); }`;

function createGLWarper(sourceCanvas) {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true, antialias: true });
  if (!gl) return null;

  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
  };
  const vs = compile(gl.VERTEX_SHADER, VERT);
  const fs = compile(gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
  gl.useProgram(prog);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourceCanvas);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  const aPos = gl.getAttribLocation(prog, 'aPos');
  const aUv = gl.getAttribLocation(prog, 'aUv');
  gl.enableVertexAttribArray(aPos);
  gl.enableVertexAttribArray(aUv);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
  gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8);
  const uRes = gl.getUniformLocation(prog, 'uRes');

  const { triangles, uvs } = source;
  const verts = new Float32Array(triangles.length * 4);

  return {
    render(dst, W, H) {
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W;
        canvas.height = H;
      }
      for (let t = 0; t < triangles.length; t++) {
        const idx = triangles[t];
        verts[t * 4] = dst[idx][0];
        verts[t * 4 + 1] = dst[idx][1];
        verts[t * 4 + 2] = uvs[idx][0];
        verts[t * 4 + 3] = uvs[idx][1];
      }
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uRes, W, H);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, triangles.length);
      return canvas;
    },
  };
}

/** Canvas 2D fallback for devices without WebGL: clip + affine transform per triangle. */
function create2DWarper(sourceCanvas) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const { triangles, points: src } = source;

  return {
    render(dst, W, H) {
      canvas.width = W;
      canvas.height = H;
      for (let t = 0; t < triangles.length; t += 3) {
        const [i0, i1, i2] = [triangles[t], triangles[t + 1], triangles[t + 2]];
        const [s0, s1, s2] = [src[i0], src[i1], src[i2]];
        const [d0, d1, d2] = [dst[i0], dst[i1], dst[i2]];

        const ux = s1[0] - s0[0], uy = s1[1] - s0[1];
        const vx = s2[0] - s0[0], vy = s2[1] - s0[1];
        const det = ux * vy - vx * uy;
        if (Math.abs(det) < 1e-6) continue;
        const dx1 = d1[0] - d0[0], dy1 = d1[1] - d0[1];
        const dx2 = d2[0] - d0[0], dy2 = d2[1] - d0[1];
        const a = (dx1 * vy - dx2 * uy) / det;
        const c = (ux * dx2 - vx * dx1) / det;
        const b = (dy1 * vy - dy2 * uy) / det;
        const d = (ux * dy2 - vx * dy1) / det;
        const e = d0[0] - a * s0[0] - c * s0[1];
        const f = d0[1] - b * s0[0] - d * s0[1];

        // Grow the clip triangle by ~0.5px to hide hairline seams between triangles.
        const mx = (d0[0] + d1[0] + d2[0]) / 3, my = (d0[1] + d1[1] + d2[1]) / 3;
        const grow = (p) => {
          const len = Math.hypot(p[0] - mx, p[1] - my) || 1;
          const k = 1 + 0.5 / len;
          return [mx + (p[0] - mx) * k, my + (p[1] - my) * k];
        };

        ctx.save();
        ctx.beginPath();
        [d0, d1, d2].map(grow).forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.clip();
        ctx.setTransform(a, b, c, d, e, f);
        ctx.drawImage(sourceCanvas, 0, 0);
        ctx.restore();
      }
      return canvas;
    },
  };
}
