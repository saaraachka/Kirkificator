// Face swap: MediaPipe Face Landmarker + piecewise-affine triangle warp,
// colour matching and a feathered blend. Everything runs on-device.

import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { triangulate } from './delaunay.js';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/** How strongly the source face replaces the selfie: 0 = untouched selfie, 1 = full swap. */
export const BLEND_STRENGTH = 0.85;

/** Width of the soft edge around the swapped face, as a fraction of face width. */
const FEATHER = 0.08;

/** Limits on how far colour matching may stretch contrast per channel. */
const MIN_CONTRAST_GAIN = 0.6;
const MAX_CONTRAST_GAIN = 1.6;

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

export class FaceSwapError extends Error {
  constructor(kind, cause) {
    super(kind);
    this.kind = kind; // 'model' | 'source-missing' | 'source-no-face'
    this.cause = cause;
  }
}

let initPromise = null;
let landmarker = null;
let source = null; // { canvas, uvs, triangles }
let warper = null;

/** Load the model and precompute the source face. Safe to call repeatedly. */
export function initFaceSwap() {
  if (!initPromise) {
    initPromise = doInit();
    initPromise.catch(() => { initPromise = null; }); // allow a later retry
  }
  return initPromise;
}

async function doInit() {
  const [lm, img] = await Promise.all([createLandmarker(), loadSourceImage()]);
  landmarker = lm;

  const scale = Math.min(1, MAX_SOURCE_SIZE / Math.max(img.naturalWidth, img.naturalHeight));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);

  const landmarks = detect(canvas);
  if (!landmarks) throw new FaceSwapError('source-no-face');

  const points = landmarks.map((p) => [p.x * canvas.width, p.y * canvas.height]);
  source = {
    canvas,
    points,
    uvs: landmarks.map((p) => [p.x, p.y]),
    triangles: triangulate(points),
  };
  warper = createGLWarper(canvas) || create2DWarper(canvas);
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

/**
 * Swap the source face onto the selfie.
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

  // Work only inside a padded box around the face to keep per-pixel work small.
  const oval = FACE_OVAL.map((i) => dst[i]);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of oval) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const faceW = maxX - minX;
  const pad = faceW * 0.1;
  const bx = Math.max(0, Math.floor(minX - pad));
  const by = Math.max(0, Math.floor(minY - pad));
  const bw = Math.min(W, Math.ceil(maxX + pad)) - bx;
  const bh = Math.min(H, Math.ceil(maxY + pad)) - by;
  if (bw < 8 || bh < 8) return null;

  const warped = warper.render(dst, W, H);

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

  const mask = buildMask(oval, bx, by, bw, bh, faceW);
  colorMatchAndBlend(selfieData.data, warpedData.data, mask);

  octx.putImageData(selfieData, bx, by);
  return out;
}

// ---------------------------------------------------------------------------
// Mask, colour matching and blending
// ---------------------------------------------------------------------------

/** Feathered face-oval mask (0..1 per pixel) over the crop box. */
function buildMask(oval, bx, by, bw, bh, faceW) {
  const r = Math.max(1, Math.round((faceW * FEATHER) / 2));

  // Inset the outline by the blur radius so the soft edge fades out at the face boundary.
  let cx = 0, cy = 0;
  for (const [x, y] of oval) { cx += x; cy += y; }
  cx /= oval.length;
  cy /= oval.length;

  const c = document.createElement('canvas');
  c.width = bw;
  c.height = bh;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  oval.forEach(([x, y], i) => {
    const dx = x - cx, dy = y - cy;
    const k = Math.max(0, 1 - r / (Math.hypot(dx, dy) || 1));
    const px = cx + dx * k - bx;
    const py = cy + dy * k - by;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fill();

  const rgba = ctx.getImageData(0, 0, bw, bh).data;
  let a = new Float32Array(bw * bh);
  for (let i = 0; i < a.length; i++) a[i] = rgba[i * 4 + 3] / 255;

  // Two separable box-blur passes approximate a Gaussian (ctx.filter is unreliable on iOS).
  let b = new Float32Array(a.length);
  const half = Math.max(1, Math.round(r / 2));
  for (let pass = 0; pass < 2; pass++) {
    blurLine(a, b, bw, bh, half, true);
    blurLine(b, a, bw, bh, half, false);
  }
  return a;
}

function blurLine(src, dst, w, h, r, horizontal) {
  const len = horizontal ? w : h;
  const lines = horizontal ? h : w;
  const step = horizontal ? 1 : w;
  const lineStep = horizontal ? w : 1;
  const norm = 1 / (2 * r + 1);
  for (let l = 0; l < lines; l++) {
    const base = l * lineStep;
    let acc = 0;
    for (let k = 0; k <= r && k < len; k++) acc += src[base + k * step];
    for (let k = 0; k < len; k++) {
      dst[base + k * step] = acc * norm;
      const add = k + r + 1;
      const sub = k - r;
      if (add < len) acc += src[base + add * step];
      if (sub >= 0) acc -= src[base + sub * step];
    }
  }
}

/**
 * Match the warped face's per-channel mean/std to the selfie's within the face
 * region, then alpha-blend it into `selfie` in place.
 */
function colorMatchAndBlend(selfie, warped, mask) {
  const sumS = [0, 0, 0], sqS = [0, 0, 0];
  const sumW = [0, 0, 0], sqW = [0, 0, 0];
  let n = 0;
  for (let p = 0, i = 0; p < mask.length; p++, i += 4) {
    if (mask[p] < 0.5 || warped[i + 3] < 250) continue;
    for (let c = 0; c < 3; c++) {
      const s = selfie[i + c], w = warped[i + c];
      sumS[c] += s; sqS[c] += s * s;
      sumW[c] += w; sqW[c] += w * w;
    }
    n++;
  }
  if (n === 0) return;

  const meanS = [], meanW = [], gain = [];
  for (let c = 0; c < 3; c++) {
    meanS[c] = sumS[c] / n;
    meanW[c] = sumW[c] / n;
    const sdS = Math.sqrt(Math.max(0, sqS[c] / n - meanS[c] ** 2));
    const sdW = Math.sqrt(Math.max(0, sqW[c] / n - meanW[c] ** 2));
    gain[c] = Math.min(MAX_CONTRAST_GAIN, Math.max(MIN_CONTRAST_GAIN, sdS / (sdW || 1)));
  }

  for (let p = 0, i = 0; p < mask.length; p++, i += 4) {
    const a = mask[p] * (warped[i + 3] / 255) * BLEND_STRENGTH;
    if (a <= 0.002) continue;
    for (let c = 0; c < 3; c++) {
      const matched = (warped[i + c] - meanW[c]) * gain[c] + meanS[c];
      selfie[i + c] = selfie[i + c] + (matched - selfie[i + c]) * a; // Uint8Clamped clamps
    }
  }
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
