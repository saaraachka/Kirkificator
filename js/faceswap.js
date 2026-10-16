// Face swap: MediaPipe Face Landmarker + piecewise-affine triangle warp,
// Lab colour matching and multi-band blending. Everything runs on-device.

import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { triangulate } from './delaunay.js';
import { colorMatchLab, multiBandBlend, rgbaToRgb } from './blend.js';

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
const SHAPE_BLEND = 0.15;

/**
 * How far the source skin is pulled toward the selfie's colour statistics.
 * 0 = source keeps its own skin tone; 1 = full mean/std match.
 */
const COLOR_MATCH_STRENGTH = 0.65;

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
const FOREHEAD = 10;
const CHIN = 152;
const LEFT_EYE = [33, 133, 159, 145];
const RIGHT_EYE = [362, 263, 386, 374];

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

  // Colour statistics come from the inner face only, away from hair and background.
  const skin = fillPolygon(shrink(oval, 0.8), bx, by, bw, bh);
  const matched = colorMatchLab(
    selfieData.data, warpedData.data, skin, COLOR_MATCH_STRENGTH, MIN_CONTRAST_GAIN, MAX_CONTRAST_GAIN,
  );

  // The foreground image is the source only inside its outline and the selfie elsewhere,
  // so the wide, coarse pyramid levels can't drag the source's hair or background in.
  const selfieRgb = rgbaToRgb(selfieData.data);
  const content = fillPolygon(outline, bx, by, bw, bh);
  for (let p = 0; p < n; p++) {
    const a = content[p] * fade[p] * coverage[p];
    for (let c = 0; c < 3; c++) {
      const i = p * 3 + c;
      matched[i] = selfieRgb[i] + (matched[i] - selfieRgb[i]) * a;
    }
  }

  const mask = buildMask(outline, bx, by, bw, bh, faceW);
  for (let p = 0; p < n; p++) mask[p] *= fade[p] * coverage[p] * BLEND_STRENGTH;

  const levels = Math.max(1, Math.min(PYRAMID_LEVELS, Math.floor(Math.log2(Math.min(bw, bh) / 8))));
  const blended = multiBandBlend(matched, selfieRgb, mask, bw, bh, levels);

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

  const b = new Float32Array(a.length);
  for (let pass = 0; pass < 3; pass++) {
    blurLine(a, b, bw, bh, q, true);
    blurLine(b, a, bw, bh, q, false);
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
