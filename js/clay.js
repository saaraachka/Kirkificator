// Clayface: the loading screen. The selfie's face turns to modelling clay, gets kneaded by
// invisible hands until the swapped photo arrives, then the new face pushes up through
// the clay and hardens into the result.
//
//   Phase 1  TURN TO CLAY  CLAY_IN_MS        the face becomes matte, sculpted clay
//   Phase 2  KNEADING      until B arrives   random landmark-driven molding, min MIN_KNEAD_MS
//   Phase 3  BECOMING      BECOME_MS         mesh-warp A -> B while crossfading the clay
//   Phase 4  HARDEN        HARDEN_MS         deformation settles, the clay fades to the photo
//   then a single snap to the stamped result.
//
// Rendering is WebGL (fragment shaders) with a simpler Canvas 2D fallback. The canvas is
// recorded from PREROLL_MS before phase 3 to the end, for sharing.

import { triangulate } from './delaunay.js';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/** Phase 1: how long the face takes to turn to clay. */
export const CLAY_IN_MS = 1200;
/** Phase 2: the least amount of kneading shown, however fast the swap is. */
export const MIN_KNEAD_MS = 4000;
/** Phase 3: how long the new face takes to push up through the clay. */
export const BECOME_MS = 2600;
/** Phase 4: how long the clay takes to harden and fade into the photo. */
export const HARDEN_MS = 1200;
/** How hard the invisible hands mold the face (1 = default; 0 = no molding). */
export const MOLD_INTENSITY = 1.0;
/** How far the jaw drips when it sags (1 = default; 0 = never drips). */
export const DRIP_AMOUNT = 1.0;
/** Strength of the clay's surface texture: fingerprints, thumb smudges and grain. */
export const CLAY_TEXTURE_STRENGTH = 1.0;

// ---------------------------------------------------------------------------

const PREROLL_MS = 2000; // kneading kept at the start of the recorded clip
const SNAP_MS = 280; // the final snap to the stamped photo
const HOLD_MS = 900; // the stamped photo stays on screen (and in the clip) this long
const MAX_DEFORMERS = 8;
const MAX_DENTS = 4;
const MAX_PIXELS = 1.1e6; // cap on the stage's drawing-buffer size, for frame rate
const VIDEO_TYPES = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];

// MediaPipe face-mesh landmark indices.
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const LM = {
  forehead: 10, midForehead: 151, chin: 152, noseTip: 1, noseBridge: 168,
  leftCheek: 50, rightCheek: 280, leftEdge: 234, rightEdge: 454,
  leftEye: [159, 145, 33, 133], rightEye: [386, 374, 263, 362], mouth: [13, 14, 61, 291],
};
const DENT_SITES = [50, 280, 151, 108, 337, 205, 425, 152, 9, 36, 266];
const PRINT_SITES = [117, 346, 104, 333, 199, 4];

const reducedMotion = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
const MOLD = MOLD_INTENSITY * (reducedMotion ? 0.35 : 1);

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const ease = (x) => { x = clamp01(x); return x * x * (3 - 2 * x); };
const easeInOut = (x) => { x = clamp01(x); return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2; };
const rand = (a, b) => a + Math.random() * (b - a);
const lerpPts = (a, b, t) => a.map((p, i) => [p[0] + (b[i][0] - p[0]) * t, p[1] + (b[i][1] - p[1]) * t]);

/** Face centre, size and axes from the landmarks. `up` points from chin to forehead. */
function faceFrame(L) {
  let cx = 0, cy = 0;
  for (const i of FACE_OVAL) { cx += L[i][0]; cy += L[i][1]; }
  cx /= FACE_OVAL.length;
  cy /= FACE_OVAL.length;
  const w = Math.hypot(L[LM.rightEdge][0] - L[LM.leftEdge][0], L[LM.rightEdge][1] - L[LM.leftEdge][1]);
  const hx = L[LM.forehead][0] - L[LM.chin][0], hy = L[LM.forehead][1] - L[LM.chin][1];
  const h = Math.hypot(hx, hy) || 1;
  const up = [hx / h, hy / h];
  return { cx, cy, w, h, up, right: [-up[1], up[0]] };
}

const avg = (L, idx) => {
  let x = 0, y = 0;
  for (const i of idx) { x += L[i][0]; y += L[i][1]; }
  return [x / idx.length, y / idx.length];
};

// ---------------------------------------------------------------------------
// Choreography: the invisible hands
// ---------------------------------------------------------------------------
//
// Each deformer is { x, y, r, kind, ax, ay, amount, aspect } in image pixels:
//   kind 0  push:   moves material by (ax, ay) * amount
//   kind 1  radial: scales material about (x, y) by (ax, ay) * amount (+ bulges, - squeezes)
//   kind 2  twist:  rotates material about (x, y) by `amount` radians
//   kind 3  drip:   stretches material downward by `amount` px over `aspect` radii, in streaks
// all with a Gaussian falloff of radius r.

const MOVES = {
  cheeks(L, f, s, k) {
    const out = [];
    for (const i of [LM.leftCheek, LM.rightCheek]) {
      const [x, y] = L[i];
      const side = Math.sign((x - f.cx) * f.right[0] + (y - f.cy) * f.right[1]) || 1;
      out.push({ x, y, r: f.w * 0.3, kind: 0, ax: f.right[0] * side, ay: f.right[1] * side, amount: f.w * 0.16 * k });
    }
    return out;
  },
  nose(L, f, s, k) {
    // Pushed in, then pulled out.
    const [x, y] = L[LM.noseTip];
    const a = Math.sin(s * Math.PI * 2 - 0.4) * 0.9 * k;
    return [{ x, y, r: f.w * 0.17, kind: 1, ax: 1, ay: 1, amount: a }];
  },
  eyes(L, f, s, k) {
    return [LM.leftEye, LM.rightEye].map((idx) => {
      const [x, y] = avg(L, idx.slice(0, 2));
      return { x, y, r: f.w * 0.14, kind: 1, ax: 0.35, ay: -0.85, amount: k };
    });
  },
  mouth(L, f, s, k) {
    const [x, y] = avg(L, LM.mouth.slice(0, 2));
    return [{ x, y, r: f.w * 0.24, kind: 1, ax: 0.9, ay: -0.35, amount: k }];
  },
  forehead(L, f, s, k) {
    const [x, y] = L[LM.midForehead];
    return [
      { x, y, r: f.w * 0.32, kind: 0, ax: -f.up[0], ay: -f.up[1], amount: f.w * 0.1 * k },
      { x, y, r: f.w * 0.3, kind: 1, ax: 0.35, ay: -0.4, amount: k },
    ];
  },
  jaw(L, f, s, k, m) {
    const [x, y] = L[LM.chin];
    const sag = { x, y, r: f.w * 0.38, kind: 0, ax: -f.up[0], ay: -f.up[1], amount: f.w * 0.13 * k };
    // The drip comes late in the move and is brief.
    const d = ease((s - 0.35) / 0.3) * (1 - ease((s - 0.75) / 0.25));
    const drip = { x, y: y - f.h * 0.05, r: f.w * 0.26, kind: 3, ax: m.seedA, ay: m.seedB, amount: f.w * 0.32 * DRIP_AMOUNT * k * d, aspect: 1.7 };
    return [sag, drip];
  },
  twist(L, f, s, k, m) {
    const [x, y] = avg(L, [LM.noseBridge, LM.noseTip]);
    return [{ x, y, r: f.w * 0.62, kind: 2, ax: 0, ay: 0, amount: 0.22 * k * m.sign }];
  },
  squash(L, f, s, k, m) {
    return [{ x: f.cx, y: f.cy, r: f.w * 0.7, kind: 1, ax: 0.16 * m.sign, ay: -0.2 * m.sign, amount: k }];
  },
  pinchSide(L, f, s, k, m) {
    const [x, y] = L[m.sign > 0 ? LM.rightEdge : LM.leftEdge];
    const side = m.sign;
    return [{ x, y, r: f.w * 0.28, kind: 0, ax: -f.right[0] * side, ay: -f.right[1] * side, amount: f.w * 0.11 * k },
      { x, y: y + f.h * 0.1, r: f.w * 0.22, kind: 0, ax: f.up[0], ay: f.up[1], amount: f.w * 0.06 * k }];
  },
};
const MOVE_NAMES = Object.keys(MOVES);

class Hands {
  constructor() {
    this.moves = [];
    this.recent = [];
    this.dents = [];
    this.nextDentAt = 0;
  }

  /** Start a random move that isn't one of the last two. */
  spawn(now) {
    const choices = MOVE_NAMES.filter((n) => !this.recent.includes(n));
    const type = choices[(Math.random() * choices.length) | 0];
    this.recent = [type, ...this.recent].slice(0, 2);
    this.moves.push({
      type,
      start: now,
      dur: rand(800, 2000),
      k: rand(0.6, 1.25),
      overlapAt: rand(0.45, 0.85),
      sign: Math.random() < 0.5 ? -1 : 1,
      seedA: rand(0, 6.28),
      seedB: rand(0, 6.28),
    });
  }

  /** Press a thumb somewhere on the face; the dent fills back in over a few seconds. */
  press(now) {
    this.dents.push({ site: DENT_SITES[(Math.random() * DENT_SITES.length) | 0], ox: rand(-0.04, 0.04), oy: rand(-0.04, 0.04), r: rand(0.055, 0.085), depth: rand(0.7, 1.2), start: now });
    if (this.dents.length > MAX_DENTS) this.dents.shift();
  }

  /**
   * Deformers and dents for this frame. `gain` scales everything; `spawning` lets new
   * moves and thumb presses start.
   */
  update(now, L, gain, spawning) {
    const f = faceFrame(L);
    if (spawning) {
      const newest = this.moves[this.moves.length - 1];
      if (!newest || (now - newest.start) / newest.dur > newest.overlapAt) this.spawn(now);
      if (now >= this.nextDentAt) {
        if (this.nextDentAt) this.press(now);
        this.nextDentAt = now + rand(1200, 3000);
      }
    }
    this.moves = this.moves.filter((m) => now - m.start < m.dur);
    const deformers = [];
    for (const m of this.moves) {
      const s = (now - m.start) / m.dur;
      const env = ease(s / 0.35) * (1 - ease((s - 0.6) / 0.4));
      for (const d of MOVES[m.type](L, f, s, m.k * env * gain * MOLD, m)) deformers.push(d);
    }
    this.dents = this.dents.filter((d) => now - d.start < 5000);
    const dents = this.dents.map((d) => {
      const t = (now - d.start) / 1000;
      const depth = d.depth * ease(t / 0.25) * Math.exp(-Math.max(0, t - 0.25) / 1.4) * MOLD;
      const [x, y] = L[d.site];
      return { x: x + d.ox * f.w, y: y + d.oy * f.w, r: d.r * f.w, depth: depth * Math.max(0, Math.min(1, gain)) };
    });
    return { deformers: deformers.slice(0, MAX_DEFORMERS), dents, frame: f };
  }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Soft face mask (white on black) from the oval, lifted a little over the forehead. */
function faceMask(W, H, L) {
  const f = faceFrame(L);
  const small = document.createElement('canvas');
  const k = 1 / 12;
  small.width = Math.max(2, Math.round(W * k));
  small.height = Math.max(2, Math.round(H * k));
  const sctx = small.getContext('2d');
  sctx.fillStyle = '#000';
  sctx.fillRect(0, 0, small.width, small.height);
  sctx.fillStyle = '#fff';
  sctx.beginPath();
  FACE_OVAL.forEach((i, n) => {
    let [x, y] = L[i];
    const dx = x - f.cx, dy = y - f.cy;
    const len = Math.hypot(dx, dy) || 1;
    const upness = Math.max(0, (dx * f.up[0] + dy * f.up[1]) / len);
    x = f.cx + dx * 1.02 + f.up[0] * f.h * 0.1 * upness;
    y = f.cy + dy * 1.02 + f.up[1] * f.h * 0.1 * upness;
    if (n) sctx.lineTo(x * k, y * k); else sctx.moveTo(x * k, y * k);
  });
  sctx.closePath();
  sctx.fill();
  const mask = document.createElement('canvas');
  mask.width = W;
  mask.height = H;
  const mctx = mask.getContext('2d');
  mctx.imageSmoothingEnabled = true;
  mctx.imageSmoothingQuality = 'high';
  mctx.drawImage(small, 0, 0, W, H);
  return mask;
}

/** Landmarks plus a ring around the face and points along the image border, for the morph mesh. */
function morphPoints(L, W, H) {
  const f = faceFrame(L);
  const pts = L.slice();
  for (const i of FACE_OVAL) {
    const [x, y] = L[i];
    pts.push([Math.min(W - 2, Math.max(2, f.cx + (x - f.cx) * 1.4)), Math.min(H - 2, Math.max(2, f.cy + (y - f.cy) * 1.4))]);
  }
  const n = 5;
  for (let i = 0; i <= n; i++) {
    pts.push([(i / n) * W, 0], [(i / n) * W, H]);
    if (i > 0 && i < n) pts.push([0, (i / n) * H], [W, (i / n) * H]);
  }
  return pts;
}

/** Cover-fit mapping from the image onto a screen-sized canvas. */
function coverFit(W, H, SW, SH) {
  const scale = Math.max(SW / W, SH / H);
  return { scale, ox: (SW - W * scale) / 2, oy: (SH - H * scale) / 2 };
}

// ---------------------------------------------------------------------------
// WebGL renderer
// ---------------------------------------------------------------------------

const QUAD_VERT = `
attribute vec2 aPos;
varying vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const PRECISION = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
`;

// Edge-preserving smoothing: removes pores and fine detail but keeps features crisp.
const BILATERAL_FRAG = `${PRECISION}
uniform sampler2D uSrc;
uniform vec2 uStep;
varying vec2 vUv;
void main() {
  vec3 c0 = texture2D(uSrc, vUv).rgb;
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int j = -4; j <= 4; j++) {
    for (int i = -4; i <= 4; i++) {
      vec2 o = vec2(float(i), float(j));
      vec3 c = texture2D(uSrc, vUv + o * uStep).rgb;
      vec3 d = c - c0;
      float w = exp(-dot(o, o) / 16.0) * exp(-dot(d, d) / 0.02);
      sum += c * w;
      wsum += w;
    }
  }
  gl_FragColor = vec4(sum / wsum, 1.0);
}`;

// One direction of a Gaussian blur of luminance (the height map for the sculpted shading).
const BLUR_FRAG = `${PRECISION}
uniform sampler2D uSrc;
uniform vec2 uStep;
uniform float uFromRgb;
varying vec2 vUv;
float lum(vec2 uv) {
  vec4 c = texture2D(uSrc, uv);
  return mix(c.r, dot(c.rgb, vec3(0.299, 0.587, 0.114)), uFromRgb);
}
void main() {
  float sum = 0.0;
  float wsum = 0.0;
  for (int i = -7; i <= 7; i++) {
    float w = exp(-float(i * i) / 24.0);
    sum += lum(vUv + float(i) * uStep) * w;
    wsum += w;
  }
  float v = sum / wsum;
  gl_FragColor = vec4(v, v, v, 1.0);
}`;

// Clay albedo (smoothed, lightly posterized, a touch more saturated) with height in alpha.
const CLAY_FRAG = `${PRECISION}
uniform sampler2D uSmooth;
uniform sampler2D uHeight;
varying vec2 vUv;
void main() {
  vec3 c = texture2D(uSmooth, vUv).rgb;
  float L = dot(c, vec3(0.299, 0.587, 0.114));
  float Lq = (floor(L * 9.0) + 0.5) / 9.0;
  float Lp = mix(L, Lq, 0.25);
  vec3 p = c * (Lp / max(L, 0.001));
  p = mix(vec3(dot(p, vec3(0.299, 0.587, 0.114))), p, 1.2) * vec3(1.02, 0.99, 0.95);
  gl_FragColor = vec4(clamp(p, 0.0, 1.0), texture2D(uHeight, vUv).r);
}`;

const PHOTO_MASK_FRAG = `${PRECISION}
uniform sampler2D uPhoto;
uniform sampler2D uMask;
varying vec2 vUv;
void main() { gl_FragColor = vec4(texture2D(uPhoto, vUv).rgb, texture2D(uMask, vUv).r); }`;

// Clay surface relief (lumps, thumb smears, partial fingerprints) baked once, in image
// coordinates, so it moves with the clay and costs one texture read per frame.
const RELIEF_FRAG = `${PRECISION}
uniform vec2 uImg;
uniform float uFaceW;
uniform vec4 uPrint[4];
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
void main() {
  vec2 q = vUv * uImg;
  float fw = uFaceW;
  float h = (vnoise(q / (fw * 0.08)) - 0.5) * 0.7
          + (vnoise(q / (fw * 0.025)) - 0.5) * 0.18
          + (vnoise(q / (fw * 0.008)) - 0.5) * 0.05
          + (vnoise(vec2(q.x / (fw * 0.015), q.y / (fw * 0.07))) - 0.5) * 0.16;
  for (int i = 0; i < 4; i++) {
    vec4 pr = uPrint[i];
    vec2 o = q - pr.xy;
    float c = cos(pr.w), s = sin(pr.w);
    o = vec2(c * o.x - s * o.y, s * o.x + c * o.y);
    float d = length(o * vec2(1.0, 1.35)) / pr.z;
    // Only one side of the whorl, with ridges that come and go: a smeared partial print
    // rather than a target.
    float side = smoothstep(-0.15, 0.45, o.x / pr.z);
    float broken = smoothstep(0.3, 0.7, vnoise(o / (pr.z * 0.3) + float(i) * 7.0));
    h += sin(d * 24.0) * 0.1 * side * broken * (1.0 - smoothstep(0.35, 1.0, d));
  }
  float v = clamp(0.5 + h * 0.5, 0.0, 1.0);
  gl_FragColor = vec4(v, v, v, 1.0);
}`;

// Morph mesh: triangles placed at the in-between landmarks, sampling A and B from their own.
const MORPH_VERT = `
attribute vec2 aPos;
attribute vec2 aUvA;
attribute vec2 aUvB;
uniform vec2 uImg;
varying vec2 vA;
varying vec2 vB;
void main() {
  vA = aUvA;
  vB = aUvB;
  gl_Position = vec4(aPos / uImg * 2.0 - 1.0, 0.0, 1.0);
}`;

const MORPH_FRAG = `${PRECISION}
uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform float uFade;
varying vec2 vA;
varying vec2 vB;
void main() { gl_FragColor = mix(texture2D(uTexA, vA), texture2D(uTexB, vB), uFade); }`;

// The molding: displacement (in image px) that each output point samples the clay from.
const DEFORM_GLSL = `
uniform vec4 uDefA[${MAX_DEFORMERS}];
uniform vec4 uDefB[${MAX_DEFORMERS}];
uniform vec4 uDent[${MAX_DENTS}];
uniform vec3 uFace;         // centre x, centre y, face width (image px)
uniform float uWobble;
uniform float uTime;
vec2 deform(vec2 p) {
  vec2 d = vec2(0.0);
  // Every deformer is anchored on the face; far from it nothing moves.
  vec2 f = p - uFace.xy;
  float fw = uFace.z;
  if (dot(f, f) > fw * fw * 2.9) return d;
  for (int i = 0; i < ${MAX_DEFORMERS}; i++) {
    vec4 a = uDefA[i];
    vec4 b = uDefB[i];
    if (b.z == 0.0) continue;
    vec2 q = p - a.xy;
    float r = a.z;
    float g = exp(-dot(q, q) / (r * r));
    if (a.w < 0.5) {
      d += b.xy * b.z * g;
    } else if (a.w < 1.5) {
      d += q * b.xy * b.z * g;
    } else if (a.w < 2.5) {
      float s = sin(b.z * g), c = cos(b.z * g);
      d += q - vec2(c * q.x + s * q.y, -s * q.x + c * q.y);
    } else {
      float gx = exp(-q.x * q.x / (r * r));
      float t = q.y / (r * b.w);
      float w = smoothstep(-0.5, 0.0, t) * (1.0 - smoothstep(0.25, 1.0, t));
      float streak = 0.62 + 0.38 * sin(q.x / r * 9.0 + b.x) + 0.22 * sin(q.x / r * 23.0 + b.y);
      d.y += b.z * gx * w * streak;
    }
  }
  for (int i = 0; i < ${MAX_DENTS}; i++) {
    vec4 t = uDent[i];
    if (t.w == 0.0) continue;
    vec2 q = p - t.xy;
    d -= q * 0.1 * t.w * exp(-dot(q, q) / (t.z * t.z));
  }
  float g = exp(-dot(f, f) / (fw * fw * 0.55));
  d += g * uWobble * fw * 0.007 * vec2(
    sin(p.y / fw * 5.0 + uTime * 1.7) + 0.5 * sin(p.y / fw * 11.0 - uTime * 2.3),
    cos(p.x / fw * 4.0 + uTime * 1.3) + 0.5 * sin(p.x / fw * 9.0 + uTime * 2.9));
  return d;
}

`;

// The displacement is smooth, so it's rendered once per frame into a small texture
// (a quarter of the image resolution), stored relative to uRange, and the frame samples
// it with bilinear filtering instead of evaluating every deformer per screen pixel.
const FIELD_FRAG = `${PRECISION}${DEFORM_GLSL}
uniform vec2 uImg;
uniform float uRange;
varying vec2 vUv;
void main() {
  vec2 d = deform(vUv * uImg) / uRange;
  gl_FragColor = vec4(clamp(d * 0.5 + 0.5, 0.0, 1.0), 0.0, 1.0);
}`;

// The frame: deform, then shade the clay inside the face mask, then vignette.
const FINAL_FRAG = `${PRECISION}
uniform sampler2D uPhoto;   // rgb photo, a face mask
uniform sampler2D uClay;    // rgb clay albedo, a height
uniform sampler2D uRelief;  // r: baked surface relief
uniform sampler2D uField;   // rg: displacement, see FIELD_FRAG
uniform float uRange;
uniform vec4 uDent[${MAX_DENTS}];
uniform vec3 uFace;         // centre x, centre y, face width (image px)
uniform vec2 uImg;
uniform vec2 uScreen;
uniform vec2 uOffset;
uniform float uScale;
uniform float uZoom;
uniform float uClayAmt;
uniform float uTex;
uniform float uVig;
uniform float uFlash;

// Height of the clay surface (sculpted relief plus baked surface texture), in px.
float heightAt(vec2 q) {
  float fw = uFace.z;
  vec2 uv = clamp(q / uImg, 0.0, 1.0);
  return texture2D(uClay, uv).a * fw * 0.09 + (texture2D(uRelief, uv).r - 0.5) * fw * 0.016 * uTex;
}

// Slope added by the thumb dents (Gaussian pits with a faint print), computed analytically.
vec2 dentSlope(vec2 q) {
  vec2 s = vec2(0.0);
  float fw = uFace.z;
  for (int i = 0; i < ${MAX_DENTS}; i++) {
    vec4 t = uDent[i];
    if (t.w == 0.0) continue;
    vec2 o = q - t.xy;
    float r2 = t.z * t.z;
    float g = exp(-dot(o, o) / r2);
    float ridge = 1.0 + 0.08 * sin(length(o) / t.z * 26.0) * uTex;
    s += t.w * fw * 0.045 * g * ridge * 2.0 * o / r2;
  }
  return s;
}

void main() {
  vec2 sp = vec2(gl_FragCoord.x, uScreen.y - gl_FragCoord.y);
  vec2 sc = uScreen * 0.5;
  sp = sc + (sp - sc) / uZoom;
  vec2 p = (sp - uOffset) / uScale;
  vec2 q = p - (texture2D(uField, clamp(p / uImg, 0.0, 1.0)).rg * 2.0 - 1.0) * uRange;
  vec2 uv = clamp(q / uImg, 0.0, 1.0);
  vec4 ph = texture2D(uPhoto, uv);
  vec3 col = ph.rgb;
  float amt = ph.a * uClayAmt;
  if (amt > 0.003) {
    vec3 alb = texture2D(uClay, uv).rgb;
    float e = max(1.0, uFace.z * 0.004);
    float h0 = heightAt(q);
    vec2 slope = vec2(heightAt(q + vec2(e, 0.0)) - h0, heightAt(q + vec2(0.0, e)) - h0) / e + dentSlope(q);
    vec3 n = normalize(vec3(-slope, 1.0));
    vec3 L = normalize(vec3(-0.45, -0.6, 0.66));
    float diff = max(dot(n, L), 0.0);
    // Matte body with a broad, dim, warm sheen: wax, not plastic.
    float spec = pow(max(dot(n, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0), 6.0) * 0.07;
    vec3 clay = min(alb * (0.55 + 0.42 * diff), alb * 1.02) + vec3(1.0, 0.9, 0.78) * spec + alb * vec3(0.04, 0.015, 0.0);
    col = mix(col, clay, amt);
  }
  vec2 v = (sp / uScreen - 0.5) * vec2(1.0, 0.8);
  col *= 1.0 - uVig * smoothstep(0.22, 0.72, length(v)) * 0.72;
  col = mix(col, vec3(1.0, 0.96, 0.86), uFlash);
  gl_FragColor = vec4(col, 1.0);
}`;

function createGLRenderer(canvas) {
  const gl = canvas.getContext('webgl', { alpha: false, antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
  if (!gl) return null;

  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.warn('Clay shader failed to compile:', gl.getShaderInfoLog(s));
      return null;
    }
    return s;
  };
  const program = (vsSrc, fsSrc) => {
    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      console.warn('Clay shader failed to link:', gl.getProgramInfoLog(p));
      return null;
    }
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, '');
      u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u, aPos: gl.getAttribLocation(p, 'aPos') };
  };

  const progs = {
    bilateral: program(QUAD_VERT, BILATERAL_FRAG),
    blur: program(QUAD_VERT, BLUR_FRAG),
    clay: program(QUAD_VERT, CLAY_FRAG),
    photoMask: program(QUAD_VERT, PHOTO_MASK_FRAG),
    relief: program(QUAD_VERT, RELIEF_FRAG),
    morph: program(MORPH_VERT, MORPH_FRAG),
    field: program(QUAD_VERT, FIELD_FRAG),
    final: program(QUAD_VERT, FINAL_FRAG),
  };
  if (Object.values(progs).some((p) => !p)) return null;
  const morphAttr = { aUvA: gl.getAttribLocation(progs.morph.p, 'aUvA'), aUvB: gl.getAttribLocation(progs.morph.p, 'aUvB') };

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

  const textures = [];
  const framebuffers = [];
  const texture = (source, w, h) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (source) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    textures.push(t);
    return t;
  };
  const target = (w, h) => {
    const tex = texture(null, w, h);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    framebuffers.push(fb);
    return { fb, tex, w, h };
  };
  const bindTex = (unit, tex, loc) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(loc, unit);
  };
  const drawQuad = (prog, out, setup) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, out ? out.fb : null);
    gl.viewport(0, 0, out ? out.w : canvas.width, out ? out.h : canvas.height);
    gl.useProgram(prog.p);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.enableVertexAttribArray(prog.aPos);
    gl.vertexAttribPointer(prog.aPos, 2, gl.FLOAT, false, 0, 0);
    setup(prog.u);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  let W = 0, H = 0;
  let morph = null;
  let stamped = null;
  let relief = null;
  let field = null;

  const api = {
    kind: 'webgl',

    /** Bake the clay's surface relief once (shared by A and B, so it stays put through the morph). */
    bakeRelief(prints, faceW) {
      const out = target(W, H);
      const data = new Float32Array(16);
      prints.forEach((p, i) => data.set([p.x, p.y, p.r, p.angle], i * 4));
      drawQuad(progs.relief, out, (u) => {
        gl.uniform2f(u.uImg, W, H);
        gl.uniform1f(u.uFaceW, faceW);
        gl.uniform4fv(u.uPrint, data);
      });
      relief = out.tex;
    },

    /** Precompute the clay version of an image: { photo: rgb+mask, clay: albedo+height }. */
    prepare(image, mask, faceW) {
      W = image.width;
      H = image.height;
      if (Math.max(W, H) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('Image too large for WebGL');
      if (!field) field = target(Math.ceil(W / 4), Math.ceil(H / 4));
      const photo = texture(image);
      const maskTex = texture(mask);
      const a = target(W, H), b = target(W, H), c = target(W, H), d = target(W, H);
      const step = Math.max(1, faceW * 0.005);
      drawQuad(progs.bilateral, a, (u) => { bindTex(0, photo, u.uSrc); gl.uniform2f(u.uStep, step / W, step / H); });
      drawQuad(progs.bilateral, b, (u) => { bindTex(0, a.tex, u.uSrc); gl.uniform2f(u.uStep, step / W, step / H); });
      const blur = Math.max(1, faceW * 0.0035);
      drawQuad(progs.blur, c, (u) => { bindTex(0, photo, u.uSrc); gl.uniform2f(u.uStep, blur / W, 0); gl.uniform1f(u.uFromRgb, 1); });
      drawQuad(progs.blur, a, (u) => { bindTex(0, c.tex, u.uSrc); gl.uniform2f(u.uStep, 0, blur / H); gl.uniform1f(u.uFromRgb, 0); });
      drawQuad(progs.clay, c, (u) => { bindTex(0, b.tex, u.uSmooth); bindTex(1, a.tex, u.uHeight); });
      drawQuad(progs.photoMask, d, (u) => { bindTex(0, photo, u.uPhoto); bindTex(1, maskTex, u.uMask); });
      return { photo: d.tex, clay: c.tex };
    },

    /** Set up the A -> B morph mesh. */
    setMorph(A, B, ptsA, ptsB) {
      const tris = triangulate(ptsA);
      const n = tris.length;
      const uvA = new Float32Array(n * 2), uvB = new Float32Array(n * 2);
      for (let k = 0; k < n; k++) {
        const i = tris[k];
        uvA[k * 2] = ptsA[i][0] / W; uvA[k * 2 + 1] = ptsA[i][1] / H;
        uvB[k * 2] = ptsB[i][0] / W; uvB[k * 2 + 1] = ptsB[i][1] / H;
      }
      const buf = (data, usage) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, usage);
        return b;
      };
      morph = {
        A, B, ptsA, ptsB, tris,
        pos: new Float32Array(n * 2),
        posBuf: buf(new Float32Array(n * 2), gl.DYNAMIC_DRAW),
        uvABuf: buf(uvA, gl.STATIC_DRAW),
        uvBBuf: buf(uvB, gl.STATIC_DRAW),
        photo: target(W, H),
        clay: target(W, H),
      };
    },

    setStamped(image) {
      stamped = texture(image);
    },

    render(st) {
      let photo = st.layers.A.photo, clay = st.layers.A.clay;
      if (st.stamped && stamped) {
        photo = stamped;
        clay = st.layers.B.clay;
      } else if (st.morphT !== null && morph) {
        const { ptsA, ptsB, tris, pos } = morph;
        for (let k = 0; k < tris.length; k++) {
          const i = tris[k];
          pos[k * 2] = ptsA[i][0] + (ptsB[i][0] - ptsA[i][0]) * st.morphT;
          pos[k * 2 + 1] = ptsA[i][1] + (ptsB[i][1] - ptsA[i][1]) * st.morphT;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, morph.posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
        const m = progs.morph;
        gl.useProgram(m.p);
        const attr = (loc, b) => {
          gl.bindBuffer(gl.ARRAY_BUFFER, b);
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        };
        attr(m.aPos, morph.posBuf);
        attr(morphAttr.aUvA, morph.uvABuf);
        attr(morphAttr.aUvB, morph.uvBBuf);
        gl.uniform2f(m.u.uImg, W, H);
        gl.uniform1f(m.u.uFade, st.fade);
        for (const [out, key] of [[morph.photo, 'photo'], [morph.clay, 'clay']]) {
          gl.bindFramebuffer(gl.FRAMEBUFFER, out.fb);
          gl.viewport(0, 0, W, H);
          bindTex(0, morph.A[key], m.u.uTexA);
          bindTex(1, morph.B[key], m.u.uTexB);
          gl.drawArrays(gl.TRIANGLES, 0, tris.length);
        }
        gl.disableVertexAttribArray(morphAttr.aUvA);
        gl.disableVertexAttribArray(morphAttr.aUvB);
        photo = morph.photo.tex;
        clay = morph.clay.tex;
      } else if (st.fade >= 1) {
        photo = st.layers.B.photo;
        clay = st.layers.B.clay;
      }

      const fit = coverFit(W, H, canvas.width, canvas.height);
      const defA = new Float32Array(MAX_DEFORMERS * 4), defB = new Float32Array(MAX_DEFORMERS * 4);
      st.deformers.forEach((d, i) => {
        defA.set([d.x, d.y, d.r, d.kind], i * 4);
        defB.set([d.ax, d.ay, d.amount || 0, d.aspect || 1], i * 4);
      });
      const dents = new Float32Array(MAX_DENTS * 4);
      st.dents.forEach((d, i) => dents.set([d.x, d.y, d.r, d.depth], i * 4));
      const range = st.frame.w * 0.7;
      drawQuad(progs.field, field, (u) => {
        gl.uniform4fv(u.uDefA, defA);
        gl.uniform4fv(u.uDefB, defB);
        gl.uniform4fv(u.uDent, dents);
        gl.uniform3f(u.uFace, st.frame.cx, st.frame.cy, st.frame.w);
        gl.uniform1f(u.uWobble, st.wobble);
        gl.uniform1f(u.uTime, st.time);
        gl.uniform2f(u.uImg, W, H);
        gl.uniform1f(u.uRange, range);
      });
      drawQuad(progs.final, null, (u) => {
        bindTex(0, photo, u.uPhoto);
        bindTex(1, clay, u.uClay);
        bindTex(2, relief, u.uRelief);
        bindTex(3, field.tex, u.uField);
        gl.uniform1f(u.uRange, range);
        gl.uniform4fv(u.uDent, dents);
        gl.uniform3f(u.uFace, st.frame.cx, st.frame.cy, st.frame.w);
        gl.uniform2f(u.uImg, W, H);
        gl.uniform2f(u.uScreen, canvas.width, canvas.height);
        gl.uniform2f(u.uOffset, fit.ox, fit.oy);
        gl.uniform1f(u.uScale, fit.scale);
        gl.uniform1f(u.uZoom, st.zoom);
        gl.uniform1f(u.uClayAmt, st.clay);
        gl.uniform1f(u.uTex, CLAY_TEXTURE_STRENGTH);
        gl.uniform1f(u.uVig, st.vignette);
        gl.uniform1f(u.uFlash, st.flash);
      });
    },

    destroy() {
      textures.forEach((t) => gl.deleteTexture(t));
      framebuffers.forEach((f) => gl.deleteFramebuffer(f));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
  return api;
}

// ---------------------------------------------------------------------------
// Canvas 2D fallback: blur + posterize + emboss, strip-warp wobble, crossfade
// ---------------------------------------------------------------------------

function create2DRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  let W = 0, H = 0;
  let stamped = null;
  let box = null;
  let warped = null; // offscreen layer the clay strips are warped into

  return {
    kind: 'canvas2d',

    prepare(image, mask, faceW, L) {
      W = image.width;
      H = image.height;
      // Blur by shrinking (in two steps, for a smoother result) and stretching back up.
      const k = Math.max(4, Math.round(faceW / 22));
      const half = document.createElement('canvas');
      half.width = Math.max(2, Math.round(W / 2));
      half.height = Math.max(2, Math.round(H / 2));
      half.getContext('2d').drawImage(image, 0, 0, half.width, half.height);
      const small = document.createElement('canvas');
      small.width = Math.max(2, Math.round(W / k));
      small.height = Math.max(2, Math.round(H / k));
      const smctx = small.getContext('2d');
      smctx.imageSmoothingQuality = 'high';
      smctx.drawImage(half, 0, 0, small.width, small.height);
      const soft = document.createElement('canvas');
      soft.width = W;
      soft.height = H;
      const sctx = soft.getContext('2d', { willReadFrequently: true });
      sctx.imageSmoothingQuality = 'high';
      sctx.drawImage(small, 0, 0, W, H);
      const px = sctx.getImageData(0, 0, W, H);
      const m = mask.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, W, H).data;
      const d = px.data;
      const lum = new Float32Array(W * H);
      for (let i = 0; i < lum.length; i++) lum[i] = (d[i * 4] * 0.299 + d[i * 4 + 1] * 0.587 + d[i * 4 + 2] * 0.114) / 255;
      const e = Math.max(1, Math.round(faceW * 0.01));
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const p = y * W + x;
          const L0 = lum[p];
          // Light from the upper left on the blurred-luminance relief.
          const gx = lum[y * W + Math.min(W - 1, x + e)] - lum[y * W + Math.max(0, x - e)];
          const gy = lum[Math.min(H - 1, y + e) * W + x] - lum[Math.max(0, y - e) * W + x];
          const shade = Math.min(1.2, Math.max(0.6, 0.95 - (gx * 0.5 + gy * 0.7) * 3.2));
          const Lq = (Math.floor(L0 * 7) + 0.5) / 7;
          const k2 = (L0 + (Lq - L0) * 0.4) / Math.max(L0, 0.001) * shade;
          for (let c = 0; c < 3; c++) d[p * 4 + c] = d[p * 4 + c] * k2;
          d[p * 4 + 3] = m[p * 4];
        }
      }
      const clay = document.createElement('canvas');
      clay.width = W;
      clay.height = H;
      clay.getContext('2d').putImageData(px, 0, 0);
      const f = faceFrame(L);
      box = { x: Math.max(0, f.cx - f.w * 0.75), y: Math.max(0, f.cy - f.h * 0.85), w: f.w * 1.5, h: f.h * 1.75 };
      if (!warped) {
        warped = document.createElement('canvas');
        warped.width = W;
        warped.height = H;
      }
      return { photo: image, clay };
    },

    bakeRelief() {},

    setMorph() {},

    setStamped(image) { stamped = image; },

    render(st) {
      const SW = canvas.width, SH = canvas.height;
      const fit = coverFit(W, H, SW, SH);
      const s = fit.scale * st.zoom;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, SW, SH);
      ctx.setTransform(s, 0, 0, s, SW / 2 - (SW / 2 - fit.ox) * st.zoom, SH / 2 - (SH / 2 - fit.oy) * st.zoom);

      if (st.stamped && stamped) {
        ctx.drawImage(stamped, 0, 0);
      } else {
        ctx.drawImage(st.layers.A.photo, 0, 0);
        if (st.fade > 0) {
          ctx.globalAlpha = st.fade;
          ctx.drawImage(st.layers.B.photo, 0, 0);
        }
        // Wobble the clay in horizontal strips; the amplitude follows how hard it's being molded.
        let energy = 0;
        for (const d of st.deformers) energy += Math.abs(d.kind === 0 || d.kind === 3 ? d.amount / st.frame.w : d.amount);
        const amp = st.frame.w * (0.006 * st.wobble + 0.05 * Math.min(1.5, energy));
        const strips = 60;
        const sh = box.h / strips;
        const wctx = warped.getContext('2d');
        for (const [layer, a] of [[st.layers.A.clay, st.clay * (1 - st.fade)], [st.layers.B?.clay, st.clay * st.fade]]) {
          if (!layer || a <= 0.003) continue;
          // Warp at full opacity into the offscreen layer (overlapping strips leave no seams),
          // then composite it once.
          wctx.clearRect(0, 0, W, H);
          for (let i = 0; i < strips; i++) {
            const y = box.y + i * sh;
            const dx = amp * Math.sin(y / st.frame.w * 6 + st.time * 2.3) + amp * 0.5 * Math.sin(y / st.frame.w * 13 - st.time * 3.1);
            wctx.drawImage(layer, box.x, y, box.w, sh + 1, box.x + dx, y, box.w, sh + 1);
          }
          ctx.globalAlpha = a;
          ctx.drawImage(warped, 0, 0);
        }
      }

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      if (st.vignette > 0) {
        const g = ctx.createRadialGradient(SW / 2, SH / 2, Math.min(SW, SH) * 0.25, SW / 2, SH / 2, Math.max(SW, SH) * 0.62);
        g.addColorStop(0, 'rgba(0,0,0,0)');
        g.addColorStop(1, `rgba(0,0,0,${0.72 * st.vignette})`);
        ctx.globalAlpha = 1;
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, SW, SH);
      }
      if (st.flash > 0) {
        ctx.globalAlpha = st.flash;
        ctx.fillStyle = '#fff5dc';
        ctx.fillRect(0, 0, SW, SH);
      }
      ctx.globalAlpha = 1;
    },

    destroy() {},
  };
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/** True if this browser can record the stage for sharing. */
export function canRecord() {
  return typeof MediaRecorder === 'function' && typeof HTMLCanvasElement.prototype.captureStream === 'function'
    && VIDEO_TYPES.some((t) => MediaRecorder.isTypeSupported?.(t));
}

function startRecorder(canvas) {
  if (!canRecord()) return null;
  try {
    const type = VIDEO_TYPES.find((t) => MediaRecorder.isTypeSupported(t));
    const stream = canvas.captureStream(30);
    const rec = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 3_000_000 });
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    const done = new Promise((resolve) => {
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        resolve(chunks.length ? new Blob(chunks, { type: type.split(';')[0] }) : null);
      };
      rec.onerror = () => resolve(null);
    });
    rec.start(250);
    return { stop: () => { if (rec.state !== 'inactive') rec.stop(); return done; } };
  } catch (err) {
    console.warn('Recording unavailable:', err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// The stage
// ---------------------------------------------------------------------------

/**
 * Start the clayface on `canvas` with the selfie A and its landmarks. Phases 1 and 2 run
 * straight away; call finish() with the swapped photo to play phases 3 and 4.
 * @param {HTMLCanvasElement} canvas full-screen stage
 * @param {HTMLCanvasElement} A the selfie
 * @param {number[][]} LA A's 468 landmarks in pixels
 * @param {{ onPhase?: (phase: string) => void }} [opts]
 */
export function createClayStage(canvas, A, LA, opts = {}) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const sizeCanvas = () => {
    let w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
    const k = Math.min(1, Math.sqrt(MAX_PIXELS / (w * h || 1)));
    canvas.width = Math.max(2, Math.round(w * k));
    canvas.height = Math.max(2, Math.round(h * k));
  };
  sizeCanvas();

  const W = A.width, H = A.height;
  const frameA = faceFrame(LA);
  const maskA = faceMask(W, H, LA);

  let renderer = null;
  let layers = null;
  try {
    canvas.dataset.glTried = '1';
    renderer = createGLRenderer(canvas);
    if (renderer) layers = { A: renderer.prepare(A, maskA, frameA.w, LA) };
  } catch (err) {
    console.warn('WebGL clay failed, using Canvas 2D:', err);
    renderer?.destroy();
    renderer = null;
  }
  if (!renderer) {
    // A canvas that had a WebGL context can't give a 2D one: swap in a fresh element.
    if (canvas.dataset.glTried) {
      const fresh = canvas.cloneNode();
      canvas.replaceWith(fresh);
      canvas = fresh;
      sizeCanvas();
    }
    renderer = create2DRenderer(canvas);
    if (!renderer) throw new Error('No canvas renderer available');
    layers = { A: renderer.prepare(A, maskA, frameA.w, LA) };
  }

  const prints = PRINT_SITES.slice().sort(() => Math.random() - 0.5).slice(0, 4).map((i) => ({
    x: LA[i][0] + rand(-0.03, 0.03) * frameA.w,
    y: LA[i][1] + rand(-0.03, 0.03) * frameA.w,
    r: rand(0.11, 0.15) * frameA.w,
    angle: rand(0, Math.PI),
  }));

  renderer.bakeRelief(prints, frameA.w);

  const hands = new Hands();
  const t0 = performance.now();
  const kneadAt = t0 + CLAY_IN_MS;
  let phase = 'clay';
  let becomeAt = Infinity, hardenAt = Infinity, snapAt = Infinity, endAt = Infinity;
  let LB = null;
  let recorder = null, recordAt = Infinity;
  let raf = 0;
  let resolveDone;
  const done = new Promise((r) => { resolveDone = r; });

  const setPhase = (p) => {
    if (phase === p) return;
    phase = p;
    opts.onPhase?.(p);
  };

  const onResize = () => sizeCanvas();
  window.addEventListener('resize', onResize);

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (now >= recordAt && !recorder) {
      recorder = startRecorder(canvas) || { stop: async () => null };
      recordAt = Infinity;
    }

    const st = {
      layers, time: (now - t0) / 1000, zoom: 1, flash: 0,
      morphT: null, fade: 0, clay: 1, wobble: 1, vignette: 1, stamped: false,
      deformers: [], dents: [], frame: frameA,
    };
    let L = LA, gain = 1, spawning = true;

    if (now < kneadAt) {
      const s = (now - t0) / CLAY_IN_MS;
      st.clay = ease(s);
      st.vignette = ease(s);
      st.wobble = ease(s);
      gain = 0;
      spawning = false;
    } else if (now < becomeAt) {
      setPhase('knead');
    } else if (now < hardenAt) {
      setPhase('become');
      const s = (now - becomeAt) / BECOME_MS;
      st.morphT = easeInOut(s);
      st.fade = ease((s - 0.2) / 0.65);
      L = lerpPts(LA, LB, st.morphT);
      gain = 1.7 * (1 - s) ** 1.6;
      spawning = s < 0.65;
      st.wobble = 1.25 - 0.9 * s;
      st.vignette = 1 - 0.4 * s;
    } else if (now < snapAt) {
      setPhase('harden');
      const s = (now - hardenAt) / HARDEN_MS;
      L = LB;
      st.fade = 1;
      st.clay = 1 - ease((s - 0.1) / 0.75);
      // Settle with a slight overshoot.
      gain = Math.exp(-4.5 * s) * Math.cos(s * 9.4) * 0.6;
      spawning = false;
      st.wobble = 0.35 * Math.exp(-4 * s) * Math.cos(s * 12);
      st.vignette = 0.6 * (1 - ease(s));
    } else {
      setPhase('snap');
      const s = (now - snapAt) / SNAP_MS;
      st.stamped = true;
      st.fade = 1;
      st.clay = 0;
      st.wobble = 0;
      st.vignette = 0;
      gain = 0;
      spawning = false;
      st.zoom = 1 + 0.035 * (1 - ease(s));
      st.flash = 0.16 * (1 - ease(s));
      L = LB;
    }

    const knead = hands.update(now, L, gain, spawning);
    st.deformers = knead.deformers;
    st.dents = knead.dents;
    st.frame = knead.frame;
    if (st.wobble < 0) st.wobble = 0;
    renderer.render(st);

    if (now >= endAt) {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      (recorder ? recorder.stop() : Promise.resolve(null)).then((video) => resolveDone({ video }));
      endAt = Infinity;
    }
  }
  raf = requestAnimationFrame(frame);
  opts.onPhase?.('clay');

  return {
    renderer: renderer.kind,
    get canvas() { return canvas; },

    /**
     * Play phases 3 and 4 into the swapped photo B (same size as A), then the snap to
     * `stampedB`. Resolves with { video } (a Blob, or null if recording isn't possible)
     * once the stamped photo has been held for the clip.
     */
    finish(B, landmarksB, stampedB) {
      LB = landmarksB || LA;
      const maskB = faceMask(W, H, LB);
      layers.B = renderer.prepare(B, maskB, faceFrame(LB).w, LB);
      renderer.setMorph(layers.A, layers.B, morphPoints(LA, W, H), morphPoints(LB, W, H));
      renderer.setStamped(stampedB);

      const now = performance.now();
      becomeAt = Math.max(kneadAt + MIN_KNEAD_MS, now + PREROLL_MS);
      recordAt = becomeAt - PREROLL_MS;
      hardenAt = becomeAt + BECOME_MS;
      snapAt = hardenAt + HARDEN_MS;
      endAt = snapAt + HOLD_MS;
      return done;
    },

    destroy() {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      recorder?.stop();
      renderer.destroy();
    },
  };
}
