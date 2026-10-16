// Frequency-separation face transfer (in CIE Lab) and multi-band (Laplacian pyramid) blending.
// Images here are Float32Arrays of interleaved RGB (3 channels, 0..255) unless noted.

// ---------------------------------------------------------------------------
// sRGB <-> Lab (D65)
// ---------------------------------------------------------------------------

const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const v = i / 255;
  SRGB_TO_LINEAR[i] = v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}
const XN = 0.95047, ZN = 1.08883;

const labF = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
const labFInv = (f) => {
  const f3 = f * f * f;
  return f3 > 0.008856 ? f3 : (f - 16 / 116) / 7.787;
};
const toSrgb = (v) => 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.max(0, v) ** (1 / 2.4) - 0.055);

/** Convert one 8-bit sRGB pixel to Lab, writing into out[o..o+2]. */
function rgbToLab(r, g, b, out, o) {
  const R = SRGB_TO_LINEAR[r], G = SRGB_TO_LINEAR[g], B = SRGB_TO_LINEAR[b];
  const fx = labF((0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / XN);
  const fy = labF(0.2126729 * R + 0.7151522 * G + 0.072175 * B);
  const fz = labF((0.0193339 * R + 0.119192 * G + 0.9503041 * B) / ZN);
  out[o] = 116 * fy - 16;
  out[o + 1] = 500 * (fx - fy);
  out[o + 2] = 200 * (fy - fz);
}

/** Convert one Lab pixel to sRGB (0..255 floats, unclamped), writing into out[o..o+2]. */
function labToRgb(L, a, b, out, o) {
  const fy = (L + 16) / 116;
  const X = XN * labFInv(fy + a / 500);
  const Y = labFInv(fy);
  const Z = ZN * labFInv(fy - b / 200);
  out[o] = toSrgb(3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z);
  out[o + 1] = toSrgb(-0.969266 * X + 1.8760108 * Y + 0.041556 * Z);
  out[o + 2] = toSrgb(0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
}

/**
 * Separable box blur (running sums) applied `passes` times in each direction, in place.
 * Three passes of radius r approximate a Gaussian with sigma ~ r + 0.5. Pixels beyond
 * the edge count as zero. `src` is single-channel, w x h.
 */
export function boxBlur(src, w, h, r, passes = 3) {
  const tmp = new Float32Array(src.length);
  for (let pass = 0; pass < passes; pass++) {
    blurLine(src, tmp, w, h, r, true);
    blurLine(tmp, src, w, h, r, false);
  }
  return src;
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
 * Low-frequency Lab image of skin only, as a normalised (weighted) blur: regions with
 * zero weight (eyes, brows, lips, hair, background) are filled from the skin around
 * them instead of leaking their own colour in. Computed on a grid `f` times coarser
 * than the image (the blur is wide, so nothing is lost) and sampled back bilinearly.
 * @returns {(x: number, y: number, out: Float32Array) => void} sampler
 */
function skinLowPass(rgba, weight, w, h, r) {
  const f = Math.max(1, Math.floor(r / 4));
  const lw = Math.ceil(w / f), lh = Math.ceil(h / f), ln = lw * lh;

  // Weighted average colour of each f x f cell, then its Lab, weighted by the cell's skin.
  const den = new Float32Array(ln);
  const rgb = new Float32Array(ln * 3);
  for (let y = 0; y < h; y++) {
    const row = ((y / f) | 0) * lw;
    for (let x = 0; x < w; x++) {
      const p = y * w + x, wt = weight[p];
      if (wt === 0) continue;
      const cell = row + ((x / f) | 0);
      den[cell] += wt;
      rgb[cell * 3] += rgba[p * 4] * wt;
      rgb[cell * 3 + 1] += rgba[p * 4 + 1] * wt;
      rgb[cell * 3 + 2] += rgba[p * 4 + 2] * wt;
    }
  }
  const num = [new Float32Array(ln), new Float32Array(ln), new Float32Array(ln)];
  const lab = new Float32Array(3);
  const mean = [0, 0, 0];
  let total = 0;
  for (let cell = 0; cell < ln; cell++) {
    const d = den[cell];
    if (d === 0) continue;
    rgbToLab(Math.round(rgb[cell * 3] / d), Math.round(rgb[cell * 3 + 1] / d), Math.round(rgb[cell * 3 + 2] / d), lab, 0);
    for (let c = 0; c < 3; c++) {
      num[c][cell] = lab[c] * d;
      mean[c] += num[c][cell];
    }
    total += d;
  }
  for (let c = 0; c < 3; c++) mean[c] = total > 0 ? mean[c] / total : c === 0 ? 50 : 0;

  const lr = Math.max(1, Math.round(r / f));
  boxBlur(den, lw, lh, lr);
  num.forEach((ch) => boxBlur(ch, lw, lh, lr));

  // Far from any skin the weights vanish; fall back smoothly to the mean skin colour.
  const eps = 1e-3 * f * f;
  const low = new Float32Array(ln * 3);
  for (let cell = 0; cell < ln; cell++) {
    const d = den[cell] + eps;
    for (let c = 0; c < 3; c++) low[cell * 3 + c] = (num[c][cell] + eps * mean[c]) / d;
  }

  return (x, y, out) => {
    const gx = Math.min(lw - 1, Math.max(0, (x + 0.5) / f - 0.5));
    const gy = Math.min(lh - 1, Math.max(0, (y + 0.5) / f - 0.5));
    const x0 = gx | 0, y0 = gy | 0;
    const x1 = Math.min(lw - 1, x0 + 1), y1 = Math.min(lh - 1, y0 + 1);
    const tx = gx - x0, ty = gy - y0;
    const a = (y0 * lw + x0) * 3, b = (y0 * lw + x1) * 3, c = (y1 * lw + x0) * 3, d = (y1 * lw + x1) * 3;
    for (let k = 0; k < 3; k++) {
      const top = low[a + k] + (low[b + k] - low[a + k]) * tx;
      const bottom = low[c + k] + (low[d + k] - low[c + k]) * tx;
      out[k] = top + (bottom - top) * ty;
    }
  };
}

/** How far the kept eye regions (and the brows' tint) shift toward the selfie's skin tone and lighting. */
const EYE_ADAPT = 0.35;

/**
 * Frequency-separation face transfer, computed in Lab:
 *   face = selfie_lowfreq + (source - source_lowfreq) * detailStrength
 * The low-frequency layers are masked, normalised blurs of skin only, so the result takes
 * the selfie's skin tone, lighting and shading, and only the source contributes features
 * and texture.
 *
 * Two feature-specific tweaks:
 * - brows: the lightness detail is scaled by `browDarken`, and the colour tint comes from
 *   the source's own brow hair (lightly adapted) rather than "brow minus skin" chroma,
 *   which turns teal on warmer or darker skin. Lightness always comes from the detail
 *   layer, so the brow stays solid with no lighter core or outline.
 * - eyes: inside the eye openings the source's own colours are kept, shifted only part way
 *   toward the selfie's lighting, so the whites stay white on any skin tone.
 * @param {Uint8ClampedArray} selfie RGBA
 * @param {Uint8ClampedArray} warped RGBA, the source face warped into place
 * @param {object} o
 * @param {Float32Array} o.selfieSkin 0..1 weight: the selfie's skin, features masked out
 * @param {Float32Array} o.sourceSkin 0..1 weight: the source's skin, features masked out
 * @param {Float32Array} o.eyes 0..1 weight: the source's eye openings
 * @param {Float32Array} o.brows 0..1 weight: the source's eyebrows
 * @param {number} o.w
 * @param {number} o.h
 * @param {number} o.radius blur radius in px
 * @param {number} o.detailStrength
 * @param {number} o.browDarken
 * @returns {Float32Array} float RGB (0..255); selfie pixels where nothing was warped
 */
export function transferDetail(selfie, warped, o) {
  const { w, h, eyes, brows, detailStrength, browDarken } = o;
  const selfieLow = skinLowPass(selfie, o.selfieSkin, w, h, o.radius);
  const sourceLow = skinLowPass(warped, o.sourceSkin, w, h, o.radius);

  const out = new Float32Array(w * h * 3);
  const src = new Float32Array(3), sLow = new Float32Array(3), wLow = new Float32Array(3), lab = new Float32Array(3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x, i = p * 3;
      if (warped[p * 4 + 3] === 0) {
        out[i] = selfie[p * 4];
        out[i + 1] = selfie[p * 4 + 1];
        out[i + 2] = selfie[p * 4 + 2];
        continue;
      }
      rgbToLab(warped[p * 4], warped[p * 4 + 1], warped[p * 4 + 2], src, 0);
      selfieLow(x, y, sLow);
      sourceLow(x, y, wLow);
      const b = brows[p], e = eyes[p];

      // Lightness: detail layer, boosted or softened inside the brows.
      const gainL = detailStrength * (1 + (browDarken - 1) * b);
      lab[0] = sLow[0] + (src[0] - wLow[0]) * gainL;
      // Tint: detail layer on skin, the source's own (adapted) hair colour in the brows.
      for (let c = 1; c < 3; c++) {
        const separated = sLow[c] + (src[c] - wLow[c]) * detailStrength;
        const own = src[c] + (sLow[c] - wLow[c]) * EYE_ADAPT;
        lab[c] = separated + (own - separated) * b;
      }
      // Eye openings: the source's own colours throughout.
      if (e > 0) {
        for (let c = 0; c < 3; c++) {
          const own = src[c] + (sLow[c] - wLow[c]) * EYE_ADAPT;
          lab[c] += (own - lab[c]) * e;
        }
      }

      const L = Math.min(100, Math.max(0, lab[0]));
      labToRgb(L, Math.min(127, Math.max(-128, lab[1])), Math.min(127, Math.max(-128, lab[2])), out, i);
      for (let c = 0; c < 3; c++) out[i + c] = Math.min(255, Math.max(0, out[i + c]));
    }
  }
  return out;
}

/** RGBA bytes -> float RGB. */
export function rgbaToRgb(rgba) {
  const n = rgba.length / 4;
  const out = new Float32Array(n * 3);
  for (let p = 0; p < n; p++) {
    out[p * 3] = rgba[p * 4];
    out[p * 3 + 1] = rgba[p * 4 + 1];
    out[p * 3 + 2] = rgba[p * 4 + 2];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Laplacian pyramid blending
// ---------------------------------------------------------------------------

/** Blur with the 5-tap binomial kernel [1 4 6 4 1]/16 and halve the size. */
function pyrDown(src, w, h, ch) {
  const nw = (w + 1) >> 1, nh = (h + 1) >> 1;
  const tmp = new Float32Array(nw * h * ch);
  const cx = (x) => (x < 0 ? 0 : x >= w ? w - 1 : x);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let nx = 0; nx < nw; nx++) {
      const x = nx * 2;
      const x0 = (row + cx(x - 2)) * ch, x1 = (row + cx(x - 1)) * ch, x2 = (row + x) * ch;
      const x3 = (row + cx(x + 1)) * ch, x4 = (row + cx(x + 2)) * ch;
      const o = (y * nw + nx) * ch;
      for (let c = 0; c < ch; c++) {
        tmp[o + c] = (src[x0 + c] + 4 * src[x1 + c] + 6 * src[x2 + c] + 4 * src[x3 + c] + src[x4 + c]) / 16;
      }
    }
  }
  const out = new Float32Array(nw * nh * ch);
  const cy = (y) => (y < 0 ? 0 : y >= h ? h - 1 : y);
  const stride = nw * ch;
  for (let ny = 0; ny < nh; ny++) {
    const y = ny * 2;
    const r0 = cy(y - 2) * stride, r1 = cy(y - 1) * stride, r2 = y * stride;
    const r3 = cy(y + 1) * stride, r4 = cy(y + 2) * stride;
    const o = ny * stride;
    for (let i = 0; i < stride; i++) {
      out[o + i] = (tmp[r0 + i] + 4 * tmp[r1 + i] + 6 * tmp[r2 + i] + 4 * tmp[r3 + i] + tmp[r4 + i]) / 16;
    }
  }
  return { data: out, w: nw, h: nh };
}

/** Expand a level back up to w x h with the matching interpolation kernel. */
function pyrUp(level, w, h, ch) {
  const { data: src, w: sw, h: sh } = level;
  const tmp = new Float32Array(w * sh * ch);
  const cx = (x) => (x < 0 ? 0 : x >= sw ? sw - 1 : x);
  for (let y = 0; y < sh; y++) {
    const row = y * sw;
    for (let x = 0; x < w; x++) {
      const i = x >> 1;
      const o = (y * w + x) * ch;
      if (x & 1) {
        const a = (row + i) * ch, b = (row + cx(i + 1)) * ch;
        for (let c = 0; c < ch; c++) tmp[o + c] = (src[a + c] + src[b + c]) / 2;
      } else {
        const a = (row + cx(i - 1)) * ch, b = (row + i) * ch, e = (row + cx(i + 1)) * ch;
        for (let c = 0; c < ch; c++) tmp[o + c] = (src[a + c] + 6 * src[b + c] + src[e + c]) / 8;
      }
    }
  }
  const out = new Float32Array(w * h * ch);
  const cy = (y) => (y < 0 ? 0 : y >= sh ? sh - 1 : y);
  const stride = w * ch;
  for (let y = 0; y < h; y++) {
    const i = y >> 1;
    const o = y * stride;
    if (y & 1) {
      const a = i * stride, b = cy(i + 1) * stride;
      for (let k = 0; k < stride; k++) out[o + k] = (tmp[a + k] + tmp[b + k]) / 2;
    } else {
      const a = cy(i - 1) * stride, b = i * stride, e = cy(i + 1) * stride;
      for (let k = 0; k < stride; k++) out[o + k] = (tmp[a + k] + 6 * tmp[b + k] + tmp[e + k]) / 8;
    }
  }
  return out;
}

/**
 * Blend `fg` over `bg` (float RGB, w x h) through `mask` (0..1, w x h) band by band:
 * coarse levels (colour, lighting) mix across a wide area, fine levels (pores,
 * eyelashes) mix across a narrow one, so there is neither a seam nor a smeared face.
 *
 * Pyramids are linear, so blending the Laplacian pyramids of fg and bg is the same as
 * adding the mask-weighted pyramid of their difference to bg. That needs one pyramid
 * instead of two, and two expansions per level instead of three.
 * @returns {Float32Array} float RGB
 */
export function multiBandBlend(fg, bg, mask, w, h, levels) {
  const diff = new Float32Array(fg.length);
  for (let i = 0; i < diff.length; i++) diff[i] = fg[i] - bg[i];

  const gd = [{ data: diff, w, h }];
  const gm = [{ data: mask, w, h }];
  for (let i = 1; i < levels; i++) {
    const d = gd[i - 1], m = gm[i - 1];
    gd.push(pyrDown(d.data, d.w, d.h, 3));
    gm.push(pyrDown(m.data, m.w, m.h, 1));
  }

  // Coarsest level: the Gaussian image itself, weighted by the mask.
  const top = levels - 1;
  let out = new Float32Array(gd[top].data.length);
  {
    const d = gd[top].data, m = gm[top].data;
    for (let p = 0; p < m.length; p++) {
      for (let c = 0; c < 3; c++) out[p * 3 + c] = d[p * 3 + c] * m[p];
    }
  }

  // Walk back down: out = expand(out) + mask * (this level's Laplacian of the difference).
  for (let l = top - 1; l >= 0; l--) {
    const { w: lw, h: lh } = gd[l];
    const upOut = pyrUp({ data: out, w: gd[l + 1].w, h: gd[l + 1].h }, lw, lh, 3);
    const upD = pyrUp(gd[l + 1], lw, lh, 3);
    const d = gd[l].data, m = gm[l].data;
    for (let p = 0; p < m.length; p++) {
      for (let c = 0; c < 3; c++) {
        const i = p * 3 + c;
        upOut[i] += (d[i] - upD[i]) * m[p];
      }
    }
    out = upOut;
  }

  for (let i = 0; i < out.length; i++) out[i] += bg[i];
  return out;
}
