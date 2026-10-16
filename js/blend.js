// Colour matching in CIE Lab and multi-band (Laplacian pyramid) blending.
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
 * Shift the warped face's Lab mean/std toward the selfie's, both measured over the
 * `skin` region, by `strength` (0..1). Returns the warped image as float RGB.
 * @param {Uint8ClampedArray} selfie RGBA
 * @param {Uint8ClampedArray} warped RGBA
 * @param {Float32Array} skin 0..1 per pixel
 */
export function colorMatchLab(selfie, warped, skin, strength, minGain, maxGain) {
  const n = skin.length;
  const lab = new Float32Array(3);
  const sumS = [0, 0, 0], sqS = [0, 0, 0], sumW = [0, 0, 0], sqW = [0, 0, 0];
  let count = 0;
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    if (skin[p] < 0.5 || warped[i + 3] < 250) continue;
    rgbToLab(selfie[i], selfie[i + 1], selfie[i + 2], lab, 0);
    for (let c = 0; c < 3; c++) { sumS[c] += lab[c]; sqS[c] += lab[c] * lab[c]; }
    rgbToLab(warped[i], warped[i + 1], warped[i + 2], lab, 0);
    for (let c = 0; c < 3; c++) { sumW[c] += lab[c]; sqW[c] += lab[c] * lab[c]; }
    count++;
  }

  const out = new Float32Array(n * 3);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    out[p * 3] = warped[i];
    out[p * 3 + 1] = warped[i + 1];
    out[p * 3 + 2] = warped[i + 2];
  }
  if (count === 0 || strength <= 0) return out;

  // Per-channel affine map in Lab, already scaled by `strength`: v' = v * k + d.
  const k = [], d = [];
  for (let c = 0; c < 3; c++) {
    const mS = sumS[c] / count, mW = sumW[c] / count;
    const sdS = Math.sqrt(Math.max(0, sqS[c] / count - mS * mS));
    const sdW = Math.sqrt(Math.max(0, sqW[c] / count - mW * mW));
    const gain = Math.min(maxGain, Math.max(minGain, sdS / (sdW || 1)));
    k[c] = 1 + (gain - 1) * strength;
    d[c] = (mS - mW * gain) * strength;
  }

  for (let p = 0, i = 0; p < n; p++, i += 4) {
    if (warped[i + 3] === 0) continue;
    rgbToLab(warped[i], warped[i + 1], warped[i + 2], lab, 0);
    labToRgb(lab[0] * k[0] + d[0], lab[1] * k[1] + d[1], lab[2] * k[2] + d[2], out, p * 3);
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
 * @returns {Float32Array} float RGB
 */
export function multiBandBlend(fg, bg, mask, w, h, levels) {
  const gf = [{ data: fg, w, h }];
  const gb = [{ data: bg, w, h }];
  const gm = [{ data: mask, w, h }];
  for (let i = 1; i < levels; i++) {
    const f = gf[i - 1], b = gb[i - 1], m = gm[i - 1];
    gf.push(pyrDown(f.data, f.w, f.h, 3));
    gb.push(pyrDown(b.data, b.w, b.h, 3));
    gm.push(pyrDown(m.data, m.w, m.h, 1));
  }

  // Coarsest level: plain blend of the Gaussian images.
  const top = levels - 1;
  let out = new Float32Array(gf[top].data.length);
  {
    const f = gf[top].data, b = gb[top].data, m = gm[top].data;
    for (let p = 0; p < m.length; p++) {
      for (let c = 0; c < 3; c++) {
        const i = p * 3 + c;
        out[i] = b[i] + (f[i] - b[i]) * m[p];
      }
    }
  }

  // Walk back down: out = expand(out) + blend of this level's Laplacians.
  for (let l = top - 1; l >= 0; l--) {
    const { w: lw, h: lh } = gf[l];
    const upOut = pyrUp({ data: out, w: gf[l + 1].w, h: gf[l + 1].h }, lw, lh, 3);
    const upF = pyrUp(gf[l + 1], lw, lh, 3);
    const upB = pyrUp(gb[l + 1], lw, lh, 3);
    const f = gf[l].data, b = gb[l].data, m = gm[l].data;
    for (let p = 0; p < m.length; p++) {
      for (let c = 0; c < 3; c++) {
        const i = p * 3 + c;
        const lf = f[i] - upF[i];
        const lb = b[i] - upB[i];
        upOut[i] += lb + (lf - lb) * m[p];
      }
    }
    out = upOut;
  }
  return out;
}
