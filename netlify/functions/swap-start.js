// POST { image: "data:image/jpeg;base64,..." } -> { id }
// Starts a face-swap prediction on Replicate and returns straight away; the client polls
// swap-status for the result, because a sync function can't wait for the model to finish.

import { DEFAULT_MODEL, json, logFailure, replicate } from '../lib/replicate.js';

const MAX_BODY_BYTES = 4 * 1024 * 1024;
const DATA_URL = /^data:image\/(jpeg|png);base64,([A-Za-z0-9+/]+={0,2})$/;
const SOURCE_FACE_PATH = '/assets/source-face.jpg';

export default async (req) => {
  if (req.method !== 'POST') return json({ error: 'Method not allowed' }, 405);

  const declared = Number(req.headers.get('content-length'));
  if (declared > MAX_BODY_BYTES) return json({ error: 'Image too large' }, 413);
  const raw = await req.text();
  if (Buffer.byteLength(raw) > MAX_BODY_BYTES) return json({ error: 'Image too large' }, 413);

  let image;
  try {
    image = JSON.parse(raw).image;
  } catch {
    return json({ error: 'Invalid JSON' }, 400);
  }
  const match = typeof image === 'string' && DATA_URL.exec(image);
  if (!match || !hasImageSignature(match[1], Buffer.from(match[2], 'base64'))) {
    return json({ error: 'Expected a JPEG or PNG data URL' }, 415);
  }

  // Replicate fetches the source face from the deployed site itself.
  const sourceFace = new URL(SOURCE_FACE_PATH, new URL(req.url).origin).href;
  const model = process.env.REPLICATE_MODEL || DEFAULT_MODEL;
  const [name, version] = model.split(':');
  const input = { input_image: image, swap_image: sourceFace };

  try {
    // A pinned version goes to /predictions; a bare "owner/name" runs the model's current version.
    const prediction = version
      ? await replicate('/predictions', { method: 'POST', body: JSON.stringify({ version, input }) })
      : await replicate(`/models/${name}/predictions`, { method: 'POST', body: JSON.stringify({ input }) });
    if (typeof prediction.id !== 'string') throw new Error('No prediction id in response');
    return json({ id: prediction.id });
  } catch (err) {
    logFailure('swap-start', err);
    return json({ error: 'Could not start the swap' }, 502);
  }
};

/** Check the decoded bytes really are the image type the data URL claims. */
function hasImageSignature(type, bytes) {
  if (type === 'jpeg') return bytes.length > 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
  return bytes.length > 8 && bytes.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
}
