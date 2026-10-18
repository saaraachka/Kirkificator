// GET ?id=<prediction id> -> { status, outputUrl }
// Only the status and the output image URL are passed back, never Replicate's full response.

import { json, logFailure, replicate } from '../lib/replicate.js';

const PREDICTION_ID = /^[a-z0-9]{8,64}$/i;

export default async (req) => {
  if (req.method !== 'GET') return json({ error: 'Method not allowed' }, 405);
  const id = new URL(req.url).searchParams.get('id');
  if (!id || !PREDICTION_ID.test(id)) return json({ error: 'Invalid id' }, 400);

  try {
    const prediction = await replicate(`/predictions/${id}`);
    // starting | processing | succeeded | failed | canceled
    const status = typeof prediction.status === 'string' ? prediction.status : 'failed';
    const output = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output;
    const outputUrl = status === 'succeeded' && typeof output === 'string' && output.startsWith('https://') ? output : null;
    return json({ status: status === 'succeeded' && !outputUrl ? 'failed' : status, outputUrl });
  } catch (err) {
    logFailure('swap-status', err);
    return json({ error: 'Could not check the swap' }, 502);
  }
};
