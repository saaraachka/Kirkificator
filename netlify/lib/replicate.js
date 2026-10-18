// Shared helpers for the swap functions (kept outside netlify/functions so they are
// bundled into each function rather than deployed as an endpoint of their own).

// cdingram/face-swap: `input_image` is the target (the photo whose face is replaced),
// `swap_image` is the face pasted onto it. The output is a single image URL.
export const DEFAULT_MODEL = 'cdingram/face-swap:d1d6ea8c8be89d664a07a457526f7128109dee7030fdac424788d762c71ed111';

// Overridable only so the functions can be tested against a local mock.
const API_BASE = process.env.REPLICATE_API_BASE || 'https://api.replicate.com/v1';

// Netlify sync functions are cut off at about 10 s, so no single Replicate call may run long.
const REQUEST_TIMEOUT_MS = 8000;

export class ReplicateError extends Error {
  constructor(status) {
    super(`Replicate responded ${status}`); // the status only, never the response body
    this.name = 'ReplicateError';
    this.status = status;
  }
}

/** Call the Replicate API and return the parsed JSON body. */
export async function replicate(path, init = {}) {
  const token = process.env.REPLICATE_API_TOKEN;
  if (!token) throw new Error('REPLICATE_API_TOKEN is not set');
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!res.ok) throw new ReplicateError(res.status);
  return res.json();
}

export function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  });
}

/** Server-side log line for a failure. The token only ever travels in a request header, so it can't appear here. */
export function logFailure(where, err) {
  console.error(`${where}: ${err?.name || 'Error'}: ${err?.message || err}`);
}
