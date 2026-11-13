/** One logical Agent submit keeps one idempotency key across network retries. */

const RETRYABLE_STATUS = new Set([502, 503, 504]);

export function newIdempotencyKey() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 12)}`;
}

export function isRetryableSubmit(error) {
  if (!error || error.name === "AbortError") return false;
  if (error.status == null) return true;
  return RETRYABLE_STATUS.has(error.status);
}

export async function submitAgentRun(api, body, options = {}) {
  const attempts = options.attempts ?? 3;
  const key = options.key || newIdempotencyKey();
  let last = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await api("/agent/run", {
        method: "POST",
        body,
        headers: { ...(options.headers || {}), "Idempotency-Key": key },
      });
    } catch (error) {
      last = error;
      if (attempt >= attempts || !isRetryableSubmit(error)) throw error;
    }
  }
  throw last;
}
