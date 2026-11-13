// Browser-only drafts and display order. Saved answers are always fetched through
// the authorized APIs; this is not a shared backend conversation or model memory.
export function chatKey(owner, record) {
  return `provident.chat:${owner || "local"}:${record}`;
}

export function readChat(key, fallback = null) {
  if (!key) return fallback;
  try {
    return JSON.parse(sessionStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeChat(key, value) {
  if (!key) return;
  try {
    if (value == null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* Storage can be disabled or full; execution remains available. */
  }
}
