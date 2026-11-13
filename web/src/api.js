export const TOKEN_KEY = "provident.token";

function token() {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export async function api(path, options = {}) {
  const init = { ...options };
  init.headers = { ...(options.headers || {}) };
  const bearer = token();
  if (bearer && !init.headers.Authorization) {
    init.headers.Authorization = `Bearer ${bearer}`;
  }
  if (init.body && !(init.body instanceof FormData) && typeof init.body !== "string") {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(init.body);
  }
  const res = await fetch(path, init);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    if (res.status === 401) {
      try {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem("provident.user");
      } catch {
        /* ignore */
      }
      const here = window.location.pathname.replace(/\/+$/, "");
      if (!here.endsWith("/login")) {
        window.location.assign("/ui/login");
      }
    }
    const detail = data && data.detail;
    let message = typeof detail === "string" ? detail : "";
    if (!message && res.status === 429) {
      const retry = res.headers.get("Retry-After");
      message = retry ? `请求过于频繁，请 ${retry} 秒后再试` : "请求过于频繁，请稍后再试";
    }
    const error = new Error(message || res.statusText || `HTTP ${res.status}`);
    error.status = res.status;
    throw error;
  }
  return data;
}

export async function apiStream(path, { onEvent, signal } = {}) {
  const headers = {};
  const bearer = token();
  if (bearer) {
    headers.Authorization = `Bearer ${bearer}`;
  }
  const res = await fetch(path, { headers, signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText || `HTTP ${res.status}`);
  }
  if (!res.body) {
    throw new Error("stream not supported");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const block of parts) {
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data && onEvent) {
        try {
          onEvent(event, JSON.parse(data));
        } catch {
          onEvent(event, data);
        }
      }
    }
  }
}
