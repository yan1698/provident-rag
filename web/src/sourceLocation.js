export function chunkAnchor(chunkId) {
  return `chunk-${chunkId}`;
}

export function sourceHref(item) {
  const params = new URLSearchParams();
  if (item.chunk_id) params.set("chunk", item.chunk_id);
  if (item.chunk_hash) params.set("hash", item.chunk_hash);
  if (item.text) params.set("hl", String(item.text).slice(0, 80));
  const query = params.toString();
  return `/library/${encodeURIComponent(item.doc_id)}${query ? `?${query}` : ""}${item.chunk_id ? `#${chunkAnchor(item.chunk_id)}` : ""}`;
}

export function orderedChunks(chunks) {
  return [...(chunks || [])].sort((a, b) => a.start - b.start || a.page - b.page);
}

// Offsets describe extracted source spans, not indices into reconstructed chunk text.
// Keep chunks intact and resolve all highlights inside one identified chunk.
export function resolveSource(chunks, { chunkId = "", hash = "", quote = "" }) {
  const needle = quote.trim();
  if (!chunkId && !needle && !hash) return { target: null, error: "" };
  let target;
  if (chunkId) {
    target = chunks.find((item) => String(item.chunk_id) === chunkId);
    if (!target)
      return { target: null, error: "定位失败：引用片段不存在，可能已被删除或重新解析。" };
  } else {
    const matches = chunks.filter((item) => needle && String(item.text || "").includes(needle));
    if (matches.length > 1)
      return { target: null, error: "定位失败：引句对应多个片段，请从引用卡片打开指定片段。" };
    target = matches[0];
    if (!target) return { target: null, error: "定位失败：文档中未找到该引句。" };
  }
  if (hash && target.chunk_hash !== hash) {
    return { target: null, error: "无法核验来源：片段指纹不匹配，原文可能已发生变化。" };
  }
  const text = String(target.text || "");
  const at = needle ? text.indexOf(needle) : 0;
  if (at < 0) return { target: null, error: "定位失败：指定片段中的文本与引用不匹配。" };
  if (needle && text.indexOf(needle, at + 1) >= 0) {
    return { target: null, error: "定位失败：引句在指定片段中出现多次，无法确认具体位置。" };
  }
  return { target, start: at, end: needle ? at + needle.length : text.length, error: "" };
}
