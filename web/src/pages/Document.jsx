import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { fmtTime } from "../lib";
import { chunkAnchor, orderedChunks, resolveSource, sourceHref } from "../sourceLocation";

function highlightQuote(text, start, end) {
  const raw = String(text || "");
  return (
    <>
      {raw.slice(0, start)}
      <mark id="hl">{raw.slice(start, end)}</mark>
      {raw.slice(end)}
    </>
  );
}

export default function DocumentPage() {
  const { docId } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sections, setSections] = useState([]);
  const hl = params.get("hl") || "";
  const chunkId = params.get("chunk") || "";
  const hash = params.get("hash") || "";

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError("");
    setSections([]);
    api(`/documents/${docId}/sections`)
      .then((payload) => {
        if (!cancelled) setSections(payload.sections || []);
      })
      .catch(() => {});
    api(`/documents/${docId}`)
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err) => {
        if (!cancelled)
          setError(
            err.status === 404
              ? "来源不存在或无权访问，无法定位引用。"
              : `无法加载来源：${err.message}`
          );
      });
    return () => {
      cancelled = true;
    };
  }, [docId]);

  const chunks = useMemo(() => orderedChunks(data?.chunks), [data]);
  const location = useMemo(
    () => resolveSource(chunks, { chunkId, hash, quote: hl }),
    [chunks, chunkId, hash, hl]
  );

  useEffect(() => {
    if (!location.target) return;
    const node = document.getElementById(chunkAnchor(location.target.chunk_id));
    if (node) node.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [location]);

  return (
    <>
      <section className="hero">
        <p>
          <Link to="/library">← 知识库</Link>
        </p>
        <h1 id="title">{data ? data.display_title || data.filename : "查阅文档"}</h1>
        <p className="lede" id="meta">
          {data ? (
            <>
              {data.display_title && data.display_title !== data.filename
                ? `原文件 ${data.filename} · `
                : ""}
              {data.collection_name || "待归类"}
              {data.doc_type_label ? ` · ${data.doc_type_label}` : ""} · {data.chunk_count}{" "}
              个片段已切分 · {data.embedded_count} 已向量化 · {fmtTime(data.created_at)} ·{" "}
              <Link to={`/graph?doc=${docId}`}>在图谱中查看</Link>
              {" · "}
              <Link to="/agent">去提问</Link>
            </>
          ) : (
            "加载中…"
          )}
        </p>
        {data?.summary ? <p className="tiny">归档摘要（不作为引用）：{data.summary}</p> : null}
        {data ? (
          <div className="row" id="doc-toolbar">
            <button
              type="button"
              className="btn bad"
              id="doc-del"
              disabled={busy}
              onClick={async () => {
                if (
                  !window.confirm(
                    `确定从知识库删除「${data.filename}」？片段、向量与概念关系会一并清除，且不可恢复。`
                  )
                )
                  return;
                setBusy(true);
                try {
                  await api(`/documents/${docId}`, { method: "DELETE" });
                  navigate("/library");
                } catch (err) {
                  setBusy(false);
                  setError(err.message);
                }
              }}
            >
              删除文档
            </button>
          </div>
        ) : null}
      </section>
      <section className="panel reader" id="box">
        {sections.length ? (
          <label className="tiny">
            章节
            <select
              aria-label="章节"
              value=""
              onChange={(event) => {
                const section = sections.find((s) => s.id === event.target.value);
                const item = chunks.find((c) => c.chunk_id === section?.chunk_ids?.[0]);
                if (item) navigate(sourceHref({ ...item, doc_id: docId }));
              }}
            >
              <option value="">选择章节定位原文</option>
              {sections.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.path.slice(0, -1).join(" / ")}
                  {s.path.length > 1 ? " / " : ""}
                  {s.label || s.title}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        {data && location.error ? (
          <p className="error" role="alert" data-locator-status="failed">
            {location.error}
          </p>
        ) : null}
        {data && location.target ? (
          <p className="tiny" role="status" data-locator-status="located">
            已定位到引用片段。
          </p>
        ) : null}
        {data && chunks.length ? (
          <article className="reader-full" id="doc-full">
            <p className="tiny">按提取片段展示；相邻片段的重叠文字会保留。</p>
            {chunks.map((item) => {
              const selected = location.target?.chunk_id === item.chunk_id;
              return (
                <section
                  key={item.chunk_id}
                  id={chunkAnchor(item.chunk_id)}
                  className={`reader-chunk${selected ? " on" : ""}`}
                  data-chunk-id={item.chunk_id}
                >
                  <p className="tiny">
                    片段 {item.index} · 第 {item.page} 页
                  </p>
                  <div className="reader-chunk-text">
                    {selected ? highlightQuote(item.text, location.start, location.end) : item.text}
                  </div>
                </section>
              );
            })}
          </article>
        ) : data ? (
          <p className="lede">这篇文档还没有切出片段。</p>
        ) : error ? null : (
          "加载中…"
        )}
      </section>
    </>
  );
}
