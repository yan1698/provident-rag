import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { IcoTrash } from "../icons";
import { fmtTime, TASK_KIND_LABEL } from "../lib";
import "./History.css";

// Both existing APIs paginate by offset. unique=false retains repeated questions.
async function readAll(endpoint, signal, onBatch) {
  const results = [];
  for (let offset = 0; ; offset += 100) {
    const data = await api(
      `${endpoint}${endpoint.includes("?") ? "&" : "?"}limit=100&offset=${offset}`,
      { signal }
    );
    const batch = data.results || [];
    results.push(...batch);
    onBatch([...results]);
    if (batch.length < 100) return;
  }
}

export default function Traces() {
  const [traces, setTraces] = useState([]),
    [runs, setRuns] = useState([]),
    [keyword, setKeyword] = useState("");
  const [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [deleting, setDeleting] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setError("");
    setTraces([]);
    setRuns([]);
    Promise.allSettled([
      readAll("/traces?unique=false", controller.signal, (batch) => {
        if (!cancelled) setTraces(batch);
      }),
      readAll("/agent", controller.signal, (batch) => {
        if (!cancelled) setRuns(batch);
      }),
    ]).then((results) => {
      if (cancelled) return;
      setLoading(false);
      const failed = results.filter((row) => row.status === "rejected");
      if (failed.length)
        setError(`部分历史读取失败：${failed.map((row) => row.reason.message).join("；")}`);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [revision]);

  const records = useMemo(() => {
    const sessions = new Map(),
      agents = [];
    for (const run of runs) {
      if (!run.session_id) {
        agents.push({
          key: `run:${run.id}`,
          kind: "run",
          id: run.id,
          title: run.goal,
          search: run.goal,
          date: run.created_at,
          label: `${TASK_KIND_LABEL[run.task_kind] || "Agent"} · 独立运行`,
          href: `/agent/${encodeURIComponent(run.id)}`,
        });
        continue;
      }
      const prior = sessions.get(run.session_id);
      if (!prior)
        sessions.set(run.session_id, {
          key: `session:${run.session_id}`,
          kind: "session",
          id: run.session_id,
          title: run.goal,
          search: run.goal,
          date: run.created_at,
          first: run.created_at,
          count: 1,
          href: `/agent/session/${encodeURIComponent(run.session_id)}`,
        });
      else {
        prior.count++;
        prior.search += ` ${run.goal || ""}`;
        if (new Date(run.created_at) > new Date(prior.date)) prior.date = run.created_at;
        if (new Date(run.created_at) < new Date(prior.first)) {
          prior.first = run.created_at;
          prior.title = run.goal;
        }
      }
    }
    return [
      ...agents,
      ...[...sessions.values()].map((row) => ({ ...row, label: `Agent 会话 · ${row.count} 轮` })),
      ...traces.map((row) => ({
        key: `query:${row.id}`,
        kind: "query",
        id: row.id,
        title: row.query,
        search: row.query,
        date: row.created_at,
        label:
          row.answer_source === "general_explanation"
            ? "通用知识解释 · 独立记录"
            : "普通问答 · 独立记录",
        href: `/agent?trace=${encodeURIComponent(row.id)}`,
      })),
    ].sort((a, b) => (new Date(b.date).getTime() || 0) - (new Date(a.date).getTime() || 0));
  }, [runs, traces]);
  const needle = keyword.trim().toLowerCase(),
    shown = records.filter(
      (row) =>
        !needle ||
        String(row.search || "")
          .toLowerCase()
          .includes(needle)
    );
  const remove = async (row) => {
    if (deleting) return;
    if (
      !window.confirm(
        `确定删除「${String(row.title || "历史记录").slice(0, 60)}」？${row.kind === "session" ? "其中的所有轮次会一起删除。" : ""}删除后不可恢复。`
      )
    )
      return;
    setDeleting(row.key);
    setError("");
    try {
      await api(
        row.kind === "query"
          ? `/traces/${encodeURIComponent(row.id)}`
          : row.kind === "session"
            ? `/agent/session/${encodeURIComponent(row.id)}`
            : `/agent/${encodeURIComponent(row.id)}`,
        { method: "DELETE" }
      );
      if (row.kind === "query") setTraces((prev) => prev.filter((item) => item.id !== row.id));
      else
        setRuns((prev) =>
          prev.filter((item) =>
            row.kind === "session" ? item.session_id !== row.id : item.id !== row.id
          )
        );
    } catch (err) {
      setError(`删除失败：${err.message}`);
    } finally {
      setDeleting("");
    }
  };

  return (
    <div className="history-page">
      <section className="hero history-heading">
        <div>
          <h1>历史记录</h1>
          <p>打开记录，继续查看或提问。</p>
        </div>
        <Link className="btn ghost" to="/agent">
          返回 Agent
        </Link>
      </section>
      <section className="panel">
        <div className="history-search">
          <label htmlFor="history-search">搜索历史</label>
          <input
            id="history-search"
            type="search"
            value={keyword}
            placeholder="搜索问题或任务"
            onChange={(e) => setKeyword(e.target.value)}
          />
          <button
            type="button"
            className="btn ghost"
            disabled={loading || Boolean(deleting)}
            onClick={() => setRevision((n) => n + 1)}
          >
            刷新记录
          </button>
        </div>
        <p className="tiny" role="status">
          {loading ? "正在读取全部历史…" : `${shown.length} 条${needle ? "匹配记录" : "历史记录"}`}{" "}
          · 普通问答独立保存，Agent 按真实会话恢复。
        </p>
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        <div className="list" id="trace-list">
          {shown.map((row) => (
            <div className="doc-row history-row" key={row.key} data-history-key={row.key}>
              <Link className="doc-link" to={row.href}>
                <div className="doc-main">
                  <strong>{row.title || "未命名记录"}</strong>
                  <p className="tiny">
                    {row.label}
                    {row.date ? ` · ${fmtTime(row.date)}` : ""}
                  </p>
                </div>
                <span className="tiny">查看</span>
              </Link>
              <button
                type="button"
                className="doc-del"
                title="删除记录"
                aria-label={`删除 ${row.title || "历史记录"}`}
                disabled={loading || Boolean(deleting)}
                onClick={() => remove(row)}
              >
                {deleting === row.key ? "…" : IcoTrash}
              </button>
            </div>
          ))}
        </div>
        {!loading && !shown.length ? (
          <p className="lede">{needle ? "没有匹配的历史记录。" : "还没有历史记录。"}</p>
        ) : null}
      </section>
    </div>
  );
}
