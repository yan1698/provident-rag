import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Navigate, useLocation, useNavigate, useParams } from "react-router-dom";
import { api, apiStream } from "../api";
import { useAuth } from "../auth";
import { AGENT_STEP_LABEL, citedEvidence } from "../lib";
import { chatKey, readChat, writeChat } from "../chatStorage";
import { readAgentDraft } from "../agentHandoff";
import AgentComposer from "./AgentComposer";
import AgentResult, { Gaps, Timeline, ReadingCoverage } from "./AgentResult";
import ResultView, { EvidenceList } from "./ResultView";
import "./AgentDesk.css";

const TERMINAL = new Set([
  "finished",
  "refused",
  "error",
  "cancelled",
  "crashed",
  "failed",
  "timeout",
]);
const STATE_LABEL = {
  queued: "等待执行",
  running: "正在执行",
  finished: "已完成",
  refused: "证据不足",
  error: "执行失败",
  failed: "执行失败",
  cancelled: "已取消",
  crashed: "任务中断",
  timeout: "执行超时",
};
const rowKey = (row) =>
  `${row._kind}:${row._kind === "query" ? row.trace_id || row._viewId : row.id}`;
const queryRow = (row) => ({
  ...row,
  _kind: "query",
  trace_id: row.trace_id || row.id,
  _viewId:
    row.trace_id || row.id || globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
  scope: row.scope || row.diagnostics?.scope,
});
const agentRow = (row) => ({ ...row, _kind: "agent" });
const refsOf = (rows) =>
  rows
    .map((row) => ({ kind: row._kind, id: row._kind === "query" ? row.trace_id : row.id }))
    .filter((row) => row.id);
const uniqueRows = (rows) => [...new Map(rows.map((row) => [rowKey(row), row])).values()];

function TaskDetails({ detail, onClose, onTab, onSelect }) {
  const dialog = useRef(null),
    row = detail?.row;
  useEffect(() => {
    if (detail && !dialog.current.open) dialog.current.showModal();
    if (!detail && dialog.current.open) dialog.current.close();
  }, [Boolean(detail)]);
  useEffect(() => {
    if (detail?.tab === "sources" && detail.selected)
      dialog.current
        ?.querySelector(`[data-index="${CSS.escape(String(detail.selected))}"]`)
        ?.scrollIntoView({ block: "nearest" });
  }, [detail]);
  const longform = ["synthesis", "digest"].includes(row?.task_kind);
  const tabs = [
    ["sources", "原文来源"],
    ...(longform && row?.steps?.length ? [["steps", "执行过程"]] : []),
    ...(longform && (row?.outline?.reading || row?.outline?.evidence_selection)
      ? [["coverage", "阅读覆盖"]]
      : []),
    ...(row?.gaps?.length ? [["gaps", "任务缺口"]] : []),
  ];
  return (
    <dialog
      ref={dialog}
      className="chat-details panel"
      onCancel={onClose}
      onClose={onClose}
      onClick={(e) => {
        if (e.target === dialog.current) onClose();
      }}
    >
      {row ? (
        <div className="chat-details-content">
          <div className="chat-detail-head">
            <h2>回答详情</h2>
            <button type="button" className="btn ghost" onClick={onClose}>
              关闭详情
            </button>
          </div>
          <p className="tiny">{row.query || row.goal}</p>
          <div className="agent-tabs" role="tablist" aria-label="回答详情">
            {tabs.map(([value, label]) => (
              <button
                type="button"
                role="tab"
                key={value}
                aria-selected={detail.tab === value}
                aria-controls="chat-detail-content"
                onClick={() => onTab(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <div
            id="chat-detail-content"
            role="tabpanel"
            aria-label={tabs.find(([v]) => v === detail.tab)?.[1]}
          >
            {detail.tab === "sources" ? (
              <>
                <p className="tiny">
                  {row.answer_source === "general_explanation"
                    ? "模型通用解释，没有文档证据，未经文档核验。"
                    : "仅显示回答实际引用的片段，引用编号与原文身份保持一致。"}
                </p>
                <EvidenceList
                  evidence={row.answer_source === "general_explanation" ? [] : citedEvidence(row)}
                  selected={detail.selected}
                  onSelect={onSelect}
                  integrity={row.citation_integrity}
                  emptyHint="这次没有可定位的文档引用。"
                />
              </>
            ) : null}
            {detail.tab === "steps" ? <Timeline steps={row.steps} open /> : null}
            {detail.tab === "coverage" ? <ReadingCoverage data={row} /> : null}
            {detail.tab === "gaps" ? <Gaps gaps={row.gaps} /> : null}
          </div>
        </div>
      ) : null}
    </dialog>
  );
}

export default function AgentDesk() {
  const { id, sessionId: sessionParam } = useParams(),
    { user } = useAuth();
  const location = useLocation(),
    navigate = useNavigate();
  const params = new URLSearchParams(location.search),
    traceId = params.get("trace") || "",
    requestedRun = params.get("run") || id || "";
  const handoff = useMemo(() => readAgentDraft(location.search), [location.search]);
  const record = traceId
    ? `query:${traceId}`
    : sessionParam
      ? `session:${sessionParam}:${requestedRun}`
      : id
        ? `agent:${id}`
        : `new:${location.search}`;
  const storageKey = chatKey(user?.id, record);
  const [rows, setRows] = useState([]),
    [loading, setLoading] = useState(true),
    [loadedRecord, setLoadedRecord] = useState(""),
    [error, setError] = useState(""),
    [warning, setWarning] = useState("");
  const [sending, setSending] = useState(false),
    [pending, setPending] = useState(""),
    [detail, setDetail] = useState(null),
    [revision, setRevision] = useState(0);
  const [answerSource, setAnswerSource] = useState("document_only"),
    [explanation, setExplanation] = useState(null),
    [original, setOriginal] = useState(null);
  const composerRef = useRef(null),
    scrollRef = useRef(null),
    accepted = useRef(null);
  const latest = rows.at(-1),
    live = rows.find((row) => row._kind === "agent" && !TERMINAL.has(row.status)),
    liveId = live?.id || "";
  const draftKey =
    latest?._kind === "agent" && latest.session_id
      ? `${chatKey(user?.id, `session:${latest.session_id}:`)}:draft`
      : `${storageKey}:draft`;
  const context =
    explanation ||
    (latest?.answer_source !== "general_explanation"
      ? latest
      : rows.find(
          (row) =>
            (latest?.source_trace_id &&
              row._kind === "query" &&
              row.trace_id === latest.source_trace_id) ||
            (latest?.source_run_id && row._kind === "agent" && row.id === latest.source_run_id)
        )) ||
    original;
  const scope = context?.scope || (!latest ? handoff.scope : null);

  useEffect(() => {
    document.body.classList.add("agent-desk-mode");
    return () => document.body.classList.remove("agent-desk-mode");
  }, []);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    setWarning("");
    setDetail(null);
    setExplanation(null);
    setOriginal(null);
    setAnswerSource("document_only");
    const boot = async () => {
      try {
        // Navigation after a successful send only appends the returned record.
        if (accepted.current?.record === record) {
          const next = accepted.current.rows;
          accepted.current = null;
          if (!cancelled) {
            setRows(next);
            setAnswerSource(next.at(-1)?.answer_source || "document_only");
          }
          return;
        }
        setRows([]);
        if (!id && !sessionParam && !traceId) return;
        let current = [],
          selected;
        if (traceId) {
          selected = {
            ...queryRow(await api(`/traces/${encodeURIComponent(traceId)}`)),
            _restored: true,
          };
          current = [selected];
        } else {
          const first = id ? agentRow(await api(`/agent/${encodeURIComponent(id)}`)) : null;
          const session = sessionParam || first?.session_id;
          current = first ? [first] : [];
          if (session) {
            const listing = await api(`/agent/session/${encodeURIComponent(session)}`);
            current = await Promise.all(
              (listing.runs || []).map((item) =>
                item.id === first?.id
                  ? first
                  : api(`/agent/${encodeURIComponent(item.id)}`).then(agentRow)
              )
            );
          }
          selected = current.find((row) => row.id === requestedRun) || current.at(-1);
        }
        // These are browser display references, never cached answers or model memory.
        const session = current.find((row) => row._kind === "agent" && row.session_id)?.session_id;
        const saved =
          readChat(storageKey, null) ||
          (session ? readChat(chatKey(user?.id, `session:${session}:`), []) : []);
        const extra = await Promise.allSettled(
          (Array.isArray(saved) ? saved : []).map(async (ref) => {
            const present = current.find((row) => rowKey(row) === `${ref.kind}:${ref.id}`);
            if (present) return present;
            if (ref.kind === "query")
              return {
                ...queryRow(await api(`/traces/${encodeURIComponent(ref.id)}`)),
                _restored: true,
              };
            if (ref.kind === "agent")
              return agentRow(await api(`/agent/${encodeURIComponent(ref.id)}`));
            throw new Error("无效的历史记录身份");
          })
        );
        let restored = uniqueRows([
          ...extra.filter((item) => item.status === "fulfilled").map((item) => item.value),
          ...current,
        ]);
        if (restored.every((row) => row.created_at))
          restored.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
        if (cancelled) return;
        setRows(restored);
        selected = restored.at(-1);
        if (extra.some((item) => item.status === "rejected"))
          setWarning("部分先前记录已删除或不可访问，未恢复其内容。当前记录仍可使用。");
        if (selected?.answer_source === "general_explanation") {
          setAnswerSource("general_explanation");
          const sourceId = selected.source_trace_id || selected.source_run_id;
          if (sourceId)
            try {
              const source = selected.source_trace_id
                ? queryRow(await api(`/traces/${encodeURIComponent(sourceId)}`))
                : agentRow(await api(`/agent/${encodeURIComponent(sourceId)}`));
              if (!cancelled) setOriginal(source);
            } catch (err) {
              if (!cancelled) setError(`原文回答暂不可用，已保留历史解释。${err.message}`);
            }
        }
      } catch (err) {
        if (!cancelled) setError(`历史恢复失败：${err.message}`);
      } finally {
        if (!cancelled) {
          setLoadedRecord(record);
          setLoading(false);
        }
      }
    };
    boot();
    return () => {
      cancelled = true;
    };
  }, [record, storageKey]);
  useEffect(() => {
    if (!loading && loadedRecord === record && rows.length) {
      writeChat(storageKey, refsOf(rows));
      if (latest?._kind === "agent" && latest.session_id)
        writeChat(chatKey(user?.id, `session:${latest.session_id}:`), refsOf(rows));
    }
  }, [storageKey, record, loadedRecord, loading, rows]);
  useEffect(() => {
    if (!liveId) return;
    let cancelled = false,
      timer;
    const controller = new AbortController();
    const apply = (row) => {
      if (cancelled || !row?.id) return;
      setRows((prev) =>
        prev.map((item) =>
          item._kind === "agent" && item.id === row.id
            ? agentRow({
                ...row,
                delivery:
                  item.delivery === "deferred" && row.status === "queued"
                    ? "deferred"
                    : row.delivery,
              })
            : item
        )
      );
      setDetail((prev) =>
        prev?.row?._kind === "agent" && prev.row.id === row.id
          ? { ...prev, row: agentRow(row) }
          : prev
      );
    };
    const poll = async () => {
      try {
        const row = await api(`/agent/${liveId}`, { signal: controller.signal });
        if (cancelled) return;
        apply(row);
        if (!TERMINAL.has(row.status)) timer = setTimeout(poll, 1000);
      } catch {
        if (!cancelled) timer = setTimeout(poll, 4000);
      }
    };
    apiStream(`/agent/${liveId}/stream`, {
      signal: controller.signal,
      onEvent: (event, row) => {
        if (event === "snapshot") apply(row);
      },
    }).catch(() => {
      if (!cancelled) poll();
    });
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [liveId]);
  useEffect(() => {
    if (!loading)
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [loading, rows.length, latest?.status, sending]);

  const created = ({ kind, data, input }) => {
    const row = kind === "query" ? queryRow(data) : agentRow(data),
      next = uniqueRows([...rows, row]);
    setExplanation(null);
    setOriginal(null);
    setAnswerSource(row.answer_source || "document_only");
    setRows(next);
    const nextRecord =
      kind === "agent" ? `agent:${data.id}` : data.trace_id ? `query:${data.trace_id}` : record;
    if (nextRecord === record) return;
    writeChat(chatKey(user?.id, nextRecord), refsOf(next));
    const nextDraft =
      kind === "agent" && data.session_id
        ? `${chatKey(user?.id, `session:${data.session_id}:`)}:draft`
        : `${chatKey(user?.id, nextRecord)}:draft`;
    writeChat(nextDraft, { ...input, query: input?.query || "" });
    accepted.current = { record: nextRecord, rows: next };
    navigate(
      kind === "agent"
        ? `/agent/${encodeURIComponent(data.id)}`
        : `/agent?trace=${encodeURIComponent(data.trace_id)}`
    );
  };
  const newConversation = () => {
    writeChat(chatKey(user?.id, "new:"), null);
    writeChat(`${chatKey(user?.id, "new:")}:draft`, null);
    setRows([]);
    setError("");
    setWarning("");
    setDetail(null);
    setExplanation(null);
    setOriginal(null);
    setAnswerSource("document_only");
    setRevision((n) => n + 1);
    navigate("/agent");
  };
  const cancelRun = async () => {
    try {
      const row = await api(`/agent/${liveId}/cancel`, { method: "POST" });
      setRows((prev) => prev.map((item) => (item.id === row.id ? agentRow(row) : item)));
    } catch (err) {
      setError(err.message);
    }
  };
  const showDetails = (row, tab = "sources", selected = "") =>
    setDetail({ row, tab, selected: String(selected) });
  const explain = (row) => {
    setExplanation(row);
    setAnswerSource("general_explanation");
    composerRef.current?.fill(`请解释这个主题的概念、原理和背景：${row.query || row.goal}`);
  };
  const changeSource = (value) => {
    setExplanation(value === "general_explanation" && context ? context : null);
    setAnswerSource(value);
  };
  const canExplain = context && (context._kind === "query" ? context.trace_id : context.session_id);
  if (params.has("history")) return <Navigate replace to="/traces" />;

  return (
    <div className="agent-chat">
      <div className="chat-toolbar">
        <span>Agent</span>
        <button
          type="button"
          className="btn ghost"
          disabled={sending || Boolean(live)}
          onClick={newConversation}
        >
          新对话
        </button>
      </div>
      <div className="chat-messages" ref={scrollRef} aria-label="对话内容" aria-busy={loading}>
        {loading ? <p role="status">正在恢复已保存的回答…</p> : null}
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
        {warning ? (
          <p className="tiny" role="status">
            {warning}
          </p>
        ) : null}
        {!loading && !rows.length && !error && !sending ? (
          <div className="chat-welcome">
            <h1>今天想了解什么？</h1>
            <p>从你的文档开始提问</p>
          </div>
        ) : null}
        {rows.map((row, index) => {
          const query = row._kind === "query",
            running = !query && !TERMINAL.has(row.status),
            longform = ["synthesis", "digest"].includes(row.task_kind),
            isLatest = index === rows.length - 1;
          const selected = detail && rowKey(detail.row) === rowKey(row) ? detail.selected : "";
          return (
            <section
              className="chat-exchange"
              key={rowKey(row)}
              data-run={query ? undefined : row.id}
              data-trace={query ? row.trace_id || "unsaved" : undefined}
            >
              <div className="chat-user" aria-label="你的问题">
                <p>{row.query || row.goal}</p>
              </div>
              <article className="chat-assistant" aria-label="AI 回答">
                <p className="chat-answer-label">
                  Provident{" "}
                  <span>
                    {row.answer_source === "general_explanation"
                      ? "通用知识解释"
                      : query || !longform
                        ? "文档问答"
                        : row.task_kind === "digest"
                          ? "按篇阅读"
                          : "综合分析"}
                    {!query ? ` · ${STATE_LABEL[row.status] || row.status}` : ""}
                  </span>
                </p>
                {running ? (
                  <div className="chat-progress" role="status">
                    <span className="dot" />
                    {row.steps?.length
                      ? `${AGENT_STEP_LABEL[row.steps.at(-1).action] || "正在处理"} · 第 ${row.steps.length} 步`
                      : row.delivery === "deferred"
                        ? "任务已保存，等待执行资源…"
                        : "任务已提交，正在处理…"}
                    <button type="button" className="btn ghost" onClick={cancelRun}>
                      取消任务
                    </button>
                  </div>
                ) : query ? (
                  <ResultView
                    data={row}
                    compact
                    mode={row._restored ? "replay" : "live"}
                    hideEvidence
                    evidenceSelection={{
                      selected,
                      open: false,
                      select: (n) => showDetails(row, "sources", n),
                    }}
                    onEditQuery={() => composerRef.current?.fill(row.query)}
                    onEditScope={() => composerRef.current?.editScope()}
                  />
                ) : (
                  <AgentResult
                    data={row}
                    compact
                    showReadingDetails={false}
                    selectedCite={selected}
                    onSelectCite={(n) => showDetails(row, "sources", n)}
                    onPickParagraph={
                      isLatest ? (text) => composerRef.current?.fill(text) : undefined
                    }
                    pickDisabled={Boolean(live) || sending}
                  />
                )}
                <div className="chat-actions">
                  {row.answer_source !== "general_explanation" && citedEvidence(row).length ? (
                    <button type="button" className="linkish" onClick={() => showDetails(row)}>
                      原文来源
                    </button>
                  ) : null}
                  {longform && (row.steps?.length || row.outline?.reading || row.gaps?.length) ? (
                    <button
                      type="button"
                      className="linkish"
                      onClick={() =>
                        showDetails(
                          row,
                          row.steps?.length ? "steps" : row.outline?.reading ? "coverage" : "gaps"
                        )
                      }
                    >
                      执行详情
                    </button>
                  ) : null}
                  {isLatest &&
                  !running &&
                  row.answer_source !== "general_explanation" &&
                  canExplain ? (
                    <button
                      type="button"
                      className="linkish"
                      disabled={sending}
                      onClick={() => explain(row)}
                    >
                      解释一下
                    </button>
                  ) : null}
                  {!query && !running && !row.answer ? (
                    <p className="tiny">
                      {row.error || "此任务没有保存最终回答，可查看已有执行详情。"}
                    </p>
                  ) : null}
                  {query && !row.trace_id ? (
                    <p className="tiny">此回答未保存，刷新后无法恢复。</p>
                  ) : null}
                </div>
              </article>
            </section>
          );
        })}
        {sending && pending ? (
          <section className="chat-exchange chat-pending">
            <div className="chat-user" aria-label="你的问题">
              <p>{pending}</p>
            </div>
            <p role="status">正在提交问题并等待回答…</p>
          </section>
        ) : null}
      </div>
      <div className="chat-input-dock">
        {!loading && loadedRecord === record ? (
          <AgentComposer
            key={`${storageKey}:${revision}`}
            draftKey={draftKey}
            ref={composerRef}
            prefill={handoff.goal || location.state?.askQuery || ""}
            defaultRoute={handoff.mode}
            answerSource={answerSource}
            onAnswerSource={changeSource}
            initialScope={scope}
            scopeConfirmed={latest ? Boolean(scope?.mode) : handoff.scopeConfirmed}
            initialTarget={context?.target_document_id || (!latest ? handoff.targetDocumentId : "")}
            reference={context?._kind === "agent" ? context : null}
            sourceTrace={context?._kind === "query" ? context : null}
            disabled={Boolean(live) || Boolean(error)}
            onSubmitting={setPending}
            onBusyChange={(value) => {
              setSending(value);
              if (!value) setPending("");
            }}
            onCreated={created}
          />
        ) : null}
        {latest ? (
          <p className="chat-context-note">
            {answerSource === "general_explanation" ? (
              <>
                通用解释不提供文档证据。
                <button
                  type="button"
                  className="linkish"
                  onClick={() => {
                    setExplanation(null);
                    setAnswerSource("document_only");
                    composerRef.current?.focus();
                  }}
                >
                  返回原文继续追问
                </button>
              </>
            ) : context?._kind === "agent" && context.session_id ? (
              "同范围的智能追问可沿用此 Agent 会话；快速问答或改变范围会独立检索。"
            ) : (
              "快速问答逐次独立检索，不继承前文记忆；请写出完整问题。"
            )}
          </p>
        ) : null}
        {!latest && handoff.traceId ? (
          <p className="tiny">
            已带入问题与范围，将独立执行。
            <Link to={`/agent?trace=${encodeURIComponent(handoff.traceId)}`}>查看原回答</Link>
          </p>
        ) : null}
      </div>
      <TaskDetails
        detail={detail}
        onClose={() => setDetail(null)}
        onTab={(tab) => setDetail((prev) => ({ ...prev, tab }))}
        onSelect={(selected) => setDetail((prev) => ({ ...prev, selected }))}
      />
    </div>
  );
}
