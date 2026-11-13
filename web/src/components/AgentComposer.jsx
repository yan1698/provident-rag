import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { api } from "../api";
import { submitAgentRun } from "../submitRun";
import ScopeFields from "./ScopeFields";
import { readChat, writeChat } from "../chatStorage";
import "./ConversationComposer.css";

function AskIcon({ name, ...props }) {
  const paths = {
    document: (
      <>
        <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
        <path d="M14 3v6h6M8 13h8M8 17h5" />
      </>
    ),
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 4 4" />
      </>
    ),
    steps: (
      <>
        <path d="M10 6h10M10 12h10M10 18h10M4 5v3M3 5h1M3 12h2l-2 2h2M3 17h2v3H3" />
      </>
    ),
    book: (
      <>
        <path d="M12 5v15M12 6C9 4 5 4 3 5v14c3-1 6-1 9 1 3-2 6-2 9-1V5c-2-1-6-1-9 1Z" />
      </>
    ),
    compare: (
      <>
        <rect x="3" y="4" width="7" height="16" rx="2" />
        <rect x="14" y="4" width="7" height="16" rx="2" />
        <path d="M6 8h1M17 8h1M6 12h1M17 12h1" />
      </>
    ),
    settings: (
      <>
        <path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h9M17 18h3" />
        <circle cx="15" cy="6" r="2" />
        <circle cx="9" cy="12" r="2" />
        <circle cx="15" cy="18" r="2" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4L12 3Z" />
      </>
    ),
    send: (
      <>
        <path d="M12 19V5m-6 6 6-6 6 6" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
  };
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name]}
    </svg>
  );
}

// One route decision and one executor per send. The server owns classification and identity checks.
const AgentComposer = forwardRef(function AgentComposer(
  {
    prefill = "",
    defaultRoute = "auto",
    initialScope = null,
    initialTarget = "",
    scopeConfirmed = true,
    answerSource = "document_only",
    onAnswerSource,
    reference = null,
    sourceTrace = null,
    selectedContext = "",
    disabled = false,
    onCreated,
    onBusyChange,
    onSubmitting,
    draftKey,
  },
  ref
) {
  const [saved] = useState(() => readChat(draftKey, {}));
  const [query, setQuery] = useState(saved.query ?? prefill);
  const queryRef = useRef(query);
  queryRef.current = query;
  const [analysis, setAnalysis] = useState(saved.analysis ?? defaultRoute);
  const [scopeMode, setScopeMode] = useState(saved.scopeMode ?? initialScope?.mode ?? "auto");
  const [collectionIds, setCollectionIds] = useState(
    saved.collectionIds ?? initialScope?.collection_ids ?? []
  );
  const collectionId = collectionIds[0] || "";
  const [confirmed, setConfirmed] = useState(saved.confirmed ?? scopeConfirmed);
  const [targetDocumentId, setTargetDocumentId] = useState(saved.targetDocumentId ?? initialTarget);
  const [libraries, setLibraries] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [decision, setDecision] = useState(null);
  const [status, setStatus] = useState("");
  const [statusErr, setStatusErr] = useState(false);
  const [scopeError, setScopeError] = useState("");
  const [asking, setAsking] = useState(false);
  const locked = useRef(false),
    mounted = useRef(true),
    abortRef = useRef(null);
  const scopeKey = JSON.stringify(initialScope);
  useEffect(() => {
    if (saved.answerSource && saved.answerSource !== answerSource)
      onAnswerSource(saved.answerSource);
  }, []);
  const previousPrefill = useRef(prefill),
    previousDefaults = useRef({ defaultRoute, scopeKey, scopeConfirmed, initialTarget });
  useEffect(() => {
    if (previousPrefill.current !== prefill) setQuery(prefill || "");
    previousPrefill.current = prefill;
  }, [prefill]);
  useEffect(() => {
    const previous = previousDefaults.current;
    if (previous.defaultRoute !== defaultRoute) setAnalysis(defaultRoute);
    if (previous.scopeKey !== scopeKey || previous.scopeConfirmed !== scopeConfirmed) {
      setScopeMode(initialScope?.mode || "auto");
      setCollectionIds(initialScope?.collection_ids || []);
      setConfirmed(scopeConfirmed);
    }
    if (previous.initialTarget !== initialTarget) setTargetDocumentId(initialTarget || "");
    previousDefaults.current = { defaultRoute, scopeKey, scopeConfirmed, initialTarget };
  }, [defaultRoute, scopeKey, scopeConfirmed, initialTarget]);
  useEffect(() => {
    if (draftKey)
      writeChat(draftKey, {
        query,
        analysis,
        scopeMode,
        collectionIds,
        confirmed,
        targetDocumentId,
        answerSource,
      });
  }, [
    draftKey,
    query,
    analysis,
    scopeMode,
    collectionIds,
    confirmed,
    targetDocumentId,
    answerSource,
  ]);
  useEffect(() => {
    mounted.current = true;
    Promise.allSettled([api("/library"), api("/documents")]).then(([a, b]) => {
      if (!mounted.current) return;
      if (a.status === "fulfilled") setLibraries(a.value.collections || []);
      if (b.status === "fulfilled") setDocuments(b.value.results || []);
    });
    return () => {
      mounted.current = false;
      abortRef.current?.abort();
      onBusyChange?.(false);
    };
  }, []);
  const chooseScope = (mode) => {
    setConfirmed(true);
    setScopeMode(mode);
    setScopeError("");
    if (!["hard", "soft"].includes(mode)) setCollectionIds([]);
  };
  const chooseCollection = (id) => {
    setConfirmed(true);
    setCollectionIds(id ? [id] : []);
    setScopeError("");
  };
  const busy = asking || disabled;
  const readingRequired =
    answerSource === "document_only" &&
    (analysis === "digest" || decision?.actual === "needs_target");
  const modeLabel =
    answerSource === "general_explanation"
      ? "通用解释"
      : { auto: "智能选择", quick: "快速问答", synthesis: "综合分析", digest: "按篇阅读" }[
          analysis
        ];
  const libraryName =
    libraries.find((item) => String(item.id) === String(collectionId))?.name || "所选知识库";
  const scopeLabel =
    {
      auto: "范围自动选择",
      all: "全部文档",
      hard: `只查 · ${libraryName}`,
      soft: `${libraryName}及待归类`,
    }[scopeMode] + (collectionIds.length > 1 ? `等 ${collectionIds.length} 个库` : "");
  const source = reference?.session_id ? reference : sourceTrace;
  const fail = (text) => {
    setStatus(text);
    setStatusErr(true);
  };
  const ask = async (raw = query) => {
    const value = String(raw || "").trim();
    if (!value || busy || locked.current) return;
    if (answerSource === "document_only" && !confirmed) {
      setAdvancedOpen(true);
      fail("旧记录没有保存知识范围，请先重新选择范围。");
      return;
    }
    if (
      answerSource === "document_only" &&
      ["hard", "soft"].includes(scopeMode) &&
      !collectionIds.length
    ) {
      setAdvancedOpen(true);
      setScopeError("请选择知识库，或改为「全部文档」。");
      return;
    }
    if (
      answerSource === "general_explanation" &&
      (!source ||
        source.answer_source === "general_explanation" ||
        (!reference?.session_id && !sourceTrace?.trace_id && !sourceTrace?.id))
    ) {
      fail("请先完成并保存文档回答，再选择「解释一下」。");
      return;
    }
    if (answerSource === "document_only" && analysis === "digest" && !targetDocumentId) {
      setDecision({ actual: "needs_target" });
      fail("按篇阅读需要选择目标文档。");
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    let completed = false;
    locked.current = true;
    setAsking(true);
    onBusyChange?.(true);
    setStatus("");
    setStatusErr(false);
    setScopeError("");
    queryRef.current = "";
    setQuery("");
    onSubmitting?.(value);
    const input = { analysis, scopeMode, collectionIds, confirmed, targetDocumentId, answerSource };
    try {
      const sameScope =
        reference?.scope?.mode === scopeMode &&
        JSON.stringify([...(reference?.scope?.collection_ids || [])].sort()) ===
          JSON.stringify([...collectionIds].sort());
      const anchor =
        reference?.session_id &&
        (answerSource === "general_explanation" || (analysis !== "quick" && sameScope))
          ? reference
          : null;
      const planned = await api("/query/route", {
        method: "POST",
        signal: controller.signal,
        body: {
          query: value,
          analysis_mode: analysis,
          answer_source: answerSource,
          scope_mode: scopeMode,
          collection_ids: collectionIds,
          target_document_id: targetDocumentId || null,
          session_id: anchor?.session_id || null,
          source_run_id: anchor?.id || null,
        },
      });
      if (!mounted.current || controller.signal.aborted) return;
      setDecision(planned);
      if (planned.fallback_reason === "reading_target_missing") {
        setDecision({ ...planned, actual: "needs_target" });
        fail("请先选择阅读目标；尚未执行这个问题。");
        return;
      }
      if (anchor || ["synthesis", "digest"].includes(planned.actual)) {
        const body = {
          goal: value,
          task_kind: ["synthesis", "digest"].includes(planned.actual) ? planned.actual : "factoid",
          analysis_mode: analysis,
          answer_source: answerSource,
          scope_mode: scopeMode,
          collection_ids: collectionIds,
          target_document_id: targetDocumentId || null,
        };
        if (anchor) {
          body.session_id = anchor.session_id;
          body.source_run_id = anchor.id;
        }
        if (selectedContext) body.selected_context = selectedContext;
        const data = await submitAgentRun(api, body);
        if (mounted.current && !controller.signal.aborted) {
          completed = true;
          writeChat(draftKey, null);
          onCreated({ kind: "agent", data, input: { ...input, query: queryRef.current } });
        }
      } else {
        if (!["quick", "general_explanation"].includes(planned.actual))
          throw new Error("无法识别执行线路，请重试。");
        const body = {
          query: value,
          analysis_mode: analysis,
          answer_source: answerSource,
          scope_mode: scopeMode,
          collection_ids: collectionIds,
          target_document_id: targetDocumentId || null,
        };
        if (answerSource === "general_explanation") {
          body.source_trace_id = sourceTrace?.trace_id || sourceTrace?.id;
          body.selected_context = selectedContext;
        }
        const data = await api("/query", { method: "POST", body, signal: controller.signal });
        if (mounted.current && !controller.signal.aborted) {
          completed = true;
          writeChat(draftKey, null);
          onCreated({ kind: "query", data, input: { ...input, query: queryRef.current } });
        }
      }
    } catch (err) {
      if (mounted.current && err.name !== "AbortError") fail(err.message || "请求失败，请重试。");
    } finally {
      locked.current = false;
      if (mounted.current) {
        if (!completed && !queryRef.current) setQuery(value);
        setAsking(false);
        onBusyChange?.(false);
      }
    }
  };
  useImperativeHandle(ref, () => ({
    focus: () => document.getElementById("q")?.focus(),
    editScope: () => {
      setAdvancedOpen(true);
      requestAnimationFrame(() => document.querySelector("#ask-scope .agent-menu-button")?.focus());
    },
    fill: (text, mode) => {
      setQuery(text);
      if (mode) setAnalysis(mode);
      requestAnimationFrame(() => document.getElementById("q")?.focus());
    },
    send: ask,
  }));
  useEffect(() => {
    if (decision?.actual === "needs_target")
      document.querySelector('[aria-label="阅读目标"]')?.focus();
  }, [decision?.actual]);
  const progressText = "正在按所选方式检索并生成回答…";

  return (
    <div className="ask-page agent-input">
      <section className="panel composer ask-composer" aria-label="提问面板">
        <div className="ask-composer-top">
          <label className="ask-source">
            <AskIcon name={answerSource === "general_explanation" ? "spark" : "document"} />
            <span className="ask-sr-only">知识来源</span>
            <select
              aria-label="知识来源"
              disabled={busy}
              value={answerSource}
              onChange={(e) => onAnswerSource(e.target.value)}
            >
              <option value="document_only">我的文档</option>
              <option value="general_explanation">通用知识</option>
            </select>
          </label>
          <span className="ask-mode">
            <AskIcon name="spark" />
            {modeLabel}
          </span>
          {answerSource === "document_only" ? (
            <span className="ask-scope-summary">{scopeLabel}</span>
          ) : null}
        </div>
        {answerSource === "general_explanation" ? (
          <p className="ask-notice" role="status">
            通用知识解释 · 未检索文档，未经文档核验。
            {source ? "已绑定原文主题。" : "请先完成文档回答，再选择「解释一下」。"}
          </p>
        ) : null}
        <label className="ask-sr-only" htmlFor="q">
          你的问题
        </label>
        <textarea
          id="q"
          className="ask-query"
          value={query}
          placeholder={
            answerSource === "general_explanation"
              ? "想了解这个概念的什么？"
              : "用自己的话提问，让答案回到原文。"
          }
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.isComposing &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              ask();
            }
          }}
        />
        {selectedContext ? <p className="ask-context">所选背景：{selectedContext}</p> : null}
        {readingRequired ? (
          <div className="ask-reading">
            <AskIcon name="book" />
            <label htmlFor="ask-reading-target">阅读目标</label>
            <select
              id="ask-reading-target"
              aria-label="阅读目标"
              disabled={busy}
              value={targetDocumentId}
              onChange={(e) => {
                setTargetDocumentId(e.target.value);
                if (e.target.value && decision?.actual === "needs_target") {
                  setDecision(null);
                  setStatus("");
                  setStatusErr(false);
                }
              }}
            >
              <option value="">选择要阅读的文档</option>
              {documents.map((doc) => (
                <option key={doc.doc_id} value={doc.doc_id}>
                  {doc.display_title || doc.filename}
                </option>
              ))}
            </select>
          </div>
        ) : null}
        {answerSource === "document_only" && advancedOpen ? (
          <div id="ask-settings" className="ask-settings">
            <label className="ask-setting-field">
              分析方式
              <select
                aria-label="分析方式"
                disabled={busy}
                value={analysis}
                onChange={(e) => setAnalysis(e.target.value)}
              >
                <option value="auto">智能选择</option>
                <option value="quick">快速问答</option>
                <option value="synthesis">综合分析</option>
                <option value="digest">按篇阅读</option>
              </select>
            </label>
            {libraries.length ? (
              <div className="ask-scope" id="ask-scope">
                <span className="ask-field-caption">文档范围</span>
                <ScopeFields
                  libraries={libraries}
                  scopeMode={scopeMode}
                  onScopeMode={chooseScope}
                  collectionId={collectionId}
                  onCollectionId={(id) => {
                    chooseCollection(id);
                  }}
                  invalid={Boolean(scopeError)}
                />
                {scopeError ? (
                  <p className="error ask-scope-error" role="alert">
                    {scopeError}
                  </p>
                ) : null}
              </div>
            ) : null}
            <p className="ask-settings-help">
              智能选择会按问题安排查找、比较或阅读；你也可以指定方式。
            </p>
          </div>
        ) : null}
        <div className="ask-composer-bottom">
          {answerSource === "document_only" ? (
            <button
              type="button"
              className={`ask-settings-toggle${advancedOpen ? " is-open" : ""}`}
              aria-label="高级设置"
              aria-expanded={advancedOpen}
              aria-controls="ask-settings"
              onClick={() => setAdvancedOpen((value) => !value)}
            >
              <AskIcon name="settings" />
              高级设置<span aria-hidden="true">{advancedOpen ? "−" : "+"}</span>
            </button>
          ) : (
            <span className="ask-source-caption">模型知识 · 非文档证据</span>
          )}
          <span className="ask-keyboard">Enter 发送 · Shift+Enter 换行</span>
          <button
            className="ask-send"
            id="ask"
            aria-label={asking ? "正在发送" : "发送问题"}
            title="发送问题（Enter）"
            disabled={busy || !query.trim()}
            onClick={() => ask()}
          >
            <AskIcon name="send" />
          </button>
        </div>
        {status ? (
          <p
            className={`ask-status${statusErr ? " error" : ""}`}
            id="status"
            role={statusErr ? "alert" : "status"}
          >
            {status}
          </p>
        ) : (
          <span id="status" />
        )}
        {!confirmed && answerSource === "document_only" ? (
          <button className="btn ghost" onClick={() => chooseScope("all")}>
            确认使用全部文档
          </button>
        ) : null}
        <div className={`progress${asking ? " on" : ""}`} id="progress">
          <span className="dot" /> <span id="progress-text">{progressText}</span>
        </div>
      </section>
    </div>
  );
});

export default AgentComposer;
