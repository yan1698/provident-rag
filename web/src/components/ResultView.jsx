import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import {
  CITATION_RE,
  STATUS_LABEL,
  citedEvidence,
  looksLikeRefusal,
  srcTone,
  stripMd,
  usageNote,
} from "../lib";
import { sourceHref } from "../sourceLocation";

function renderCited(text, selected, onSelect, hoverSelect) {
  const nodes = [];
  let last = 0;
  let key = 0;
  for (const match of text.matchAll(CITATION_RE)) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const n = match[1];
    const on = String(selected) === String(n) ? "on" : "";
    nodes.push(
      <button
        key={`c-${key++}`}
        type="button"
        className={`cite src-${srcTone(n)} ${on}`}
        data-cite={n}
        title={`查看原文 ${n}`}
        onClick={() => onSelect(n)}
        onMouseEnter={hoverSelect ? () => onSelect(n) : undefined}
      >
        {n}
      </button>
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export function Answer({ answer, selected, onSelect, hoverSelect = true }) {
  const cleaned = stripMd(String(answer || ""));
  const blocks = cleaned
    .split(/\n\n+/)
    .map((block) => block.trim())
    .filter(Boolean);
  if (!blocks.length) return <p className="answer" />;
  return (
    <>
      {blocks.map((block, index) => (
        <p className="answer" key={index}>
          {renderCited(block, selected, onSelect, hoverSelect)}
        </p>
      ))}
    </>
  );
}

function CiteButtons({ sources, onSelect }) {
  return (sources || []).map((n) => (
    <button
      key={n}
      type="button"
      className={`cite src-${srcTone(n)}`}
      data-cite={n}
      onClick={() => onSelect(String(n))}
    >
      {n}
    </button>
  ));
}

export function useEvidenceSelection(initial) {
  const [selected, setSelected] = useState(String(initial));
  const [open, setOpen] = useState(true);
  const pending = useRef(null);

  const select = (index) => {
    const id = String(index);
    pending.current = id;
    setOpen(true);
    setSelected(id);
  };

  useEffect(() => {
    if (!open || !pending.current) return undefined;
    const id = pending.current;
    pending.current = null;
    const card = document.getElementById(`ev-${id}`);
    if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return undefined;
  }, [selected, open]);

  return { selected, open, setOpen, select };
}

export function EvidenceToggle({ open, onToggle }) {
  return (
    <button
      type="button"
      className="evidence-toggle"
      aria-expanded={open}
      aria-label={open ? "收起依据" : "展开依据"}
      onClick={onToggle}
    >
      ›
    </button>
  );
}

export function EvidenceColumn({
  title,
  lede,
  evidence,
  selected,
  onSelect,
  emptyHint,
  open,
  onToggle,
  docNames,
  integrity,
}) {
  if (!open) return null;
  return (
    <aside className="evidence-rail">
      <div className="stack evidence-body">
        <div className="evidence-headbar">
          <h2>{title}</h2>
          <EvidenceToggle open onToggle={onToggle} />
        </div>
        {lede ? <p className="lede">{lede}</p> : null}
        <EvidenceList
          evidence={evidence}
          selected={selected}
          onSelect={onSelect}
          emptyHint={emptyHint}
          docNames={docNames}
          integrity={integrity}
        />
      </div>
    </aside>
  );
}

function pageLabel(page) {
  const value = Number(page);
  if (!Number.isFinite(value) || value <= 0) return "";
  return `第 ${value} 页`;
}

const SOURCE_STATUS = {
  missing: "来源已删除、不存在或无权访问，当前无法核验。",
  hash_mismatch: "来源指纹已变化，与保存时的引用不一致。",
  text_mismatch: "来源文本已变化，与保存时的引用不一致。",
  unverifiable: "历史来源信息不完整，当前无法核验。",
  unavailable: "当前来源核验暂不可用，请稍后重新打开。",
};

export function EvidenceList({ evidence, selected, onSelect, emptyHint, docNames, integrity }) {
  const [open, setOpen] = useState(() => new Set());
  if (!evidence || !evidence.length) {
    return (
      <p className="lede">
        {emptyHint || "没有从知识库里取出片段。换一道示例问题，或先去知识库上传文档。"}
      </p>
    );
  }
  return evidence.map((item) => {
    const active = String(selected) === String(item.index) ? "active" : "";
    const text = item.text || "";
    const short = text.length > 280;
    const expanded = open.has(item.index);
    const title = docNames && item.doc_id ? docNames[item.doc_id] : "";
    const where = pageLabel(item.page);
    const check = integrity?.find(
      (row) => String(row.index) === String(item.index) && row.chunk_id === item.chunk_id
    );
    const sourceStatus = check?.status || (Array.isArray(integrity) ? "unverifiable" : "");
    const sourceError = sourceStatus && sourceStatus !== "ok";
    return (
      <article
        key={item.index}
        className={`evidence src-${srcTone(item.index)} ${active}`}
        id={`ev-${item.index}`}
        data-index={item.index}
        onClick={(event) => {
          if (event.target.closest("[data-expand], a")) return;
          onSelect(String(item.index));
        }}
      >
        <div className="evidence-head">
          <span className="src-label">
            <span className={`src-badge src-${srcTone(item.index)}`}>{item.index}</span>
            {title ? <span className="evidence-doc">{title}</span> : null}
            {where ? <span>{where}</span> : null}
          </span>
          <span className="row">
            {item.doc_id && !sourceError ? (
              <Link className="linkish" to={sourceHref(item)}>
                查阅文档
              </Link>
            ) : null}
            {short ? (
              <button
                type="button"
                className="linkish"
                data-expand="1"
                onClick={(event) => {
                  event.stopPropagation();
                  setOpen((prev) => {
                    const next = new Set(prev);
                    if (next.has(item.index)) next.delete(item.index);
                    else next.add(item.index);
                    return next;
                  });
                }}
              >
                {expanded ? "收起" : "展开全文"}
              </button>
            ) : null}
          </span>
        </div>
        {sourceError ? (
          <p className="error" role="status" data-source-status={sourceStatus}>
            {SOURCE_STATUS[sourceStatus] || "当前来源无法核验。"}
          </p>
        ) : null}
        {!expanded ? (
          <p data-preview>
            {text.slice(0, 280)}
            {short ? "…" : ""}
          </p>
        ) : (
          <p data-full>{text}</p>
        )}
      </article>
    );
  });
}

function statusPill(status) {
  const label = STATUS_LABEL[status] || status;
  return <span className={`pill ${status}`}>{label}</span>;
}

const PROBLEM_STATUS = new Set([
  "unsupported",
  "unverified",
  "uncited",
  "invalid_citation",
  "insufficient",
]);

function claimNeedsReview(claim) {
  return Boolean(claim && (claim.hallucination || PROBLEM_STATUS.has(claim.status)));
}

function ClaimList({ verification }) {
  const [open, setOpen] = useState(false);
  if (!verification) return <p className="lede">这次没有做依据核对。</p>;
  const claims = verification.claims || [];
  if (!claims.length) return <p className="lede">已返回核对结果，但没有可展示的句子。</p>;
  const supported = claims.filter((claim) => claim.status === "supported").length;
  const review = claims.filter(claimNeedsReview).length;
  const total = verification.claim_count ?? claims.length;
  return (
    <div className="verify-fold">
      <button
        type="button"
        className="verify-summary"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span>
          核对了 {total} 句，有依据 {supported} 句{review ? `，需查看 ${review} 句` : ""}
        </span>
        <span className="tiny">{open ? "收起" : "展开明细"}</span>
      </button>
      {review && !open ? <p className="tiny verify-note">有问题的句子在明细里都会列出。</p> : null}
      {open ? (
        <div className="stack verify-claims">
          {claims.map((claim, i) => (
            <div className={`claim${claimNeedsReview(claim) ? " problem" : ""}`} key={i}>
              {statusPill(claim.status)}
              {claim.hallucination ? <span className="pill bad">需警惕</span> : null}
              <p>{claim.text}</p>
              {(claim.source_gaps || []).map((gap, j) => (
                <p className="tiny" key={j}>
                  {gap}
                </p>
              ))}
              {claim.status === "unverified" ? (
                <p className="tiny">缺少可核验的原文支持，未判定事实错误。</p>
              ) : null}
              <div className="tiny">
                对应{" "}
                {(claim.citation_indices || []).length
                  ? (claim.citation_indices || []).map((n) => (
                      <span key={n} className={`cite src-${srcTone(n)}`}>
                        {n}
                      </span>
                    ))
                  : "无编号"}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ExploreAction({ query, onExplore }) {
  const text = String(query || "").trim();
  if (!onExplore || !text) return null;
  return (
    <div className="qa-actions">
      <button type="button" className="btn ghost" onClick={() => onExplore(text)}>
        在 Agent 中继续探索
      </button>
      <span className="tiny">打开新的 Agent 对话并填入这个问题，不会自动开始。</span>
    </div>
  );
}

function scopeSummary(scope) {
  if (!scope || !scope.mode) return "";
  if (scope.widened || scope.mode === "all") return "实际检索范围：全部文档";
  const labels = (scope.labels || []).filter(Boolean).join("、");
  const name = labels || "所选库";
  if (scope.mode === "hard") return `实际检索范围：${name}（不含待归类）`;
  if (scope.mode === "soft") return `实际检索范围：${name}（含待归类）`;
  return "";
}

export default function ResultView({
  data,
  extra,
  docNames,
  mode = "live",
  onEditQuery,
  onEditScope,
  onExplore,
  hideEvidence = false,
  evidenceSelection,
  compact = false,
}) {
  const used = citedEvidence(data);
  const firstCite =
    (data.citations && data.citations[0] && data.citations[0].index) ||
    (used[0] && used[0].index) ||
    1;
  const localSelection = useEvidenceSelection(firstCite);
  const { selected, open, setOpen, select } = evidenceSelection || localSelection;
  const evidence = data.evidence || [];
  const noHits = evidence.length === 0;
  const widened = Boolean(data.scope && data.scope.widened);
  const scopeLine = scopeSummary(data.scope);
  const refused = looksLikeRefusal(data.answer);
  const integrity = data.citation_integrity;
  const savedIntegrity = data.diagnostics?.citation_integrity;

  if (data.answer_source === "general_explanation")
    return (
      <div className="qa-result">
        <p className="callout" role="status">
          模型通用知识解释 · 未检索文档，未经文档核验
        </p>
        <Answer answer={data.answer} />
        {data.source_trace_id ? (
          <Link to={`/agent?trace=${data.source_trace_id}`}>返回原文回答</Link>
        ) : null}
      </div>
    );

  if (noHits) {
    return (
      <div className="qa-result">
        {widened ? (
          <div className="callout" role="status">
            检索范围已扩大：指定知识库中未找到匹配证据，本次已扩大到全部文档。答案可能引用指定库以外的内容。
          </div>
        ) : null}
        <div className="ask-empty" role="status">
          <p>当前检索范围内没有找到相关证据。</p>
          <p className="tiny">这只说明这次检索没有命中，不代表知识库里完全没有相关内容。</p>
          {scopeLine ? <p className="tiny">{scopeLine}</p> : null}
          {!compact && data.query ? <p className="question-echo">{data.query}</p> : null}
          {!compact ? (
            <p className="tiny">{usageNote(data.usage, { recorded: data.usage != null })}</p>
          ) : null}
          {mode === "replay" && !onEditScope ? (
            <Link className="linkish" to={`/agent?q=${encodeURIComponent(data.query || "")}`}>
              回到提问页修改
            </Link>
          ) : (
            <div className="row">
              <button type="button" className="btn ghost" onClick={onEditQuery}>
                修改问题
              </button>
              <button type="button" className="btn ghost" onClick={onEditScope}>
                重新选择范围
              </button>
            </div>
          )}
          <ExploreAction query={data.query} onExplore={onExplore} />
        </div>
        {extra}
        <details className="verify-block" open={!compact}>
          <summary>回答核验</summary>
          <h2>每句话核过没有</h2>
          <ClaimList verification={data.verification} />
        </details>
      </div>
    );
  }

  const emptyHint = evidence.length
    ? "答案没有标注引用编号，因此不展示原文片段。"
    : "没有从知识库里取出片段。换一道示例问题，或先去知识库上传文档。";

  return (
    <div className="qa-result">
      {mode === "replay" &&
      !(compact && Array.isArray(integrity) && integrity.every((item) => item.status === "ok")) ? (
        <div className={compact ? "tiny" : "callout"} role="status" data-source-check="current">
          {Array.isArray(integrity)
            ? integrity.some((item) => item.status === "unavailable")
              ? "当前来源核验暂不可用；下方保留历史回答和证据文本。"
              : "已重新检查当前来源；下方保留历史回答和证据文本。"
            : "尚未取得当前来源状态，保存时的核验结果不代表来源仍然有效。"}
          {Array.isArray(savedIntegrity) ? (
            <span>
              {" "}
              生成时的完整性快照已保留（
              {savedIntegrity.filter((item) => item.status === "ok").length}/{savedIntegrity.length}{" "}
              条通过），与本次检查分开记录。
            </span>
          ) : null}
        </div>
      ) : null}
      {widened ? (
        <div className="callout" role="status">
          检索范围已扩大：指定知识库中未找到匹配证据，本次已扩大到全部文档。答案可能引用指定库以外的内容。
        </div>
      ) : null}
      {refused ? <div className="callout">已检索到片段，但这些片段不足以回答这个问题。</div> : null}
      <div
        className={`${hideEvidence ? "answer-only" : "split"}${open ? "" : " evidence-collapsed"}`}
      >
        <div className="stack answer-pane">
          {open || hideEvidence ? null : (
            <div className="evidence-headbar">
              <EvidenceToggle open={false} onToggle={() => setOpen(true)} />
            </div>
          )}
          {!compact && data.query ? <p className="question-echo">{data.query}</p> : null}
          {scopeLine ? <p className="tiny">{scopeLine}</p> : null}
          {!compact ? (
            <p className="tiny">{usageNote(data.usage, { recorded: data.usage != null })}</p>
          ) : null}
          <Answer
            answer={data.answer}
            selected={selected}
            onSelect={select}
            hoverSelect={!compact}
          />
          {extra}
          <ExploreAction query={data.query} onExplore={onExplore} />
        </div>
        {hideEvidence ? null : (
          <EvidenceColumn
            title="原文依据"
            lede="仅显示答案实际引用的片段；编号与答案中的编号一致。"
            evidence={used}
            selected={selected}
            onSelect={select}
            emptyHint={emptyHint}
            open={open}
            onToggle={() => setOpen((value) => !value)}
            docNames={docNames}
            integrity={integrity}
          />
        )}
      </div>
      <details className="verify-block" open={!compact}>
        <summary>
          回答核验
          {data.verification?.claims?.some(
            (c) => c.status === "unverified" || c.status === "insufficient"
          )
            ? " · 有未核实或支持不足的内容"
            : ""}
        </summary>
        <h2>每句话核过没有</h2>
        <ClaimList verification={data.verification} />
      </details>
    </div>
  );
}

export { CiteButtons };
