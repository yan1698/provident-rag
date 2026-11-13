import { useEffect, useState } from "react";
import { AGENT_STEP_LABEL, citedEvidence, stepDetail, stepUsageLine, stripMd } from "../lib";
import { Answer, CiteButtons } from "./ResultView";
import ConceptChips from "./ConceptChips";
import { api } from "../api";

export function Timeline({ steps, open = false }) {
  if (!steps || !steps.length) return null;
  return (
    <details className="trace-log" open={open}>
      <summary>查看推理过程 · {steps.length} 步</summary>
      <div className="stack">
        {steps.map((step, index) => (
          <div className="step" key={index}>
            <div className="tiny">
              第 {index + 1} 步 · {AGENT_STEP_LABEL[step.action] || step.action}
              {step.source === "guard" ? " · 护栏" : ""}
              {step.elapsed_ms ? ` · ${step.elapsed_ms}ms` : ""}
              {stepUsageLine(step) ? ` · ${stepUsageLine(step)}` : ""}
            </div>
            <p>{stepDetail(step)}</p>
          </div>
        ))}
      </div>
    </details>
  );
}

function Modules({ outline, onSelect }) {
  const modules = (outline && outline.modules) || [];
  const note = outline && outline.order_note;
  if (!modules.length) return <p className="lede">没有可交付的正文。</p>;
  return (
    <>
      {note ? <p className="order-note">{stripMd(note)}</p> : null}
      {modules.map((mod, mi) => {
        const items = (mod.paragraphs || mod.bullets || [])
          .map((b, bi) => {
            const text = stripMd(b.text || "");
            if (!text) return null;
            return (
              <p key={`${mi}-${bi}`}>
                {text} <CiteButtons sources={b.sources} onSelect={onSelect} />
              </p>
            );
          })
          .filter(Boolean);
        return items.length ? (
          <div className="module-block" key={mi}>
            {items}
          </div>
        ) : null;
      })}
    </>
  );
}

export function Gaps({ gaps }) {
  // Old drafts also listed unasked, hypothetical neighbouring chapters as gaps.
  // Hide only that explicit non-task category; retain reading and evidence failures.
  const relevant = (gaps || []).filter(
    (gap) =>
      !(
        /相邻|邻近|其他.*章节/.test(`${gap.topic} ${gap.reason}`) &&
        /未被提问|未被问及|不属于必答|不纳入正文/.test(gap.reason || "")
      )
  );
  if (!relevant.length) return <p className="lede">这次没有单独列出的未覆盖项。</p>;
  return relevant.map((gap, i) => (
    <div className="gap-card" key={`${gap.topic || "gap"}-${i}`}>
      <strong>{stripMd(gap.topic)}</strong>
      <span className="tiny">{stripMd(gap.reason || "")}</span>
    </div>
  ));
}

function anchorLine(outline) {
  const reference = outline && outline.reference;
  const anchor = (outline && outline.answer_anchor) || {};
  const anchorId = anchor.run_id || (outline && outline.anchor_run_id) || "";
  const index = reference && reference.index;
  const indices = reference?.indices || [];
  if (!anchorId || (!index && !indices.length)) return "";
  if (indices.length) return `基于上一回答第 ${indices.join("、")} 步分别读取原文后回答`;
  const verb =
    reference.action === "expand"
      ? "展开"
      : reference.action === "explain"
        ? "解释"
        : reference.action === "recite"
          ? "复述"
          : "";
  const unit = reference.kind === "step" ? "步" : "段";
  return verb ? `基于上一回答第 ${index} ${unit}${verb}` : `基于上一回答第 ${index} ${unit}`;
}

function ParagraphChoices({ outline, disabled, onPick }) {
  const rows = (outline && outline.pending_paragraphs) || [];
  if (!rows.length || !onPick) return null;
  const action = outline.pending_action;
  const kind = outline.pending_reference_kind || "paragraph";
  const unit = kind === "step" ? "步" : "段";
  const title =
    action === "expand"
      ? "你想展开哪一段？"
      : action === "explain"
        ? "你想解释哪一段？"
        : "你想看哪一段？";
  return (
    <div className="para-choices">
      <p className="tiny">{title}</p>
      <div className="chips">
        {rows.map((item) => (
          <button
            key={item.index}
            type="button"
            className="chip"
            disabled={disabled}
            onClick={() => onPick(`第${item.index}${unit}`)}
          >
            第 {item.index} {unit}
            {item.head ? ` · ${item.head}` : ""}
          </button>
        ))}
      </div>
    </div>
  );
}

export function ReadingCoverage({ data }) {
  const outline = data?.outline || {};
  const longform = data?.task_kind === "synthesis" || data?.task_kind === "digest";
  const extractive = (data?.compose_mode || outline.compose_mode || "") === "extractive";
  const delivery = outline.reading?.delivery;
  const gapLabels = {
    fact_binding_pending: "尚未逐项绑定事实，不代表来源失效",
    source_unverifiable: "必要原文未绑定或已失效",
    not_read: "必要原文未读取",
    not_fed: "已读原文未进入写稿",
    input_unconfirmed: "真实模型输入尚未确认",
    generation_unavailable: "没有可核验的模型最终正文",
    not_generated: "模型未生成该要点",
    removed_by_cleaning: "该要点被结构清洗移除",
    citation_unsupported: "引用未支持该结论",
    delivery_unassessed: "尚未逐项核验用户可见结论",
  };
  return (
    <>
      {longform && outline.reading ? (
        <div className="reading-delivery" role="status">
          <p>
            {delivery?.status === "complete"
              ? "已完整交付本次任务：必要要点已逐项核验。"
              : delivery?.status === "unverifiable" || extractive
                ? "无法核验为完整回答：当前内容仅供对照原文。"
                : "答案已生成；逐项事实完整性尚未确认，请分别核对原文、输入和引用支持。"}
          </p>
          {delivery?.source_coverage ? (
            <p className="tiny">
              有效原文 {delivery.source_coverage.valid_chunk_ids?.length || 0} 个，已读取{" "}
              {delivery.source_coverage.read_chunk_ids?.length || 0} 个，实际模型输入{" "}
              {delivery.source_coverage.model_input_chunk_ids?.length || 0} 个，正文引用{" "}
              {delivery.source_coverage.answer_cited_chunk_ids?.length || 0}{" "}
              个；这些数量不等于逐项事实已交付。
            </p>
          ) : null}
          {(delivery?.requirements || []).some((r) => r.stage !== "delivered") ? (
            <details>
              <summary>查看逐项核验状态</summary>
              {delivery.requirements
                .filter((r) => r.stage !== "delivered")
                .map((r) => (
                  <p key={r.id}>
                    {r.requirement}：
                    {gapLabels[
                      r.stage === "source_unverifiable" &&
                      !r.sources?.length &&
                      !delivery.requirements_exhaustive
                        ? "fact_binding_pending"
                        : r.stage
                    ] || r.stage}
                  </p>
                ))}
            </details>
          ) : !delivery?.requirements_exhaustive ? (
            <p className="tiny">尚未建立穷尽的必要事实清单，无法确认步骤、条件及风险均已交付。</p>
          ) : null}
        </div>
      ) : null}
      {Object.values(outline.reading?.documents || {}).map((doc) => (
        <details key={doc.doc_id} className="reading-coverage">
          <summary>
            任务证据 · {doc.filename} ·{" "}
            {doc.complete ? "原文范围已载入，交付待核验" : "仍有阅读或写稿缺口"}
          </summary>
          {Object.entries(doc.goals || {}).map(([goal, value]) => (
            <p key={goal}>
              {{ steps: "操作步骤", conditions: "前置条件", risks: "风险提示" }[goal] || goal}
              ：已读取 {value.read_chunk_ids?.length || 0} 个来源，写稿输入{" "}
              {value.sent_chunk_ids?.length || 0}，正文引用{" "}
              {value.answer_cited_chunk_ids?.length || 0}。引用数量不代表重要信息已完整交付。
            </p>
          ))}
          <p>
            扩展章节范围 {doc.required_chunk_ids?.length || 0} 个片段，已载入{" "}
            {doc.read_chunk_ids?.length || 0}，写稿上下文 {doc.prompt_chunk_ids?.length || 0}，未读{" "}
            {doc.unread_chunk_ids?.length || 0}，预算裁剪 {doc.budget_pruned_chunk_ids?.length || 0}
            。未读片段数不等于用户要点遗漏数。
          </p>
          <p className="tiny">
            {doc.full_document_read ? "全文片段均已载入。" : "本次没有读取全文。"}
            {doc.complete ? "" : "无法确认完整覆盖操作步骤、前置条件和风险。"}
          </p>
          {(doc.missing_categories || []).length ? (
            <p>未识别范围：{doc.missing_categories.join("、")}</p>
          ) : null}
        </details>
      ))}
      {outline.evidence_selection?.delivery?.lost_paragraphs > 0 ? (
        <p role="status">
          有 {outline.evidence_selection.delivery.lost_paragraphs}{" "}
          段生成内容未通过校验，已省略；请查看证据缺口。
        </p>
      ) : null}
      {outline.evidence_selection?.compose?.finish_reason === "length" ? (
        <p role="status">模型输出达到上限，答案可能不完整；请核对步骤、条件和风险缺口。</p>
      ) : null}
      {outline.reading?.mode === "bounded" ? (
        <p className="tiny">
          本次读取 {outline.reading.batches?.length || 0}/{outline.reading.budget?.max_batches || 0}{" "}
          批，状态：{outline.reading.status}。阅读覆盖、答案完整性与引用支持分别核验。
        </p>
      ) : null}
    </>
  );
}

export default function AgentResult({
  data,
  onPickConcept,
  onSelectCite,
  onPickParagraph,
  pickDisabled = false,
  selectedCite = "",
  showReadingDetails = true,
  compact = false,
}) {
  const outline = data.outline || {};
  const used = citedEvidence(data);
  const select = (index) => onSelectCite && onSelectCite(String(index));
  const [concepts, setConcepts] = useState([]);
  const noted = anchorLine(outline);

  useEffect(() => {
    const ids = (used || []).map((item) => item.chunk_id).filter(Boolean);
    if (!ids.length) {
      setConcepts([]);
      return undefined;
    }
    let cancelled = false;
    api(`/graph/related?chunk_ids=${encodeURIComponent(ids.join(","))}`)
      .then((payload) => {
        if (!cancelled) setConcepts(payload.concepts || []);
      })
      .catch(() => {
        if (!cancelled) setConcepts([]);
      });
    return () => {
      cancelled = true;
    };
  }, [data]);

  const unverified = (data.verification?.claims || []).filter((c) => c.status === "unverified");
  const hasModules = (outline.modules || []).length;
  const fallback = data.fallback_reason || outline.fallback_reason || "";
  const extractive = (data.compose_mode || outline.compose_mode || "") === "extractive";
  const scope = data.scope;
  const longform = data.task_kind === "synthesis" || data.task_kind === "digest";

  if (data.answer_source === "general_explanation")
    return (
      <div className="stack">
        <p className="callout" role="status">
          模型通用知识解释 · 未检索文档，未经文档核验
        </p>
        <Answer answer={data.answer} />
      </div>
    );

  return (
    <>
      {noted ? <p className="tiny anchor-line">{noted}</p> : null}
      {unverified.length ? (
        <details className="unverified-claims" open={!compact}>
          <summary>有 {unverified.length} 条陈述未核实，不能据此视为完整可信交付</summary>
          <p className="tiny">缺少原文支持，未判定事实错误；回答内容保留供核对。</p>
          {unverified.map((claim, i) => (
            <div key={i}>
              <p>{claim.text}</p>
              {(claim.source_gaps || []).map((gap, j) => (
                <p className="tiny" key={j}>
                  {gap}
                </p>
              ))}
            </div>
          ))}
        </details>
      ) : null}
      {scope && scope.mode && scope.mode !== "all" ? (
        <p className="tiny scope-line">
          本次在{(scope.labels || []).join("、") || "所选库"}检索
          {scope.mode === "hard" ? "（只查此库）" : "（含待归类）"}
          {scope.widened ? "。该范围没有，已在全部文档中找到。" : "。"}
        </p>
      ) : null}
      {longform && hasModules ? (
        <Modules outline={outline} onSelect={select} />
      ) : (
        <Answer
          answer={data.answer}
          selected={selectedCite}
          onSelect={select}
          hoverSelect={!compact}
        />
      )}
      {showReadingDetails && (outline.reading || outline.evidence_selection) ? (
        <details className="reading-details">
          <summary>阅读覆盖与交付核验</summary>
          <ReadingCoverage data={data} />
        </details>
      ) : null}
      {!compact ? (
        <ConceptChips concepts={concepts} onPick={(name) => onPickConcept && onPickConcept(name)} />
      ) : null}
      {longform && (extractive || fallback) ? (
        <p className="tiny">{fallback || "当前为抽取式证据摘要，未调用生成模型组织长回答。"}</p>
      ) : null}
      {!longform && data.fallback_reason ? <p className="tiny">{data.fallback_reason}</p> : null}
      <ParagraphChoices outline={outline} disabled={pickDisabled} onPick={onPickParagraph} />
    </>
  );
}
