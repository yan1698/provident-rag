"""Compose a readable long answer from retrieved evidence.

Display headings follow the user task. Source sections must stay in the catalog.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

import markdown
from bs4 import BeautifulSoup

from app.agent.state import AgentState, GapItem, SynthesisNode
from app.generate.fence import EVIDENCE_RULE, format_evidence_block
from app.generate.prompt import EvidenceItem, EvidencePrompt


class CitationContractError(ValueError):
    """A draft cited material outside the exact model input contract."""


class DraftStructureError(ValueError):
    """A provider response exists but cannot be safely delivered as a draft."""


def validate_draft_schema(data: dict) -> None:
    """Validate the existing writer shape without coercing source identifiers."""
    if not isinstance(data.get("modules"), list) or not isinstance(data.get("gaps", []), list):
        raise DraftStructureError("modules和gaps必须为数组")
    if data.get("mode", "explain") not in {"explain", "compare", "plan", "inventory"}:
        raise DraftStructureError("mode无效")
    for module in data["modules"]:
        if not isinstance(module, dict):
            raise DraftStructureError("模块必须为对象")
        paragraphs = module.get("paragraphs", module.get("bullets"))
        if not isinstance(paragraphs, list):
            raise DraftStructureError("paragraphs必须为数组")
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict) or not isinstance(paragraph.get("text"), str):
                raise DraftStructureError("正文必须为带text的对象")
            if not isinstance(paragraph.get("sources"), list):
                raise CitationContractError("正文缺少证据编号数组，草稿已拒绝。")
            if any(not isinstance(n, int) or isinstance(n, bool) for n in paragraph["sources"]):
                raise CitationContractError("证据编号必须为整数，草稿已拒绝。")
        sections = module.get("source_sections", [])
        if not isinstance(sections, (list, str, dict)):
            raise DraftStructureError("source_sections结构无效")
    if any(
        not isinstance(gap, dict) or not isinstance(gap.get("topic"), str) or not isinstance(gap.get("reason"), str)
        for gap in data.get("gaps", [])
    ):
        raise DraftStructureError("缺口必须为带topic和reason的对象")


def check_source_contract(value: Any, allowed: set[int]) -> None:
    if isinstance(value, dict):
        if "sources" in value:
            if value["sources"] is not None and not isinstance(value["sources"], list):
                raise CitationContractError("回答包含无效的证据编号数组，草稿已拒绝。")
            for number in value.get("sources") or []:
                if isinstance(number, bool) or not str(number).isdigit() or int(number) not in allowed:
                    raise CitationContractError("回答包含未投喂或无效的证据编号，草稿已拒绝。")
        for child in value.values():
            check_source_contract(child, allowed)
    elif isinstance(value, list):
        for child in value:
            check_source_contract(child, allowed)
    elif isinstance(value, str):
        if any(int(n) not in allowed for n in re.findall(r"\[S(\d+)", normalize_cite_markup(value))):
            raise CitationContractError("回答包含未投喂或无效的证据编号，草稿已拒绝。")


DEFAULT_ORDER_NOTE = ""
COMPARE_EXTRACTIVE_NOTE = "当前仅提供两侧原文摘录，尚未完成比较回答。"

SYNTHESIS_SYSTEM = (
    EVIDENCE_RULE + "你是知识库综合任务写作者。只覆盖用户目标和列出的子问题，写够即止，不要为了完整而写长。"
    "paragraphs.text 必须是连贯正文，禁止 Markdown（不要 # 标题、表格、列表、加粗、代码围栏）。"
    "paragraphs.text 不要内嵌 [S n]、[S5,6] 或文末脚注数字；编号只放 sources 数组。"
    "可以按用户意图组织 heading（普通短句即可），不要照抄目录当唯一结构。"
    "每个事实段落必须带 sources（证据编号 n）。禁止用模型常识补全知识库没有的内容。"
    "命中文本如果和当前子问题明显无关，不要写进正文，放到 gaps。"
    "gaps 只描述用户要求中实际未覆盖的内容，不列未提问的相邻章节，也不得假设存在未提供的章节。"
    "禁止把无关文档的方法论类比为用户要的主题"
    "（例如证据是前端设计规范，任务却是 Agent 学习：不要写「设计思维可启发 Agent」）。"
    "若全部证据都与任务不符，modules 为空，只用 gaps 说明知识库没有对应内容。"
    "intent=compare 时必须按维度对照两侧证据写差异与相同点，不要只摘录各自前言。"
    "source_sections 必须从提供的来源章节里选，不要发明文档或章节名。"
    "只输出一个 JSON 对象，不要 Markdown 围栏。"
    "操作方案必须逐步组织；每个操作段落增加 step_number 正整数并明确写第N步，附实际来源编号。"
    '形状：{"mode":"explain|compare|plan|inventory","compose_mode":"written",'
    '"modules":[{"heading":"<面向用户的短句>","purpose":"<这段回答哪个子问题>",'
    '"source_sections":[{"filename":"...","section":"..."}],'
    '"paragraphs":[{"text":"……","sources":[1,2]}]}],'
    '"gaps":[{"topic":"<未覆盖>","reason":"知识库无对应章节或检索未命中"}]}。'
)

_SENTENCE_RE = re.compile(r"(?<=[。！？；;!?])")
_CITE_RE = re.compile(r"\[S(\d+)\]")
_CITE_COMMA_RE = re.compile(r"\[S(\d+(?:\s*,\s*\d+)+)\]")
_CITE_TOKEN_RE = re.compile(r"@@CITE(\d+)@@")
_FOOTNOTE_TAIL_RE = re.compile(r"([。！？；;!?])\s*\d{1,4}(?:\s+\d{1,4})*(?=\s|$)")
_FOOTNOTE_END_RE = re.compile(r"\s+\d{1,4}(?:\s+\d{1,4})*$")


@dataclass
class SynthesisDraft:
    mode: str = "explain"
    modules: list[dict[str, Any]] = field(default_factory=list)
    order_note: str = DEFAULT_ORDER_NOTE
    gaps: list[dict[str, str]] = field(default_factory=list)
    markdown: str = ""
    stripped: list[dict[str, str]] = field(default_factory=list)
    compose_mode: str = "extractive"
    fallback_reason: str = ""


def allowed_source_sections(state: AgentState) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in state.nodes:
        for item in node.source_sections:
            key = (str(item.get("doc_id") or node.doc_id or ""), str(item.get("section") or ""))
            if key[1] and key not in seen:
                seen.add(key)
                rows.append(
                    {
                        "doc_id": key[0],
                        "section": key[1],
                        "filename": str(item.get("filename") or ""),
                    }
                )
        if node.section:
            key = (node.doc_id or "", node.section)
            if key not in seen:
                seen.add(key)
                filename = next((doc.filename for doc in state.catalog if doc.doc_id == node.doc_id), "")
                rows.append({"doc_id": node.doc_id or "", "section": node.section, "filename": filename})
    for doc in state.catalog:
        for section in doc.sections:
            key = (doc.doc_id, section)
            if key not in seen:
                seen.add(key)
                rows.append({"doc_id": doc.doc_id, "section": section, "filename": doc.filename})
    return rows


def source_allowed(entry: dict[str, Any], allowed: list[dict[str, str]]) -> bool:
    section = str(entry.get("section") or "").strip()
    if not section:
        return False
    filename = str(entry.get("filename") or "").strip()
    doc_id = str(entry.get("doc_id") or "").strip()
    for item in allowed:
        if section != item.get("section"):
            continue
        if doc_id and item.get("doc_id") and doc_id != item.get("doc_id"):
            continue
        if filename and item.get("filename") and filename != item.get("filename"):
            continue
        return True
    return False


def apply_g3(
    modules: list[dict[str, Any]],
    allowed: list[dict[str, str]],
    *,
    valid_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep modules whose facts cite evidence and whose sources are in catalog."""
    kept: list[dict[str, Any]] = []
    stripped: list[dict[str, str]] = []
    valid = valid_indices or set()
    for module in modules:
        if not isinstance(module, dict):
            continue
        heading = plain_prose(str(module.get("heading") or module.get("title") or "").strip())
        sources_in = [
            item
            for item in (module.get("source_sections") or [])
            if isinstance(item, dict) and source_allowed(item, allowed)
        ]
        paragraphs: list[dict[str, Any]] = []
        for item in module.get("paragraphs") or module.get("bullets") or []:
            try:
                if valid_indices is not None:
                    check_source_contract(item, valid)
            except CitationContractError:
                stripped.append({"topic": heading or "引用", "reason": "段落引用未投喂或无效证据，已拒绝整段"})
                continue
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            cites = _clean_sources(item.get("sources"), valid)
            if not text:
                continue
            text, cites = attach_citations(text, cites, valid=valid)
            text = plain_prose(text)
            if not text:
                continue
            if valid and not cites:
                continue
            paragraph = {"text": text, "sources": cites}
            number = item.get("step_number")
            if isinstance(number, int) and not isinstance(number, bool) and 0 < number < 100:
                paragraph["step_number"] = number
                if not re.match(r"^第\s*\d+\s*步", text):
                    paragraph["text"] = f"第{number}步：{text}"
            paragraphs.append(paragraph)
        if not paragraphs:
            stripped.append({"topic": heading or "(空段落)", "reason": "该段没有可溯源的依据，已从正文剥离"})
            continue
        if allowed and not sources_in:
            sources_in = [dict(item) for item in allowed if heading and item.get("section") == heading][:1]
        if allowed and not sources_in:
            stripped.append({"topic": heading or "(无来源)", "reason": "无法对应到知识库章节，已从正文剥离"})
            continue
        kept.append(
            {
                "heading": heading or "说明",
                "title": heading or "说明",
                "purpose": str(module.get("purpose") or "").strip(),
                "source_sections": sources_in,
                "paragraphs": paragraphs,
                "bullets": paragraphs,
            }
        )
    return kept, stripped


def stub_compose(state: AgentState) -> dict[str, Any]:
    """Extractive but question-shaped draft: one module per retrieved task unit."""
    if state.digests:
        return _stub_compose_from_digests(state)
    by_id = {item.chunk_id: item for item in state.evidence}
    intent = state.task_spec.intent if state.task_spec else "explain"
    modules: list[dict[str, Any]] = []
    for node in state.nodes:
        if node.status != "retrieved":
            continue
        items = _evidence_for_node(node, state, by_id)
        if not items:
            continue
        paragraph = _extractive_paragraph(items, question=node.question or node.title)
        if not paragraph:
            continue
        sources = [item.index for item in items[:3]]
        heading = node.question or node.title or "说明"
        source_sections = list(node.source_sections)
        if not source_sections and node.section:
            source_sections = [{"doc_id": node.doc_id or "", "section": node.section, "filename": ""}]
        modules.append(
            {
                "heading": heading,
                "title": heading,
                "purpose": node.question or heading,
                "source_sections": source_sections,
                "paragraphs": [{"text": paragraph, "sources": sources}],
            }
        )
    gaps = [{"topic": gap.topic, "reason": gap.reason} for gap in state.gaps]
    order_note = ""
    if intent == "compare":
        order_note = COMPARE_EXTRACTIVE_NOTE
        gaps.append(
            {
                "topic": "按维度对比",
                "reason": "抽取式只能摘录两侧原文，对照需生成模型写稿。",
            }
        )
    return {
        "mode": intent,
        "compose_mode": "extractive",
        "modules": modules,
        "order_note": order_note,
        "gaps": gaps,
    }


def _stub_compose_from_digests(state: AgentState) -> dict[str, Any]:
    intent = state.task_spec.intent if state.task_spec else "explain"
    modules: list[dict[str, Any]] = []
    for item in state.digests:
        if str(item.get("mode") or "") in {"off_topic", "skipped", "empty"}:
            continue
        paragraphs = []
        for bullet in item.get("bullets") or []:
            if not isinstance(bullet, dict):
                continue
            text = str(bullet.get("text") or "").strip()
            if not text:
                continue
            sources: list[int] = []
            for number in bullet.get("sources") or []:
                try:
                    sources.append(int(number))
                except (TypeError, ValueError):
                    continue
            text, sources = attach_citations(text, sources)
            paragraphs.append({"text": text, "sources": sources})
        if not paragraphs:
            continue
        heading = str(item.get("filename") or "文档理解")
        modules.append(
            {
                "heading": heading,
                "title": heading,
                "purpose": f"读懂《{heading}》",
                "source_sections": list(item.get("source_sections") or []),
                "paragraphs": paragraphs,
            }
        )
    gaps = [{"topic": gap.topic, "reason": gap.reason} for gap in state.gaps]
    return {
        "mode": intent,
        "compose_mode": "extractive",
        "modules": modules,
        "order_note": "",
        "gaps": gaps,
    }


def generation_chars(state: AgentState, settings: Any | None = None) -> int:
    if state.reading.get("mode") == "bounded":
        return int(state.reading["budget"]["generation_chars"])
    return int(getattr(settings, "qa_max_context_chars", 6000))


def compose_synthesis(state: AgentState, settings: Any | None = None) -> SynthesisDraft:
    state.evidence_selection.pop("generated_output", None)
    state.evidence_selection.pop("model_output_text", None)
    prepared = prepare_synthesis_prompt(state, max_chars=generation_chars(state, settings))
    previous = state.evidence_selection.get("compose", {})
    state.evidence_selection["compose"] = prepared.selection
    if previous:
        state.evidence_selection.setdefault("previous_compose", []).append(previous)
    state.evidence = prepared.evidence
    from app.agent.reading import finalize_reading

    finalize_reading(state)
    allowed = {item.index for item in state.evidence}
    # Digest summaries cannot smuggle omitted sources back into the writer prompt.
    for digest in state.digests:
        digest["bullets"] = [b for b in digest.get("bullets", []) if b.get("sources") and set(b["sources"]) <= allowed]
    raw = stub_compose(state)
    compose_mode = "extractive"
    fallback = ""
    intent = state.task_spec.intent if state.task_spec else str(raw.get("mode") or "explain")
    provider = (
        str(getattr(settings, "answer_llm_provider", "") or getattr(settings, "llm_provider", "stub") or "stub")
        .strip()
        .lower()
    )
    state.evidence_selection["compose"].update(
        provider=provider, client_invoked=False, provider_response_received=False
    )
    if not allowed:
        raw = {"modules": [], "gaps": [{"topic": "阅读证据", "reason": "预算内没有可核验原文，无法完整回答。"}]}
        compose_mode = "refused"
        fallback = "预算内没有可核验原文，无法完整回答。"
    elif provider and provider != "stub":
        from app.llm import LLMError

        try:
            raw = _llm_compose(state, settings)
            state.evidence_selection["compose"]["provider_response_received"] = True
            # Preserve real generated content even if the source contract rejects
            # it. Fallback extracts are a different delivery origin.
            state.evidence_selection["generated_output"] = {
                "origin": "model",
                "modules": raw.get("modules", []),
            }
            check_source_contract(raw, allowed)
            compose_mode = str(raw.get("compose_mode") or "written")
        except CitationContractError as exc:
            state.evidence_selection["compose"]["source_contract_error"] = str(exc)
            raw = {"modules": [], "gaps": [{"topic": "引用校验", "reason": str(exc)}]}
            compose_mode = "refused"
            fallback = str(exc)
        except (ValueError, json.JSONDecodeError, LLMError) as exc:
            structural = isinstance(exc, DraftStructureError)
            fallback = (
                "模型结果结构无法安全恢复，未完成本次任务；原始响应已保留，未自动重新生成。"
                if structural
                else "模型生成未能完成，本次任务未交付；未用原文摘录替代回答。"
            )
            state.evidence_selection["compose"]["structure_error" if structural else "generation_error"] = str(exc)
            raw = {"modules": [], "gaps": [{"topic": "结构交付" if structural else "生成失败", "reason": fallback}]}
            compose_mode = "refused"
    elif provider == "stub":
        fallback = (
            COMPARE_EXTRACTIVE_NOTE if intent == "compare" else "当前为抽取式证据摘要，未调用生成模型组织长回答。"
        )
    draft = finalize_draft(raw, state)
    state.evidence_selection.pop("generated_output", None)
    state.evidence_selection.pop("model_output_text", None)
    if compose_mode != "refused" and not draft.modules:
        compose_mode = "refused"
        fallback = "生成内容未通过来源核验，无法提供可靠回答；请查看证据缺口。"
    draft.compose_mode = compose_mode
    draft.fallback_reason = fallback
    if compose_mode == "refused":
        draft.markdown = fallback or "现有证据不足以核验回答，草稿已拒绝。"
    for record in state.reading.get("documents", {}).values():
        record["generation_provider"] = provider
        record["generation_client_invoked"] = state.evidence_selection["compose"].get("client_invoked", False)
        record["model_prompt_chunk_ids"] = (
            record.get("prompt_chunk_ids", [])
            if state.evidence_selection["compose"].get("provider_response_received") and provider != "stub"
            else []
        )
    from app.agent.batch_reading import record_delivery

    record_delivery(state, draft.modules)
    return draft


def finalize_draft(raw: dict[str, Any], state: AgentState) -> SynthesisDraft:
    if not isinstance(raw.get("modules", []), list):
        raw = {"modules": [], "gaps": [{"topic": "结构校验", "reason": "modules必须为数组，无法安全展示正文。"}]}
    allowed = allowed_source_sections(state)
    valid = {item.index for item in state.evidence}
    modules_in = []
    shape_errors = []
    for module in raw.get("modules") or []:
        if not isinstance(module, dict):
            shape_errors.append({"topic": "结构校验", "reason": "模块必须为对象，已拒绝该模块。"})
            continue
        paragraphs = module.get("paragraphs", module.get("bullets", []))
        if not isinstance(paragraphs, list):
            shape_errors.append(
                {"topic": str(module.get("heading") or "结构校验"), "reason": "paragraphs必须为数组，已拒绝该模块。"}
            )
            continue
        # Some real model responses serialize the supplied filename/section pair
        # as a string. Resolve only exact catalog pairs belonging to cited input;
        # never infer a section from an invented display heading.
        cited = {
            int(n) for p in paragraphs if isinstance(p, dict) for n in (p.get("sources") or []) if str(n).isdigit()
        }
        docs = {item.doc_id for item in state.evidence if item.index in cited}
        sections = []
        entries = module.get("source_sections") or []
        if isinstance(entries, (str, dict)):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, str):
                sections.extend(
                    dict(a) for a in allowed if a["doc_id"] in docs and entry == f"{a['filename']} / {a['section']}"
                )
            else:
                sections.append(entry)
        modules_in.append({**module, "paragraphs": paragraphs, "source_sections": sections})
    modules, stripped = apply_g3(
        modules_in,
        allowed,
        valid_indices=valid,
    )
    stripped.extend(shape_errors)
    original_modules = state.evidence_selection.get("generated_output", {}).get("modules", raw.get("modules", []))
    original_modules = original_modules if isinstance(original_modules, list) else []
    before_paragraphs = sum(
        len(m.get("paragraphs", m.get("bullets", [])))
        for m in original_modules
        if isinstance(m, dict) and isinstance(m.get("paragraphs", m.get("bullets", [])), list)
    )
    after_paragraphs = sum(len(m["paragraphs"]) for m in modules)
    state.evidence_selection["delivery"] = {
        "origin": "model" if state.evidence_selection.get("generated_output") else "extractive",
        "raw_content": original_modules,
        "model_output_text": state.evidence_selection.get("model_output_text", ""),
        "visible_content": modules,
        "raw_modules": len(original_modules),
        "visible_modules": len(modules),
        "raw_paragraphs": before_paragraphs,
        "visible_paragraphs": after_paragraphs,
        "lost_paragraphs": max(0, before_paragraphs - after_paragraphs),
        "stripped": stripped,
        "semantic_coverage": "not_assessed: module preservation is not answer completeness",
    }
    uncovered = _uncovered_units(state, modules)
    raw_gaps = factual_gaps(raw.get("gaps") or [], state)
    if state.reading.get("documents") and any(not d.get("complete") for d in state.reading["documents"].values()):
        raw_gaps = [
            {**g, "reason": "本轮读取或投喂的片段不足；请查看未读及预算裁剪范围，不能据此推断原文或知识库缺失。"}
            if isinstance(g, dict)
            and re.search(
                r"知识库未提供|知识库没有|知识库无|文档未写|原文没有|文档没有|未提供.*内容", str(g.get("reason") or "")
            )
            else g
            for g in raw_gaps
        ]
    gaps = _merge_gaps(raw_gaps, state.gaps, stripped + uncovered)
    intent = str(raw.get("mode") or (state.task_spec.intent if state.task_spec else "explain"))
    draft = SynthesisDraft(
        mode=intent,
        modules=modules,
        order_note=str(raw.get("order_note") or "").strip(),
        gaps=gaps,
        stripped=stripped,
        compose_mode=str(raw.get("compose_mode") or "extractive"),
    )
    draft.markdown = render_synthesis_markdown(draft)
    if before_paragraphs > after_paragraphs:
        state.partial = True
        draft.markdown = "部分生成内容未通过结构或来源校验，已省略；请查看具体缺口。\n\n" + draft.markdown
    if state.partial and state.reading:
        incomplete = any(
            d.get("unread_chunk_ids") or d.get("budget_pruned_chunk_ids") or d.get("prompt_omitted_chunk_ids")
            for d in state.reading.get("documents", {}).values()
        )
        note = (
            "阅读范围未完整覆盖；以下仅为已获得来源的部分内容。"
            if incomplete
            else "已获得本次范围原文；正文完整性与引用支持仍需分别核对。"
        )
        draft.markdown = note + "\n\n" + draft.markdown
    return draft


def render_synthesis_markdown(draft: SynthesisDraft) -> str:
    lines: list[str] = []
    if draft.order_note:
        lines.append(plain_prose(draft.order_note))
        lines.append("")
    for module in draft.modules:
        for item in module.get("paragraphs") or module.get("bullets") or []:
            text, cites = attach_citations(
                plain_prose(str(item.get("text") or "")),
                [int(n) for n in (item.get("sources") or []) if str(n).isdigit() or isinstance(n, int)],
            )
            if not text:
                continue
            marker = format_cite_markers(cites)
            lines.append(f"{text} {marker}".strip() if marker else text)
            lines.append("")
    return "\n".join(lines).strip()


def synthesis_verify_body(modules: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for module in modules:
        for item in module.get("paragraphs") or module.get("bullets") or []:
            text, cites = attach_citations(
                plain_prose(str(item.get("text") or "").strip()),
                [int(n) for n in (item.get("sources") or []) if str(n).isdigit() or isinstance(n, int)],
            )
            if text:
                marker = format_cite_markers(cites)
                parts.append(f"{text} {marker}".strip() if marker else text)
    return "\n".join(parts)


def reindex_evidence_state(state: AgentState) -> None:
    old_to_chunk = {item.index: item.chunk_id for item in state.evidence}
    rebuilt: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in state.evidence:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        rebuilt.append(replace(item, index=len(rebuilt) + 1))
    state.evidence = rebuilt
    chunk_to_index = {item.chunk_id: item.index for item in rebuilt}
    mapping = {old: chunk_to_index[cid] for old, cid in old_to_chunk.items() if cid in chunk_to_index}
    state.evidence_selection["reindex"] = {str(old): new for old, new in mapping.items()}
    for digest in state.digests:
        bullets = []
        for bullet in digest.get("bullets") or []:
            old_sources = bullet.get("sources") or []
            if not old_sources or any(n not in mapping for n in old_sources):
                continue
            body = _CITE_RE.sub("", normalize_cite_markup(str(bullet.get("text") or "")))
            bullets.append({**bullet, "text": body, "sources": [mapping[n] for n in old_sources]})
        digest["bullets"] = bullets
    remapped: dict[str, list[int]] = {}
    for node_id, indexes in state.evidence_by_node.items():
        fresh: list[int] = []
        for old in indexes:
            chunk_id = old_to_chunk.get(old)
            new = chunk_to_index.get(chunk_id or "")
            if new and new not in fresh:
                fresh.append(new)
        remapped[node_id] = fresh
    for node in state.nodes:
        remapped.setdefault(
            node.id,
            [chunk_to_index[cid] for cid in node.chunk_ids if cid in chunk_to_index],
        )
    state.evidence_by_node = remapped
    state.hit_count = len(state.evidence)


def evidence_snapshot(state: AgentState) -> list[dict[str, Any]]:
    return [
        {
            "index": item.index,
            "chunk_id": item.chunk_id,
            "doc_id": item.doc_id,
            "page": item.page,
            "text": item.text,
            "chunk_hash": item.chunk_hash,
            "section_id": item.section_id,
            "section_path": item.section_path,
        }
        for item in state.evidence
    ]


def _evidence_for_node(
    node: SynthesisNode,
    state: AgentState,
    by_id: dict[str, EvidenceItem],
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for chunk_id in node.chunk_ids:
        item = by_id.get(chunk_id)
        if item is not None and item.chunk_id not in seen:
            seen.add(item.chunk_id)
            items.append(item)
    for index in state.evidence_by_node.get(node.id) or []:
        hit = next((item for item in state.evidence if item.index == index), None)
        if hit is not None and hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            items.append(hit)
    return items


def format_cite_markers(indices: list[int]) -> str:
    """Render evidence indexes as consecutive [S n] tokens."""
    seen: list[int] = []
    for number in indices:
        try:
            value = int(number)
        except (TypeError, ValueError):
            continue
        if value and value not in seen:
            seen.append(value)
    return "".join(f"[S{n}]" for n in seen)


def parse_cite_indices(text: str) -> list[int]:
    found: list[int] = []
    for match in _CITE_RE.finditer(normalize_cite_markup(text or "")):
        number = int(match.group(1))
        if number not in found:
            found.append(number)
    return found


def normalize_cite_markup(text: str) -> str:
    """Turn [S5,6] / [S5, 6] into [S5][S6] so Markdown cannot parse them as links."""

    def repl(match: re.Match[str]) -> str:
        numbers = [int(part) for part in re.split(r"\s*,\s*", match.group(1)) if part]
        return format_cite_markers(numbers)

    return _CITE_COMMA_RE.sub(repl, text or "")


def attach_citations(
    text: str,
    sources: list[int] | None = None,
    *,
    valid: set[int] | None = None,
) -> tuple[str, list[int]]:
    """Strip inline markers and footnote dumps from prose; merge indexes into sources."""
    raw = normalize_cite_markup(text or "")
    from_text = parse_cite_indices(raw)
    body = _CITE_RE.sub("", raw)
    body = _FOOTNOTE_TAIL_RE.sub(r"\1", body)
    body = _FOOTNOTE_END_RE.sub("", body)
    body = re.sub(r"\s+", " ", body).strip(" -•*")
    merged: list[int] = []
    for number in list(from_text) + list(sources or []):
        try:
            value = int(number)
        except (TypeError, ValueError):
            continue
        if valid is not None and value not in valid:
            continue
        if value and value not in merged:
            merged.append(value)
    return body, merged


def _md_to_text(text: str) -> str:
    """Markdown → plain text. Code stays; [S<n>] citations are parked first."""
    raw = normalize_cite_markup(text or "")
    if not raw.strip():
        return ""
    parked = _CITE_RE.sub(lambda match: f"@@CITE{match.group(1)}@@", raw)
    html = markdown.markdown(parked, extensions=["fenced_code", "tables", "sane_lists"])
    out = BeautifulSoup(html, "html.parser").get_text("\n")
    out = _CITE_TOKEN_RE.sub(r"[S\1]", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def plain_prose(text: str) -> str:
    """One paragraph: drop Markdown markers, keep code identifiers and [S<n>]."""
    return re.sub(r"\s+", " ", _md_to_text(text)).strip(" -•*")


def plain_answer(text: str) -> str:
    """Full answer: drop Markdown, keep paragraph breaks, fenced code, and [S<n>]."""
    return _md_to_text(text)


def _extractive_paragraph(
    items: list[EvidenceItem],
    *,
    question: str = "",
    max_chars: int = 480,
) -> str:
    from app.agent.decompose import question_text_score, tokenize, coverage

    filler = tokenize("说明 如何 定义 要点 知识库 哪些 部分 怎么 有何 异同")
    q_toks = tokenize(question) - filler
    ranked: list[tuple[float, str]] = []
    for item in items[:4]:
        text = plain_prose(item.text or "")
        pieces = [piece.strip() for piece in _SENTENCE_RE.split(text) if piece.strip()]
        if not pieces and text:
            pieces = [text]
        for piece in pieces:
            if len(piece) < 4:
                continue
            if q_toks:
                score = max(coverage(q_toks, tokenize(piece)), question_text_score(question, piece))
            else:
                score = 1.0
            ranked.append((score, piece))
    ranked.sort(key=lambda pair: (-pair[0], -len(pair[1])))
    sentences: list[str] = []
    seen: set[str] = set()
    for score, piece in ranked:
        if score < 0.28 and sentences:
            continue
        if piece in seen:
            continue
        seen.add(piece)
        sentences.append(piece)
        if len(sentences) >= 3:
            break
    blob = "".join(sentences)
    if not blob:
        return ""
    blob, _ = attach_citations(blob, [])
    if len(blob) > max_chars:
        blob = blob[: max_chars - 1].rstrip() + "…"
    return blob


def _uncovered_units(state: AgentState, modules: list[dict[str, Any]]) -> list[dict[str, str]]:
    from app.verify.meta import unit_is_covered

    missing: list[dict[str, str]] = []
    for node in state.nodes:
        question = node.question or node.title
        if not question:
            continue
        if node.status == "empty":
            missing.append({"topic": question, "reason": "检索未命中"})
            continue
        if not unit_is_covered(question, modules):
            missing.append({"topic": question, "reason": "已有证据未能覆盖该子问题"})
    return missing


def _clean_sources(raw: Any, valid: set[int]) -> list[int]:
    sources: list[int] = []
    if not isinstance(raw, list):
        return sources
    for item in raw:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if valid and number not in valid:
            continue
        if number not in sources:
            sources.append(number)
    return sources


def factual_gaps(gaps, state):
    """Discard model-invented adjacent-section omissions, keep actual runtime gaps."""
    known = {s.get("title", "") for doc in state.catalog for s in (doc.section_entries or [])}
    kept = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        text = str(gap.get("topic", "")) + " " + str(gap.get("reason", ""))
        adjacent = re.search(r"相邻|邻近|前言|总述|其他.*章节|未(?:被)?(?:提问|问及)|不属于必答|不纳入正文", text)
        if adjacent and not any(title and len(title) >= 2 and title in str(gap.get("topic", "")) for title in known):
            continue
        if adjacent and re.search(r"相邻|可能存在|未(?:被)?(?:提问|问及)|不属于必答|不纳入正文", text):
            continue  # Unasked material is not a failure to cover this task.
        kept.append(gap)
    return kept


def _merge_gaps(
    raw: Any,
    existing: list[GapItem],
    stripped: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(topic: str, reason: str) -> None:
        key = topic.strip()
        if not key or key in seen:
            return
        seen.add(key)
        merged.append({"topic": key, "reason": reason.strip() or "知识库无对应章节或检索未命中"})

    for gap in existing:
        add(gap.topic, gap.reason)
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                add(str(item.get("topic") or ""), str(item.get("reason") or ""))
    for item in stripped:
        add(item.get("topic") or "", item.get("reason") or "")
    return merged


def _llm_compose(state: AgentState, settings: Any) -> dict[str, Any]:
    from app.llm import get_llm_client
    from app.llm.json_io import parse_complete_object

    client = get_llm_client(settings)
    prompt = _build_prompt(state, max_chars=generation_chars(state, settings))
    budget = min(2400, max(1200, int(getattr(settings, "llm_max_output_tokens", 1024))))
    # One generation. Malformed JSON is a local delivery issue, never a reason
    # to incur an automatic extra provider request.
    state.evidence_selection.setdefault("compose", {})["client_invoked"] = True
    response = client.generate(
        system=SYNTHESIS_SYSTEM,
        prompt=prompt,
        max_output_tokens=budget,
        temperature=0.0,
        purpose="compose",
    )
    raw = response.text
    state.evidence_selection["compose"]["model_input_confirmed"] = True
    state.evidence_selection["compose"]["provider_response_received"] = True
    state.evidence_selection["model_output_text"] = raw
    finish_reason = getattr(response, "finish_reason", "") or ""
    state.evidence_selection.setdefault("compose", {})["finish_reason"] = finish_reason
    if finish_reason == "length":
        state.partial = True
        if not any(g.topic == "生成预算" for g in state.gaps):
            state.gaps.append(
                GapItem(
                    topic="生成预算",
                    reason="模型输出达到思考与正文合计上限，答案可能不完整；保留已通过校验的部分，不据此宣称完整交付。",
                )
            )
    data, recovery = parse_complete_object(raw)
    state.evidence_selection["compose"]["local_json_recovery"] = recovery
    if data is None:
        raise DraftStructureError("compose未返回可完整恢复的JSON对象")
    state.evidence_selection["generated_output"] = {"origin": "model", "modules": data.get("modules", [])}
    validate_draft_schema(data)
    check_source_contract(data, {item.index for item in state.evidence})
    allowed = allowed_source_sections(state)
    by_index = {item.index: item for item in state.evidence}
    for module in data["modules"]:
        paragraphs = module.get("paragraphs", module.get("bullets", []))
        cited_docs = {by_index[n].doc_id for paragraph in paragraphs for n in paragraph["sources"]}
        entries = module.get("source_sections", [])
        entries = [entries] if isinstance(entries, (str, dict)) else entries
        for entry in entries:
            actual = [s for s in allowed if s["doc_id"] in cited_docs]
            valid = (
                any(entry == f"{s['filename']} / {s['section']}" for s in actual)
                if isinstance(entry, str)
                else isinstance(entry, dict) and source_allowed(entry, actual)
            )
            if not valid:
                raise CitationContractError("未通过来源核验：引用章节与实际投喂来源身份不一致，草稿已拒绝。")
    data.setdefault("compose_mode", "written")
    return data


def prepare_synthesis_prompt(state: AgentState, *, max_chars: int) -> EvidencePrompt:
    from app.agent.batch_reading import writer_order

    selected: list[EvidenceItem] = []
    used = 0
    for item in writer_order(state):
        size = len(format_evidence_block(item.text, index=item.index, page=item.page) + "\n")
        if used + size > max_chars and (selected or state.reading.get("mode") == "bounded"):
            if state.reading.get("mode") == "bounded":
                continue
            break
        selected.append(item)
        used += size
    sent = {item.index for item in selected}
    spec = state.task_spec
    intent = spec.intent if spec else "explain"
    lines = [
        f"用户目标：{state.goal}",
        f"任务类型：{intent}",
        f"期望交付：{spec.deliverable if spec else '紧扣子问题的说明'}",
        "",
        "需要覆盖的子问题：",
    ]
    if intent == "compare":
        lines.append("写稿要求：按维度对照两侧，写出相同点与差异；不要只粘贴各自前言。")
        lines.append("")
    for node in state.nodes:
        status = node.status
        lines.append(f"- [{status}] {node.question or node.title}")
    lines.append("")
    lines.append("允许引用的来源章节（source_sections 只能从这里选）：")
    for item in allowed_source_sections(state)[:40]:
        lines.append(f"- {item.get('filename')} / {item.get('section')}")
    lines.append("")
    if state.reading.get("documents"):
        lines.append(
            "阅读范围：未投喂或预算裁剪不代表知识库缺少内容。gaps只能描述本轮未读、未投喂或证据不足，不得据此宣称原文不存在。"
        )
    if state.reading.get("mode") == "bounded":
        lines.append(
            "交付目标：按用户要求分别核对操作步骤、前置条件、风险提示；逐项写出原文支持的必要信息。读取数量不等于答案完整。未获原文支持的要求明确列入gaps，不得声称全文无遗漏。"
        )
    if state.digests:
        lines.append("已读懂的文档要点（优先依据这些写稿，仍须带 sources）：")
        for digest in state.digests:
            if str(digest.get("mode") or "") in {"off_topic", "skipped"}:
                lines.append(f"《{digest.get('filename')}》与任务不符，不要引用、不要类比。")
                continue
            lines.append(f"《{digest.get('filename')}》 mode={digest.get('mode')}")
            for bullet in digest.get("bullets") or []:
                if isinstance(bullet, dict) and bullet.get("text"):
                    if not bullet.get("sources") or not set(bullet["sources"]) <= sent:
                        continue
                    text, cites = attach_citations(
                        str(bullet.get("text") or ""), _clean_sources(bullet.get("sources"), set())
                    )
                    marker = format_cite_markers(cites)
                    lines.append(f"- {text} {marker}".strip() if marker else f"- {text}")
        lines.append("")
        lines.append("写稿按用户问题抽槽，不要按文档小节从头复述。")
        lines.append("")
    lines.append("证据列表：")
    for item in selected:
        block = format_evidence_block(item.text, index=item.index, page=item.page) + "\n"
        lines.append(block)
    lines.append(
        "请输出 JSON。paragraphs.text 用连贯中文句子，不要 Markdown，不要内嵌 [S n] 或脚注数字。"
        "只覆盖上面列出的子问题；gaps只描述这些问题实际缺少的证据，不列未提问或假设的相邻章节。"
        "不要把无关章节写进正文，不要做跨主题类比。"
    )
    return EvidencePrompt(
        text="\n".join(lines),
        evidence=selected,
        selection={
            "candidate_evidence": [vars(item).copy() for item in state.evidence],
            "sent_indices": [item.index for item in selected],
            "omitted_indices": [item.index for item in state.evidence if item.index not in sent],
            "max_context_chars": max_chars,
            "evidence_chars": used,
            "first_block_exceeds_budget": bool(selected and used > max_chars),
        },
    )


def _build_prompt(state: AgentState, *, max_chars: int) -> str:
    return prepare_synthesis_prompt(state, max_chars=max_chars).text


def _parse_json_object(raw: str) -> dict[str, Any]:
    from app.llm import parse_json_object

    data = parse_json_object(raw)
    if data is None:
        raise ValueError("compose did not return a JSON object")
    return data
