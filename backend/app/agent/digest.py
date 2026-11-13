"""Read uploaded documents with DeepSeek before compose (digest lane)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.state import AgentPhase, AgentState, GapItem, SynthesisNode
from app.generate.fence import EVIDENCE_RULE, format_evidence_block
from app.generate.prompt import EvidenceItem

if TYPE_CHECKING:
    from app.core.config import Settings

_SENTENCE_RE = re.compile(r"(?<=[。！？；;!?])")

DIGEST_SYSTEM = (
    EVIDENCE_RULE + "你阅读知识库里的一篇上传文档，只摘出与用户问题槽位直接相关的要点"
    "（例如如何做、为什么），不要按文档小节从头复述，也不要把未问到的升维结论写成要点。"
    "禁止用常识补全。禁止把本文主题类比到用户要的主题"
    "（例如文档是界面设计规范，任务却是 Agent 学习路径：bullets 必须为空）。"
    "若文档与任务明显无关，bullets 为空，gaps 说明「文档主题与任务不符」，off_topic 为 true。"
    "每个要点必须带 sources（证据编号 n）。"
    "bullets.text 不要写 [S n]、[S5,6] 或文末脚注数字，编号只放 sources。"
    "只输出一个 JSON 对象，不要 Markdown。"
    '形状：{"off_topic":false,"bullets":[{"text":"连贯中文句子","sources":[1]}],'
    '"gaps":[{"topic":"未覆盖点","reason":"文档未写明"}]}。'
)


def pending_digest_docs(state: AgentState) -> list:
    """Unread catalog docs, best match first, clearly off-topic files dropped.

    Once some document produced usable bullets, stop: reading the leftovers only
    tempts the writer into cross-topic analogies.
    """
    from app.corpus.relevance import OFF_TOPIC_FLOOR, score_document

    done = {str(item.get("doc_id") or "") for item in state.digests}
    pending = [doc for doc in state.catalog if doc.doc_id not in done]
    if not pending:
        return []

    def score(doc) -> float:
        return score_document(state.goal, doc.filename, doc.sections, getattr(doc, "preview", "") or "")

    ranked = sorted(pending, key=lambda doc: -score(doc))
    scores = [score(doc) for doc in ranked]
    has_bullets = any(item.get("bullets") for item in state.digests)
    best = scores[0]
    if best < OFF_TOPIC_FLOOR:
        # Nothing matches on the surface: read one document rather than refuse
        # outright, but never keep grinding through the rest of the corpus.
        return [] if state.digests else ranked[:1]
    floor = max(OFF_TOPIC_FLOOR, best * 0.3)
    kept = [doc for doc, value in zip(ranked, scores) if value >= floor]
    return kept if kept or not has_bullets else []


def apply_digest(
    state: AgentState,
    source: Session | Any,
    settings: Settings,
) -> dict[str, Any]:
    """Read the next catalog document. Empty pending → done=true."""
    state.phase = AgentPhase.DIGESTING
    cap = max(1, int(getattr(settings, "agent_digest_max_docs", 8) or 8))
    pending = pending_digest_docs(state)
    if len(state.digests) >= cap:
        for doc in pending:
            state.gaps.append(GapItem(topic=doc.filename, reason="超出读懂文档篇数预算，未阅读"))
            state.digests.append(
                {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "bullets": [],
                    "mode": "skipped",
                    "source_sections": [
                        {"doc_id": doc.doc_id, "section": s, "filename": doc.filename} for s in doc.sections[:3]
                    ],
                }
            )
        return {"done": True, "skipped": len(pending), "digest_count": len(state.digests)}
    if not pending:
        return {"done": True, "digest_count": len(state.digests)}

    doc = pending[0]
    items = _load_doc_evidence(source, state, doc.doc_id, settings)
    if not items:
        state.gaps.append(GapItem(topic=doc.filename, reason="该文档没有可阅读的片段"))
        state.digests.append(
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "bullets": [],
                "mode": "empty",
                "source_sections": [],
            }
        )
        _ensure_node(state, doc, [], "empty")
        return {
            "done": False,
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "bullet_count": 0,
            "mode": "empty",
            "digest_count": len(state.digests),
        }

    bullets, gaps, mode, note = _read_document(state.goal, items, settings, doc)
    from app.agent.reading import mark_read

    if mode != "off_topic":
        mark_read(state, doc.doc_id, items)
    if doc.doc_id in state.reading.get("documents", {}):
        state.reading["documents"][doc.doc_id]["reader_mode"] = mode
    if mode == "off_topic":
        source_sections = [
            {"doc_id": doc.doc_id, "section": section, "filename": doc.filename} for section in doc.sections[:6]
        ]
        state.gaps.append(GapItem(topic=doc.filename, reason="文档主题与当前任务不符，未纳入回答"))
        state.digests.append(
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "bullets": [],
                "mode": "off_topic",
                "source_sections": source_sections,
            }
        )
        _ensure_node(state, doc, items, "empty")
        remaining = pending_digest_docs(state)
        return {
            "done": not remaining,
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "bullet_count": 0,
            "mode": "off_topic",
            "digest_count": len(state.digests),
            "remaining": len(remaining),
        }

    source_sections = [
        {"doc_id": doc.doc_id, "section": section, "filename": doc.filename} for section in doc.sections[:6]
    ]
    state.digests.append(
        {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "bullets": bullets,
            "mode": mode,
            "source_sections": source_sections,
        }
    )
    for gap in gaps:
        topic = str(gap.get("topic") or "").strip()
        if topic:
            state.gaps.append(GapItem(topic=topic, reason=str(gap.get("reason") or "文档未写明")))
    status = "retrieved" if bullets else "empty"
    if not bullets:
        state.gaps.append(GapItem(topic=doc.filename, reason="读懂后没有与任务相关的要点"))
    node = _ensure_node(state, doc, items, status)
    node.source_sections = source_sections
    wanted = {item.chunk_id for item in items}
    state.evidence_by_node[node.id] = [item.index for item in state.evidence if item.chunk_id in wanted]
    remaining = pending_digest_docs(state)
    observation = {
        "done": not remaining,
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "bullet_count": len(bullets),
        "mode": mode,
        "digest_count": len(state.digests),
        "remaining": len(remaining),
    }
    if note:
        observation["note"] = note
    return observation


def _load_doc_evidence(
    source: Session | Any,
    state: AgentState,
    doc_id: str,
    settings: Settings,
) -> list[EvidenceItem]:
    loader = getattr(source, "load_doc_chunks", None)
    if loader is not None:
        rows = loader(doc_id)
        if not rows:
            return [item for item in state.evidence if item.doc_id == str(doc_id)]
    else:
        try:
            uid = UUID(str(doc_id))
        except ValueError:
            return [item for item in state.evidence if item.doc_id == str(doc_id)]
        from app.models.chunk import Chunk

        rows = list(
            source.execute(select(Chunk).where(Chunk.doc_id == uid).order_by(Chunk.start_offset, Chunk.page))
            .scalars()
            .all()
        )
    if not rows:
        return []
    cap = max(800, int(getattr(settings, "agent_digest_max_chars", 6000) or 6000))
    from app.agent.reading import select_reading

    doc = next((d for d in state.catalog if d.doc_id == str(doc_id)), None)
    chosen = (
        select_reading(state, doc, rows, cap, settings) if doc else _select_doc_chunks(state.goal, rows, cap, settings)
    )
    fresh = [
        EvidenceItem(
            index=0,
            chunk_id=str(row.id),
            doc_id=str(row.doc_id),
            page=int(row.page or 1),
            text=(row.text or "").strip(),
            chunk_hash=row.chunk_hash or "",
            start_offset=int(getattr(row, "start_offset", 0)),
            section_path=getattr(row, "section_path", ""),
        )
        for row in chosen
    ]
    if not fresh:
        return []
    seen = {item.chunk_id for item in state.evidence}
    for item in fresh:
        if item.chunk_id in seen:
            existing = next(e for e in state.evidence if e.chunk_id == item.chunk_id)
            existing.text, existing.chunk_hash = item.text, item.chunk_hash
            existing.section_path, existing.start_offset = item.section_path, item.start_offset
            continue
        seen.add(item.chunk_id)
        state.evidence.append(
            EvidenceItem(
                index=len(state.evidence) + 1,
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                page=item.page,
                text=item.text,
                chunk_hash=item.chunk_hash,
                section_path=item.section_path,
                start_offset=item.start_offset,
            )
        )
    state.hit_count = len(state.evidence)
    wanted = {item.chunk_id for item in fresh}
    selected = [item for item in state.evidence if item.chunk_id in wanted]
    state.evidence_selection.setdefault("digest", []).append(
        {
            "doc_id": str(doc_id),
            "sent_indices": [i.index for i in selected],
            "chunk_ids": [i.chunk_id for i in selected],
            "max_context_chars": cap,
        }
    )
    return selected


def _chunk_scores(goal: str, texts: list[str], settings: Settings) -> list[float]:
    from app.corpus.relevance import coverage, lexical_keys, topic_keys

    try:
        from app.rerank.model import rerank_scores

        scored = rerank_scores(goal, texts, settings=settings)
        if len(scored) == len(texts):
            return [float(value) for value in scored]
    except Exception:  # noqa: BLE001 - reranker may be absent offline; fall back to lexical
        pass
    topic = topic_keys(goal)
    return [coverage(topic, lexical_keys(text)) if topic else 0.0 for text in texts]


def _select_doc_chunks(goal: str, rows: list[Any], cap: int, settings: Settings) -> list[Any]:
    """Pick the passages a document should be read through.

    Under budget: read it all. Over budget: rank passages against the task instead
    of taking the first ``cap`` characters, or the second half is never seen.
    """
    texts = [(getattr(row, "text", "") or "").strip() for row in rows]
    filled = [index for index, text in enumerate(texts) if text]
    if sum(len(texts[index]) for index in filled) <= cap:
        return [rows[index] for index in filled]

    scores = _chunk_scores(goal, texts, settings)
    order = sorted(filled, key=lambda index: (-scores[index], index))
    head = filled[0]
    # Keep the opening for context, but never let it crowd out the relevant part.
    if len(texts[head]) <= cap * 0.35:
        order = [head] + [index for index in order if index != head]

    picked: list[int] = []
    used = 0
    for index in order:
        size = len(texts[index])
        if used and used + size > cap:
            continue
        used += size
        picked.append(index)
    picked.sort()
    return [rows[index] for index in picked]


def _read_document(
    goal: str,
    items: list[EvidenceItem],
    settings: Settings,
    doc: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], str, str]:
    """Return (bullets, gaps, mode, note). ``note`` explains any silent downgrade."""
    from app.corpus.relevance import OFF_TOPIC_FLOOR, score_document
    from app.llm import LLMError, router_llm_available

    filename = getattr(doc, "filename", "") if doc is not None else ""
    sections = list(getattr(doc, "sections", None) or []) if doc is not None else []
    preview = getattr(doc, "preview", "") if doc is not None else ""
    if filename and score_document(goal, filename, sections, preview) < OFF_TOPIC_FLOOR:
        return [], [{"topic": filename, "reason": "文档主题与当前任务不符"}], "off_topic", ""

    note = ""
    if not router_llm_available(settings):
        note = "未配置任务模型密钥，本篇改为抽取式摘句"
    else:
        try:
            data = _llm_digest(goal, items, settings, filename)
            from app.generate.synthesis import check_source_contract

            check_source_contract(data.get("bullets", []), {item.index for item in items})
            if data.get("off_topic") is True:
                return [], [{"topic": filename or "文档", "reason": "文档主题与当前任务不符"}], "off_topic", ""
            bullets = _clean_bullets(data.get("bullets"), {item.index for item in items})
            gaps = [
                item
                for item in (data.get("gaps") or [])
                if isinstance(item, dict) and str(item.get("topic") or "").strip()
            ]
            if bullets:
                return bullets, gaps, "model", ""
            if gaps:
                return [], gaps, "model", ""
            note = "模型未给出可溯源要点，本篇改为抽取式摘句"
        except (LLMError, ValueError, TypeError) as exc:
            note = f"阅读文档失败，已回退抽取式：{exc}"
    return _extractive_bullets(goal, items), [], "extractive", note


def _llm_digest(
    goal: str,
    items: list[EvidenceItem],
    settings: Settings,
    filename: str = "",
) -> dict[str, Any]:
    from app.llm import get_router_llm_client
    from app.llm.json_io import parse_complete_object

    lines = [f"用户任务：{goal}"]
    if filename:
        lines.append(f"当前文档：{filename}")
    lines += ["", "证据列表："]
    for item in items:
        lines.append(format_evidence_block(item.text, index=item.index, source=filename, page=item.page))
    allowed = ", ".join(str(item.index) for item in items)
    lines.append(
        "请输出 JSON。text 用连贯中文，不要 Markdown。"
        f"sources 只能取这些编号：{allowed}；编号不在其中的要点会被整条丢弃。"
        "若本文与用户任务无关，off_topic=true 且 bullets=[]，不要做类比。"
    )
    client = get_router_llm_client(settings)
    prompt = "\n".join(lines)
    budget = min(1600, max(600, int(getattr(settings, "llm_max_output_tokens", 1024))))
    raw = client.generate(
        system=DIGEST_SYSTEM, prompt=prompt, max_output_tokens=budget, temperature=0.0, purpose="digest"
    ).text
    data, _ = parse_complete_object(raw)
    if data is None or not isinstance(data.get("bullets", []), list) or not isinstance(data.get("gaps", []), list):
        raise ValueError("digest结果结构无法安全恢复；未自动重新生成")
    return data


def _extractive_bullets(goal: str, items: list[EvidenceItem]) -> list[dict[str, Any]]:
    from app.corpus.relevance import coverage, tokenize, topic_tokens
    from app.generate.synthesis import plain_prose

    topic = topic_tokens(goal) or tokenize(goal)
    bullets: list[dict[str, Any]] = []
    for item in items[:8]:
        parts = [piece.strip() for piece in _SENTENCE_RE.split(item.text or "") if piece.strip()]
        text = "".join(parts[:2]).strip()
        if len(text) > 280:
            text = text[:279].rstrip() + "…"
        text = plain_prose(text)
        if not text:
            continue
        if topic and coverage(topic, tokenize(text)) < 0.12 and not (topic & tokenize(text)):
            continue
        bullets.append({"text": text, "sources": [item.index]})
        if len(bullets) >= 4:
            break
    return bullets


def _ensure_node(state: AgentState, doc, items: list[EvidenceItem], status: str) -> SynthesisNode:
    node_id = f"digest-{doc.doc_id}"
    node = next((item for item in state.nodes if item.id == node_id), None)
    if node is None:
        node = SynthesisNode(
            id=node_id,
            title=doc.filename,
            query=state.goal,
            question=f"读懂《{doc.filename}》中与任务相关的内容",
            status=status,
            doc_id=doc.doc_id,
        )
        state.nodes.append(node)
    node.status = status
    node.chunk_ids = [item.chunk_id for item in items]
    return node


def _clean_bullets(raw: Any, valid: set[int]) -> list[dict[str, Any]]:
    from app.generate.synthesis import plain_prose

    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = plain_prose(str(item.get("text") or "").strip())
        if not text:
            continue
        sources: list[int] = []
        for number in item.get("sources") or []:
            try:
                n = int(number)
            except (TypeError, ValueError):
                continue
            if valid and n not in valid:
                continue
            if n not in sources:
                sources.append(n)
        if valid and not sources:
            continue
        out.append({"text": text, "sources": sources})
    return out


def _parse_json(raw: str) -> dict[str, Any] | None:
    from app.llm import parse_json_object

    return parse_json_object(raw)
