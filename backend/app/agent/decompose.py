"""Pick catalog sections for a synthesis goal. Rules only, no domain lexicon."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.state import AgentState, SynthesisNode
from app.corpus.catalog import CatalogDocument, headings_in_order

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
_SPACE_RE = re.compile(r"\s+")
HIT_COVER_MIN = 0.34
_QUESTION_FILLER = "说明 如何 定义 要点 知识库 哪些 部分 怎么 有何 异同 根据 整理 梳理 相关 内容 一下 请 关于"


@dataclass(frozen=True)
class ScoredSection:
    doc_id: str
    filename: str
    section: str
    query: str
    score: float
    catalog_index: int


def tokenize(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text or "")}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def coverage(part: set[str], whole: set[str]) -> float:
    """How much of `part` is covered by `whole`."""
    if not part:
        return 0.0
    return len(part & whole) / len(part)


def filename_stem(filename: str) -> str:
    return Path(filename or "").stem


def node_query(section: str, filename: str) -> str:
    stem = filename_stem(filename)
    return f"{section} {stem}".strip()


def search_query(question: str, section: str = "", filename: str = "") -> str:
    """User-semantic query first; section/filename only disambiguate."""
    parts = [question.strip()]
    if section and section not in question:
        parts.append(section)
    stem = filename_stem(filename)
    if stem and stem not in question:
        parts.append(stem)
    return " ".join(part for part in parts if part).strip()


def score_pair(goal: str, filename: str, section: str) -> float:
    from app.corpus.relevance import coverage, jaccard, score_document, tokenize, topic_tokens

    topic = topic_tokens(goal) or tokenize(goal)
    section_toks = tokenize(section)
    file_toks = tokenize(filename_stem(filename))
    base = max(
        coverage(section_toks, topic),
        jaccard(topic, section_toks | file_toks),
    )
    return max(base, score_document(goal, filename, [section]))


def score_catalog(goal: str, catalog: list[CatalogDocument]) -> list[ScoredSection]:
    scored: list[ScoredSection] = []
    index = 0
    for doc in catalog:
        for section in doc.sections:
            scored.append(
                ScoredSection(
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    section=section,
                    query=node_query(section, doc.filename),
                    score=score_pair(goal, doc.filename, section),
                    catalog_index=index,
                )
            )
            index += 1
    scored.sort(key=lambda item: (-item.score, item.catalog_index))
    return scored


def select_scored(
    scored: list[ScoredSection],
    *,
    max_subtopics: int = 6,
    score_min: float = 0.1,
) -> list[ScoredSection]:
    cap = max(1, int(max_subtopics))
    floor = max(0.0, float(score_min))
    picked = [item for item in scored if item.score >= floor]
    return picked[:cap]


def nodes_from_scored(picked: list[ScoredSection], *, start: int = 1) -> list[SynthesisNode]:
    nodes: list[SynthesisNode] = []
    for offset, item in enumerate(picked):
        nodes.append(
            SynthesisNode(
                id=f"n{start + offset}",
                title=item.section,
                query=item.query,
                status="pending",
                section=item.section,
                doc_id=item.doc_id,
                question=item.section,
            )
        )
    return nodes


def section_passage(section: str, filename: str, preview: str = "") -> str:
    """Title plus a short window when the preview actually contains that title.

    Catalog preview is the document opening, not a per-section excerpt. Using it
    for every heading would make sections in the same file look identical.
    """
    title = (section or "").strip()
    stem = filename_stem(filename)
    head = f"{title} {stem}".strip()
    blob = preview or ""
    index = blob.find(title) if title else -1
    if index < 0:
        return head
    return f"{head}\n{blob[index : index + 80]}"


def semantic_section_hint(
    question: str,
    catalog: list[CatalogDocument],
    *,
    encode_query=None,
    encode_passages=None,
    min_score: float = 0.45,
) -> ScoredSection | None:
    """Cosine-match a question to a heading. ``None`` leaves the lexical hint in place."""
    text = (question or "").strip()
    if not text or not catalog:
        return None
    if encode_query is None or encode_passages is None:
        loaded = _load_section_encoders()
        if loaded is None:
            return None
        encode_query, encode_passages, loaded_floor = loaded
        if min_score == _HINT_DEFAULT:
            min_score = loaded_floor
    pairs: list[tuple[CatalogDocument, str, str]] = []
    for doc in catalog:
        preview = str(getattr(doc, "preview", "") or "")
        for section in doc.sections:
            if not str(section or "").strip():
                continue
            pairs.append((doc, section, section_passage(section, doc.filename, preview)))
    if not pairs:
        return None
    try:
        query_vec = list(encode_query(text))
        passage_vecs = [list(row) for row in encode_passages([item[2] for item in pairs])]
    except Exception:  # noqa: BLE001 — missing model must not fail decompose
        return None
    if not query_vec or len(passage_vecs) != len(pairs):
        return None
    best = max(range(len(pairs)), key=lambda index: _dot(query_vec, passage_vecs[index]))
    score = _dot(query_vec, passage_vecs[best])
    if score < float(min_score):
        return None
    doc, section, _passage = pairs[best]
    return ScoredSection(
        doc_id=doc.doc_id,
        filename=doc.filename,
        section=section,
        query=search_query(text, section, doc.filename),
        score=round(float(score), 4),
        catalog_index=best,
    )


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


_HINT_DEFAULT = 0.45


def _load_section_encoders():
    try:
        from app.core.config import get_settings
        from app.embed.model import encode_query, encode_texts
    except Exception:  # noqa: BLE001 — CI has no sentence-transformers
        return None
    try:
        settings = get_settings()
        prefix = settings.embedding_passage_prefix or ""
        floor = float(getattr(settings, "agent_section_hint_min", _HINT_DEFAULT) or _HINT_DEFAULT)
    except Exception:  # noqa: BLE001
        return None

    def _query(text: str) -> list[float]:
        return encode_query(text, settings=settings)

    def _passages(texts: list[str]) -> list[list[float]]:
        payload = [f"{prefix}{item}" if prefix else item for item in texts]
        return encode_texts(payload, settings=settings)

    return _query, _passages, floor


def hint_for_question(
    question: str,
    catalog: list[CatalogDocument],
    *,
    score_min: float = 0.1,
    encode_query=None,
    encode_passages=None,
    hint_min: float | None = None,
    allow_model: bool = False,
) -> ScoredSection | None:
    if encode_query is not None or encode_passages is not None or allow_model:
        semantic = semantic_section_hint(
            question,
            catalog,
            encode_query=encode_query,
            encode_passages=encode_passages,
            min_score=_HINT_DEFAULT if hint_min is None else float(hint_min),
        )
        if semantic is not None:
            return semantic
    picked = select_scored(score_catalog(question, catalog), max_subtopics=1, score_min=score_min)
    if not picked:
        return None
    item = picked[0]
    if not question_binds_section(question, item.section, item.filename):
        return None
    return item


def question_binds_section(question: str, section: str, filename: str) -> bool:
    """Bind a catalog section only when the sub-question actually mentions it."""
    if section and section in (question or ""):
        return True
    stem = filename_stem(filename)
    if stem and stem in (question or ""):
        return True
    filler = tokenize(_QUESTION_FILLER)
    q_toks = tokenize(question) - filler
    sec_toks = tokenize(section)
    if not q_toks or not sec_toks:
        return False
    return coverage(q_toks, sec_toks) >= 0.5


def question_text_score(question: str, text: str) -> float:
    """How much of the question's content appears in a hit. CJK bigrams preferred."""
    q = question or ""
    blob = text or ""
    if not q.strip() or not blob.strip():
        return 0.0
    filler = tokenize(_QUESTION_FILLER)
    q_toks = tokenize(q) - filler
    t_toks = tokenize(blob)
    uni = coverage(q_toks, t_toks) if q_toks else 0.0
    chars = [ch for ch in q if "\u4e00" <= ch <= "\u9fff"]
    skip = set("".join(sorted(filler)))
    bigrams = [a + b for a, b in zip(chars, chars[1:]) if a not in skip or b not in skip]
    if not bigrams:
        return uni
    haystack = normalize_text(blob)
    hit = sum(1 for gram in bigrams if gram in haystack or gram in blob)
    return max(uni, hit / len(bigrams))


def hit_covers_question(question: str, text: str, *, min_score: float = HIT_COVER_MIN) -> bool:
    return question_text_score(question, text) >= min_score


def filter_hits_for_question(
    question: str,
    hits: list[Any],
    *,
    scores: dict[str, float] | None = None,
    semantic_min: float | None = None,
    exempt_doc_ids: set[str] | None = None,
) -> list[Any]:
    """Keep hits that are semantically relevant OR lexically overlapping.

    The Cross-Encoder already ranked these hits; word overlap must not be able to
    veto it, or synonyms and English passages get filed as uncovered. Lexical
    coverage stays as a fallback for when no rerank score is available.
    """
    floor = HIT_COVER_MIN if semantic_min is None else float(semantic_min)
    exempt = exempt_doc_ids or set()
    kept: list[Any] = []
    for item in hits:
        chunk_id = str(getattr(item, "chunk_id", "") or "")
        score = (scores or {}).get(chunk_id)
        if score is not None and score >= floor:
            kept.append(item)
            continue
        if str(getattr(item, "doc_id", "") or "") in exempt:
            kept.append(item)
            continue
        if hit_covers_question(question, str(getattr(item, "text", "") or "")):
            kept.append(item)
    return kept


def nodes_from_questions(
    questions: list[str],
    catalog: list[CatalogDocument],
    *,
    max_subtopics: int = 6,
    score_min: float = 0.1,
    allow_model: bool = False,
) -> list[SynthesisNode]:
    cap = max(1, int(max_subtopics))
    nodes: list[SynthesisNode] = []
    seen: set[str] = set()
    for question in questions:
        q = (question or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        hint = hint_for_question(q, catalog, score_min=score_min, allow_model=allow_model)
        section = hint.section if hint else None
        filename = hint.filename if hint else ""
        nodes.append(
            SynthesisNode(
                id=f"n{len(nodes) + 1}",
                title=q,
                query=search_query(q, section or "", filename),
                status="pending",
                section=section,
                doc_id=hint.doc_id if hint else None,
                question=q,
            )
        )
        if len(nodes) >= cap:
            break
    return nodes


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub("", (text or "").casefold())


def map_hit_to_sections(
    text: str,
    doc_id: str,
    catalog: list[CatalogDocument],
) -> list[tuple[CatalogDocument, str]]:
    """Map one chunk onto catalog headings. Reuses the shared heading regex."""
    doc = next((item for item in catalog if item.doc_id == doc_id), None)
    if doc is None or not doc.sections:
        return []
    matched: list[tuple[CatalogDocument, str]] = []
    seen: set[str] = set()
    headings = headings_in_order([text or ""])
    blob = normalize_text(text)
    for section in doc.sections:
        hit = section in headings or (section and normalize_text(section) in blob)
        if hit and section not in seen:
            seen.add(section)
            matched.append((doc, section))
    if not matched and len(doc.sections) == 1:
        return [(doc, doc.sections[0])]
    return matched


def count_section_hits(section: str | None, fresh: list[Any], catalog: list[CatalogDocument]) -> dict[str, int]:
    """How many fresh hits sit in the node's section, versus the whole fresh set.

    Counts only. Does not reorder or drop hits.
    """
    hits = [item for item in fresh if getattr(item, "role", "hit") == "hit"]
    bound = 0
    title = (section or "").strip()
    if title:
        for item in hits:
            mapped = map_hit_to_sections(
                str(getattr(item, "text", "") or ""),
                str(getattr(item, "doc_id", "") or ""),
                catalog,
            )
            if any(name == title for _doc, name in mapped):
                bound += 1
    return {"corpus_hit_count": len(hits), "section_hit_count": bound}


def fill_nodes_from_hits(
    state: AgentState,
    hits: list[Any],
    *,
    max_subtopics: int = 6,
) -> list[SynthesisNode]:
    """Add catalog sections implied by probe hits. Never invent titles."""
    cap = max(1, int(max_subtopics))
    existing = {(node.doc_id, node.section) for node in state.nodes}
    added: list[SynthesisNode] = []
    start = len(state.nodes) + 1
    for hit in hits:
        doc_id = str(getattr(hit, "doc_id", "") or (hit.get("doc_id") if isinstance(hit, dict) else "") or "")
        text = str(getattr(hit, "text", "") or (hit.get("text") if isinstance(hit, dict) else "") or "")
        for doc, section in map_hit_to_sections(text, doc_id, state.catalog):
            key = (doc.doc_id, section)
            if key in existing:
                continue
            if len(state.nodes) + len(added) >= cap:
                return added
            question = state.goal
            node = SynthesisNode(
                id=f"n{start + len(added)}",
                title=section,
                query=search_query(question, section, doc.filename),
                status="pending",
                section=section,
                doc_id=doc.doc_id,
                question=question,
                source_sections=[{"doc_id": doc.doc_id, "section": section, "filename": doc.filename}],
            )
            added.append(node)
            existing.add(key)
    return added


def merge_catalog_nodes(
    nodes: list[SynthesisNode],
    catalog: list[CatalogDocument],
    goal: str,
    *,
    max_subtopics: int = 6,
    score_min: float = 0.1,
    intent: str = "explain",
) -> list[SynthesisNode]:
    """Keep user questions, then fill remaining slots with on-topic catalog sections.

    Compare stays at the two-sided questions. Catalog is only a hint source there.
    Off-topic documents (filename/headings vs this-turn subjects) never fill extras.
    """
    from app.corpus.relevance import OFF_TOPIC_FLOOR, score_document

    cap = max(1, int(max_subtopics))
    if intent == "compare":
        return _renumber_nodes(nodes[:cap])
    existing_keys = {(node.doc_id, node.section) for node in nodes if node.section}
    existing_sections = {node.section for node in nodes if node.section}
    extras: list[SynthesisNode] = []
    on_topic = {
        doc.doc_id
        for doc in catalog
        if score_document(goal, doc.filename, doc.sections, getattr(doc, "preview", "") or "") >= OFF_TOPIC_FLOOR
    }
    for item in select_scored(score_catalog(goal, catalog), max_subtopics=cap, score_min=score_min):
        if item.doc_id not in on_topic:
            continue
        key = (item.doc_id, item.section)
        if key in existing_keys or item.section in existing_sections:
            continue
        extras.extend(nodes_from_scored([item], start=len(nodes) + len(extras) + 1))
        existing_keys.add(key)
        existing_sections.add(item.section)
        if len(nodes) + len(extras) >= cap:
            break
    hinted = [node for node in nodes if node.section]
    unhinted = [node for node in nodes if not node.section]
    return _renumber_nodes((hinted + extras + unhinted)[:cap])


def _renumber_nodes(nodes: list[SynthesisNode]) -> list[SynthesisNode]:
    for offset, node in enumerate(nodes, start=1):
        node.id = f"n{offset}"
    return nodes


def apply_decompose(
    state: AgentState,
    *,
    max_subtopics: int = 6,
    score_min: float = 0.1,
    allow_model: bool = False,
) -> dict[str, Any]:
    """Plan task units from the user goal. Catalog only supplies source hints."""
    if state.task_spec is None:
        from app.agent.analyze import analyze_task

        state.task_spec = analyze_task(state.goal, state.requested_intent)
    questions = list(state.task_spec.questions or [])
    intent = state.task_spec.intent if state.task_spec else "explain"
    if intent == "compare" and state.graph_dimensions:
        from app.graph.walk import questions_for_compare

        dim_questions = questions_for_compare(list(state.task_spec.subjects or []), state.graph_dimensions)
        if dim_questions:
            questions = dim_questions
    if not questions:
        questions = [state.goal]
    scoring = "、".join(state.task_spec.subjects or []) if state.task_spec else ""
    scoring = scoring or (questions[0] if questions else state.goal)
    state.nodes = merge_catalog_nodes(
        nodes_from_questions(
            questions,
            state.catalog,
            max_subtopics=max_subtopics,
            score_min=score_min,
            allow_model=allow_model,
        ),
        state.catalog,
        scoring,
        max_subtopics=max_subtopics,
        score_min=score_min,
        intent=intent,
    )
    hinted = sum(1 for node in state.nodes if node.section)
    probe_needed = len(state.nodes) < 2 or hinted == 0
    return {
        "intent": state.task_spec.intent if state.task_spec else "explain",
        "nodes": [
            {
                "id": node.id,
                "title": node.title,
                "question": node.question,
                "query": node.query,
                "status": node.status,
                "doc_id": node.doc_id,
                "section": node.section,
            }
            for node in state.nodes
        ],
        "probe_needed": probe_needed,
        "scored_above": hinted,
        "catalog_sections": sum(len(doc.sections) for doc in state.catalog),
        "graph_dimensions": list(state.graph_dimensions),
        "gaps": [{"topic": gap.topic, "reason": gap.reason} for gap in state.gaps],
    }


def append_probe_nodes(
    state: AgentState,
    hits: list[Any],
    *,
    max_subtopics: int = 6,
) -> list[SynthesisNode]:
    added = fill_nodes_from_hits(state, hits, max_subtopics=max_subtopics)
    if added:
        state.nodes.extend(added)
    return added
