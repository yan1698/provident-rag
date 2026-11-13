"""Rule skeleton: document → section tree with grounded mentions. No LLM."""

from __future__ import annotations

from typing import Any

from app.corpus.catalog import headings_in_order, iter_headings, label_chunk_sections
from app.graph.schema import DraftEdge, DraftEntity, GraphDrafts, first_sentence, quote_in_chunk

QUOTE_LIMIT = 80


class RuleDrafts(GraphDrafts):
    """Back-compat alias used by G0 tests."""


def build_document_graph(document: Any, chunks: list[Any]) -> RuleDrafts:
    """Build a part_of tree from ATX headings. Empty chunks → empty draft."""
    ordered = sorted(chunks, key=lambda row: (row.page, row.start_offset))
    if not ordered:
        return RuleDrafts()
    texts = [row.text or "" for row in ordered]
    filename = document.filename or "unnamed"
    first = ordered[0]
    doc_quote = _clip_prefix(first.text or "", QUOTE_LIMIT)
    span = quote_in_chunk(doc_quote, first.text or "")
    if span is None:
        return RuleDrafts()

    drafts = RuleDrafts()
    drafts.entities.append(
        DraftEntity(
            name=filename,
            kind="document",
            quote=doc_quote,
            chunk_id=str(first.id),
            chunk_hash=first.chunk_hash or "",
            start_char=span[0],
            end_char=span[1],
        )
    )

    sections = headings_in_order(texts, fallback=filename)
    labels = label_chunk_sections(texts)
    first_span_by_section = _first_heading_spans(ordered)
    _add_section_entities(drafts, filename, ordered, labels, first_span_by_section)
    _add_part_of_edges(drafts, filename, ordered, first_span_by_section, sections)
    return drafts


def _clip_prefix(text: str, limit: int) -> str:
    blob = text or ""
    if len(blob) <= limit:
        return blob
    return blob[:limit]


def _first_heading_spans(chunks: list[Any]) -> dict[str, tuple[Any, str, int, int]]:
    """First occurrence of each heading title → (chunk, quote, start, end)."""
    found: dict[str, tuple[Any, str, int, int]] = {}
    for chunk in chunks:
        for _level, title, _start in iter_headings(chunk.text or ""):
            if title in found:
                continue
            quote = title
            span = quote_in_chunk(quote, chunk.text or "")
            if span is None:
                continue
            found[title] = (chunk, quote, span[0], span[1])
    return found


def _section_quote(
    title: str,
    chunk: Any,
    first_span_by_section: dict[str, tuple[Any, str, int, int]],
) -> tuple[str, int, int] | None:
    hit = first_span_by_section.get(title)
    if hit and hit[0].id == chunk.id:
        return hit[1], hit[2], hit[3]
    quote = first_sentence(chunk.text or "", limit=QUOTE_LIMIT)
    span = quote_in_chunk(quote, chunk.text or "")
    if span is None:
        return None
    return quote, span[0], span[1]


def _add_section_entities(
    drafts: RuleDrafts,
    filename: str,
    chunks: list[Any],
    labels: list[tuple[str, ...]],
    first_span_by_section: dict[str, tuple[Any, str, int, int]],
) -> None:
    seen: set[str] = set()
    for chunk, sections in zip(chunks, labels):
        names = sections or ((filename,) if not first_span_by_section else ())
        for title in names:
            quote_row = _section_quote(title, chunk, first_span_by_section)
            if quote_row is None:
                continue
            quote, start, end = quote_row
            drafts.entities.append(
                DraftEntity(
                    name=title,
                    kind="doc_section",
                    quote=quote,
                    chunk_id=str(chunk.id),
                    chunk_hash=chunk.chunk_hash or "",
                    start_char=start,
                    end_char=end,
                )
            )
            seen.add(title)
    fallback = headings_in_order([c.text or "" for c in chunks], fallback=filename)
    for title in fallback:
        if title in seen:
            continue
        hit = first_span_by_section.get(title)
        if hit is None and title == filename:
            chunk = chunks[0]
            quote = _clip_prefix(chunk.text or "", QUOTE_LIMIT)
            span = quote_in_chunk(quote, chunk.text or "")
            if span is None:
                continue
            drafts.entities.append(
                DraftEntity(
                    name=title,
                    kind="doc_section",
                    quote=quote,
                    chunk_id=str(chunk.id),
                    chunk_hash=chunk.chunk_hash or "",
                    start_char=span[0],
                    end_char=span[1],
                )
            )
            continue
        if hit is None:
            continue
        chunk, quote, start, end = hit
        drafts.entities.append(
            DraftEntity(
                name=title,
                kind="doc_section",
                quote=quote,
                chunk_id=str(chunk.id),
                chunk_hash=chunk.chunk_hash or "",
                start_char=start,
                end_char=end,
            )
        )


def _add_part_of_edges(
    drafts: RuleDrafts,
    filename: str,
    chunks: list[Any],
    first_span_by_section: dict[str, tuple[Any, str, int, int]],
    sections: list[str],
) -> None:
    stack: list[tuple[int, str]] = []
    seen_edge: set[tuple[str, str, str, str]] = set()
    walked = False
    for chunk in chunks:
        for level, title, _start in iter_headings(chunk.text or ""):
            walked = True
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_name = stack[-1][1] if stack else filename
            parent_kind = "doc_section" if stack else "document"
            _append_edge(
                drafts,
                seen_edge,
                src_name=title,
                src_kind="doc_section",
                dst_name=parent_name,
                dst_kind=parent_kind,
                chunk=chunk,
                first_span_by_section=first_span_by_section,
            )
            stack.append((level, title))
    if walked:
        return
    if not sections:
        return
    title = sections[0]
    chunk = chunks[0]
    _append_edge(
        drafts,
        seen_edge,
        src_name=title,
        src_kind="doc_section",
        dst_name=filename,
        dst_kind="document",
        chunk=chunk,
        first_span_by_section=first_span_by_section,
        fallback_quote=_clip_prefix(chunk.text or "", QUOTE_LIMIT),
    )


def _append_edge(
    drafts: RuleDrafts,
    seen_edge: set[tuple[str, str, str, str]],
    *,
    src_name: str,
    src_kind: str,
    dst_name: str,
    dst_kind: str,
    chunk: Any,
    first_span_by_section: dict[str, tuple[Any, str, int, int]],
    fallback_quote: str = "",
) -> None:
    key = (src_name, src_kind, dst_name, dst_kind)
    if key in seen_edge:
        return
    if src_name == dst_name and src_kind == dst_kind:
        return
    hit = first_span_by_section.get(src_name)
    if hit is not None:
        use_chunk, quote, start, end = hit
    else:
        use_chunk = chunk
        quote = fallback_quote or first_sentence(chunk.text or "", limit=QUOTE_LIMIT)
        span = quote_in_chunk(quote, use_chunk.text or "")
        if span is None:
            return
        start, end = span
    if quote_in_chunk(quote, use_chunk.text or "") is None:
        return
    seen_edge.add(key)
    drafts.edges.append(
        DraftEdge(
            src_name=src_name,
            src_kind=src_kind,
            dst_name=dst_name,
            dst_kind=dst_kind,
            relation="part_of",
            quote=quote,
            chunk_id=str(use_chunk.id),
            chunk_hash=use_chunk.chunk_hash or "",
            source="rules",
            start_char=start,
            end_char=end,
        )
    )
