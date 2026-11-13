"""Sort graph entities by first grounded mention in document order, not by title."""

from __future__ import annotations

from typing import Any
from uuid import UUID

_KIND_RANK = {
    "document": 0,
    "doc_section": 1,
    "concept": 2,
    "method": 2,
    "metric": 2,
    "tool": 2,
    "role": 2,
}


def _chunk_key(chunk_pos: dict[Any, tuple[int, int]], chunk_id: Any) -> tuple[int, int]:
    if chunk_id in chunk_pos:
        return chunk_pos[chunk_id]
    text = str(chunk_id or "")
    if text in chunk_pos:
        return chunk_pos[text]
    try:
        uid = UUID(text)
        if uid in chunk_pos:
            return chunk_pos[uid]
    except (TypeError, ValueError):
        pass
    return (9_999_999, 9_999_999)


def entity_doc_pos(
    *,
    kind: str,
    mentions: list[Any],
    chunk_pos: dict[Any, tuple[int, int]] | None = None,
) -> tuple[int, int, int, int]:
    """Return (kind_rank, page, start_offset, start_char) for stable document-order sort."""
    chunk_pos = chunk_pos or {}
    kind_rank = _KIND_RANK.get(kind, 3)
    best: tuple[int, int, int] | None = None
    pool = [item for item in mentions if getattr(item, "edge_id", None) is None] or list(mentions)
    for men in pool:
        chunk_id = getattr(men, "chunk_id", None)
        page, offset = _chunk_key(chunk_pos, chunk_id)
        start = int(getattr(men, "start_char", 0) or 0)
        key = (page, offset, start)
        if best is None or key < best:
            best = key
    if best is None:
        best = (9_999_999, 9_999_999, 9_999_999)
    return (kind_rank, best[0], best[1], best[2])


def entity_sort_key(entity: Any, mentions: list[Any], chunk_pos: dict[Any, tuple[int, int]] | None) -> tuple:
    pos = entity_doc_pos(kind=getattr(entity, "kind", ""), mentions=mentions, chunk_pos=chunk_pos)
    name = str(getattr(entity, "name", "") or "")
    return (*pos, name)
