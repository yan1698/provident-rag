"""Observe whether cited relations still align to chunk quotes. Not a hallucination gate."""

from __future__ import annotations

from typing import Any

from app.graph.schema import quote_in_chunk


def relation_grounded_rate(
    edges: list[dict[str, Any]],
    *,
    chunk_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    """edges items: {relation, quote, chunk_id?}. Unaligned → relation_unbacked."""
    rows = [item for item in edges if isinstance(item, dict)]
    if not rows:
        return {
            "relation_grounded_rate": None,
            "relation_unbacked": [],
            "checked": 0,
        }
    texts = chunk_text or {}
    backed = 0
    unbacked: list[dict[str, str]] = []
    for item in rows:
        quote = str(item.get("quote") or "")
        chunk_id = str(item.get("chunk_id") or "")
        hay = texts.get(chunk_id) if chunk_id else str(item.get("text") or "")
        if hay is None:
            hay = str(item.get("text") or "")
        if quote and hay and quote_in_chunk(quote, hay) is not None:
            backed += 1
            continue
        unbacked.append(
            {
                "relation": str(item.get("relation") or ""),
                "src_name": str(item.get("src_name") or ""),
                "dst_name": str(item.get("dst_name") or ""),
            }
        )
    return {
        "relation_grounded_rate": backed / len(rows),
        "relation_unbacked": unbacked,
        "checked": len(rows),
    }
