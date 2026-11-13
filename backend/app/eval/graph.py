"""G3 graph eval. Separate from golden.json gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.eval.golden import eval_data_file
from app.graph.metrics import compare_dimension_hit, graph_recall_lift
from app.graph.schema import quote_in_chunk
from app.models import Chunk, GraphEdge, GraphMention


def load_graph_golden(path: Path | None = None) -> dict[str, Any]:
    target = path or eval_data_file("graph_golden.json")
    if not target.is_file():
        return {"name": "graph", "edges": [], "compare": {}}
    return json.loads(target.read_text(encoding="utf-8"))


def mention_valid_rate(db: Session, *, owner_id=None) -> float | None:
    stmt = select(GraphMention, Chunk).outerjoin(Chunk, Chunk.id == GraphMention.chunk_id)
    if owner_id is not None:
        from app.models import Document

        stmt = stmt.join(Document, Document.id == GraphMention.doc_id).where(Document.owner_id == owner_id)
    rows = list(db.execute(stmt).all())
    if not rows:
        return None
    ok = 0
    for mention, chunk in rows:
        if chunk is None:
            continue
        if mention.chunk_hash and chunk.chunk_hash and mention.chunk_hash != chunk.chunk_hash:
            continue
        if quote_in_chunk(mention.quote, chunk.text or "") is not None:
            ok += 1
    return ok / len(rows)


def edge_precision(db: Session, golden: dict[str, Any] | None = None, *, owner_id=None) -> float | None:
    """If golden lists expected edges, precision against stored edges; else share of evidenced edges."""
    expected = [
        (str(item.get("src") or ""), str(item.get("relation") or ""), str(item.get("dst") or ""))
        for item in (golden or {}).get("edges") or []
        if item.get("src") and item.get("dst")
    ]
    stmt = select(GraphEdge)
    if owner_id is not None:
        stmt = stmt.where(GraphEdge.owner_id == owner_id)
    stored = list(db.execute(stmt).scalars().all())
    if expected:
        from app.models import GraphEntity

        names = {row.id: row.name for row in db.execute(select(GraphEntity)).scalars().all()}
        got = {(names.get(e.src_id, ""), e.relation, names.get(e.dst_id, "")) for e in stored}
        hit = sum(1 for item in expected if item in got)
        return hit / len(expected)
    if not stored:
        return None
    evidenced = sum(1 for item in stored if int(item.evidence_count or 0) > 0)
    return evidenced / len(stored)


def graph_eval_payload(db: Session, *, owner_id=None) -> dict[str, Any]:
    golden = load_graph_golden()
    lift_spec = golden.get("recall") or {}
    lift = graph_recall_lift(
        hits_off=list(lift_spec.get("hits_off") or []),
        hits_on=list(lift_spec.get("hits_on") or []),
        gold=list(lift_spec.get("gold") or []),
    )
    compare_spec = golden.get("compare") or {}
    dim_hit = compare_dimension_hit(
        list(compare_spec.get("predicted") or compare_spec.get("dimensions") or []),
        list(compare_spec.get("expected_dimensions") or []),
    )
    return {
        "eval_set": golden.get("name") or "graph",
        "edge_precision": edge_precision(db, golden, owner_id=owner_id),
        "mention_valid_rate": mention_valid_rate(db, owner_id=owner_id),
        "graph_recall_lift": lift,
        "compare_dimension_hit": dim_hit,
    }
