"""G3: compare dimensions from neighbor relation types. Skip if the graph is empty."""

from __future__ import annotations

from typing import Any

from app.graph.schema import RELATION_LABELS, norm_name


def compare_dimensions(
    relation_types_a: set[str],
    relation_types_b: set[str],
) -> list[str]:
    """Intersection of relation labels attached to two subjects' neighborhoods."""
    shared = [item for item in sorted(relation_types_a & relation_types_b) if item]
    return shared


def questions_for_compare(
    subjects: list[str],
    dimensions: list[str],
) -> list[str]:
    if len(subjects) < 2 or not dimensions:
        return []
    left, right = subjects[0], subjects[1]
    out = []
    for relation in dimensions:
        label = RELATION_LABELS.get(relation, relation)
        out.append(f"{left} 与 {right} 在「{label}」上有何不同")
    return out


def graph_walk_compare(
    db,
    subjects: list[str],
    *,
    owner_id=None,
    hops: int = 1,
    scope=None,
) -> dict[str, Any]:
    names = [item for item in subjects if str(item).strip()][:2]
    if len(names) < 2:
        return {"skipped": True, "reason": "不足两个对比主体", "dimensions": [], "edges": []}
    ents = _filter_scope(db, _entities_named(db, names, owner_id=owner_id), scope)
    if len(ents) < 2:
        return {"skipped": True, "reason": "图谱未构建或主体未入图", "dimensions": [], "edges": []}
    from app.graph.query import neighbor_entity_ids

    left_ids = _filter_ids(db, neighbor_entity_ids(db, [ents[0].id], hops=hops, owner_id=owner_id), scope)
    right_ids = _filter_ids(db, neighbor_entity_ids(db, [ents[1].id], hops=hops, owner_id=owner_id), scope)
    left_rel = _relation_types(db, left_ids, owner_id=owner_id)
    right_rel = _relation_types(db, right_ids, owner_id=owner_id)
    dimensions = compare_dimensions(left_rel, right_rel)
    edges = _edges_among(db, set(left_ids) | set(right_ids), owner_id=owner_id)
    return {
        "skipped": False,
        "reason": "",
        "subjects": [ents[0].name, ents[1].name],
        "dimensions": dimensions,
        "questions": questions_for_compare([ents[0].name, ents[1].name], dimensions),
        "edges": edges,
    }


def related_concepts(
    db,
    chunk_ids: list[str],
    *,
    owner_id=None,
    limit: int = 8,
) -> dict[str, Any]:
    """Entities evidenced in cited chunks, excluding document/section nodes."""
    from uuid import UUID as _UUID

    from sqlalchemy import or_, select

    from app.graph.schema import KINDS_EXTRACT
    from app.models import GraphEdge, GraphEntity, GraphMention

    ids: list[_UUID] = []
    for raw in chunk_ids:
        try:
            ids.append(_UUID(str(raw)))
        except ValueError:
            continue
    if not ids:
        return {"concepts": [], "edges": []}
    mentions = list(db.execute(select(GraphMention).where(GraphMention.chunk_id.in_(ids))).scalars().all())
    entity_ids = {row.entity_id for row in mentions}
    if not entity_ids:
        return {"concepts": [], "edges": []}
    stmt = select(GraphEntity).where(
        GraphEntity.id.in_(entity_ids),
        GraphEntity.kind.in_(tuple(KINDS_EXTRACT)),
    )
    if owner_id is None:
        stmt = stmt.where(GraphEntity.owner_id.is_(None))
    else:
        stmt = stmt.where(GraphEntity.owner_id == owner_id)
    entities = list(db.execute(stmt).scalars().all())[: max(1, int(limit))]
    kept = {row.id for row in entities}
    edge_stmt = select(GraphEdge).where(or_(GraphEdge.src_id.in_(kept), GraphEdge.dst_id.in_(kept)))
    if owner_id is None:
        edge_stmt = edge_stmt.where(GraphEdge.owner_id.is_(None))
    else:
        edge_stmt = edge_stmt.where(GraphEdge.owner_id == owner_id)
    edges = []
    by_id = {row.id: row for row in entities}
    for edge in db.execute(edge_stmt).scalars().all():
        if edge.src_id not in kept or edge.dst_id not in kept:
            continue
        src = by_id.get(edge.src_id)
        dst = by_id.get(edge.dst_id)
        if src is None or dst is None:
            continue
        edges.append(
            {
                "relation": edge.relation,
                "src_name": src.name,
                "dst_name": dst.name,
                "label": RELATION_LABELS.get(edge.relation, edge.relation),
            }
        )
    return {
        "concepts": [{"id": str(row.id), "name": row.name, "kind": row.kind} for row in entities],
        "edges": edges[:12],
    }


def _filter_scope(db, entities, scope):
    if scope is None or getattr(scope, "mode", "all") == "all":
        return entities
    return [row for row in entities if scope.allows(getattr(row, "collection_id", None))]


def _filter_ids(db, entity_ids, scope):
    if scope is None or getattr(scope, "mode", "all") == "all" or not entity_ids:
        return entity_ids
    from sqlalchemy import select

    from app.models import GraphEntity

    rows = list(db.execute(select(GraphEntity).where(GraphEntity.id.in_(entity_ids))).scalars().all())
    allowed = {row.id for row in rows if scope.allows(row.collection_id)}
    return [item for item in entity_ids if item in allowed]


def _entities_named(db, names: list[str], *, owner_id=None):
    from sqlalchemy import select

    from app.models import GraphEntity

    keys = [norm_name(item) for item in names]
    stmt = select(GraphEntity).where(GraphEntity.norm_name.in_(keys))
    if owner_id is None:
        stmt = stmt.where(GraphEntity.owner_id.is_(None))
    else:
        stmt = stmt.where(GraphEntity.owner_id == owner_id)
    rows = list(db.execute(stmt).scalars().all())
    by_key = {row.norm_name: row for row in rows}
    return [by_key[key] for key in keys if key in by_key]


def _relation_types(db, entity_ids, *, owner_id=None) -> set[str]:
    from sqlalchemy import or_, select

    from app.models import GraphEdge

    if not entity_ids:
        return set()
    stmt = select(GraphEdge.relation).where(or_(GraphEdge.src_id.in_(entity_ids), GraphEdge.dst_id.in_(entity_ids)))
    if owner_id is None:
        stmt = stmt.where(GraphEdge.owner_id.is_(None))
    else:
        stmt = stmt.where(GraphEdge.owner_id == owner_id)
    return {row[0] for row in db.execute(stmt).all() if row[0]}


def _edges_among(db, entity_ids, *, owner_id=None) -> list[dict[str, str]]:
    from sqlalchemy import select

    from app.models import GraphEdge, GraphEntity

    if not entity_ids:
        return []
    stmt = select(GraphEdge).where(
        GraphEdge.src_id.in_(entity_ids),
        GraphEdge.dst_id.in_(entity_ids),
    )
    if owner_id is None:
        stmt = stmt.where(GraphEdge.owner_id.is_(None))
    else:
        stmt = stmt.where(GraphEdge.owner_id == owner_id)
    names = {
        row.id: row.name
        for row in db.execute(select(GraphEntity).where(GraphEntity.id.in_(entity_ids))).scalars().all()
    }
    out = []
    for edge in db.execute(stmt).scalars().all():
        out.append(
            {
                "relation": edge.relation,
                "src_name": names.get(edge.src_id, ""),
                "dst_name": names.get(edge.dst_id, ""),
                "label": RELATION_LABELS.get(edge.relation, edge.relation),
            }
        )
    return out
