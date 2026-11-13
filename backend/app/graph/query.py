"""Neighbor helpers. Owner filter is applied when loading edges, then every hop."""

from __future__ import annotations


def walk_ids(
    seeds: list,
    edges: list[tuple],
    *,
    hops: int = 2,
) -> list:
    """Pure undirected BFS. Used by expand and by tests with no database."""
    adj: dict = {}
    for src, dst in edges:
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    seen = set(seeds)
    frontier = list(seeds)
    for _ in range(max(0, hops)):
        nxt = []
        for node in frontier:
            for other in adj.get(node, ()):
                if other in seen:
                    continue
                seen.add(other)
                nxt.append(other)
        frontier = nxt
        if not frontier:
            break
    return list(seen)


def neighbor_entity_ids(db, seed_ids, *, hops: int = 2, owner_id=None):
    """Undirected walk from seed entities. Seeds themselves are included."""
    from uuid import UUID

    from sqlalchemy import select

    from app.models import GraphEdge

    seeds = [item for item in seed_ids if item]
    if not seeds:
        return []
    hops = max(1, min(int(hops or 1), 3))
    stmt = select(GraphEdge.src_id, GraphEdge.dst_id)
    if owner_id is None:
        stmt = stmt.where(GraphEdge.owner_id.is_(None))
    else:
        stmt = stmt.where(GraphEdge.owner_id == owner_id)
    pairs = [(row[0], row[1]) for row in db.execute(stmt).all()]
    return walk_ids(seeds, pairs, hops=hops)


def seed_entities_for_chunks(db, chunk_ids, *, owner_id=None):
    from sqlalchemy import select

    from app.models import GraphEntity, GraphMention

    if not chunk_ids:
        return []
    stmt = select(GraphMention.entity_id).where(GraphMention.chunk_id.in_(chunk_ids)).distinct()
    ids = [row[0] for row in db.execute(stmt).all()]
    if not ids:
        return []
    owner_filter = GraphEntity.owner_id.is_(None) if owner_id is None else (GraphEntity.owner_id == owner_id)
    owned = set(db.execute(select(GraphEntity.id).where(GraphEntity.id.in_(ids), owner_filter)).scalars().all())
    return [item for item in ids if item in owned]
