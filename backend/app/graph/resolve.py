"""G1 alias merge. Prefer duplicate entities over merging 重排序 with 重排序模型."""

from __future__ import annotations

from app.graph.schema import DraftEntity, GraphDrafts, norm_name, prefix_compatible


def resolve_drafts(drafts: GraphDrafts) -> GraphDrafts:
    """Collapse aliases inside one batch. Different kinds never merge."""
    canon: dict[tuple[str, str], DraftEntity] = {}
    alias_index: dict[tuple[str, str], tuple[str, str]] = {}
    kept: list[DraftEntity] = []
    for item in drafts.entities:
        key = (item.kind, norm_name(item.name))
        if not key[1]:
            continue
        hit = _find_canon(item, canon, alias_index)
        if hit is None:
            canon[key] = item
            alias_index[key] = key
            for alias in item.aliases:
                alias_key = (item.kind, norm_name(alias))
                if alias_key[1] and prefix_compatible(item.name, alias):
                    alias_index[alias_key] = key
            kept.append(item)
            continue
        if item.name not in hit.aliases and norm_name(item.name) != norm_name(hit.name):
            hit.aliases = list(dict.fromkeys([*hit.aliases, item.name, *item.aliases]))
        alias_index[key] = (hit.kind, norm_name(hit.name))
    rename: dict[tuple[str, str], DraftEntity] = {}
    for item in kept:
        rename[(item.kind, norm_name(item.name))] = item
        for alias in item.aliases:
            rename[(item.kind, norm_name(alias))] = item
    edges = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for edge in drafts.edges:
        src = rename.get((edge.src_kind, norm_name(edge.src_name)))
        dst = rename.get((edge.dst_kind, norm_name(edge.dst_name)))
        src_name = src.name if src is not None else edge.src_name
        dst_name = dst.name if dst is not None else edge.dst_name
        src_kind = src.kind if src is not None else edge.src_kind
        dst_kind = dst.kind if dst is not None else edge.dst_kind
        if src_name == dst_name and src_kind == dst_kind:
            continue
        key = (src_name, src_kind, edge.relation, dst_name, dst_kind)
        if key in seen:
            continue
        seen.add(key)
        edge.src_name = src_name
        edge.src_kind = src_kind
        edge.dst_name = dst_name
        edge.dst_kind = dst_kind
        edges.append(edge)
    drafts.entities = kept
    drafts.edges = edges
    return drafts


def _find_canon(
    item: DraftEntity,
    canon: dict[tuple[str, str], DraftEntity],
    alias_index: dict[tuple[str, str], tuple[str, str]],
) -> DraftEntity | None:
    key = (item.kind, norm_name(item.name))
    mapped = alias_index.get(key)
    if mapped is not None:
        hit = canon.get(mapped)
        if hit is not None and prefix_compatible(hit.name, item.name):
            return hit
    for alias in item.aliases:
        mapped = alias_index.get((item.kind, norm_name(alias)))
        if mapped is None:
            continue
        hit = canon.get(mapped)
        if hit is not None and prefix_compatible(hit.name, item.name):
            return hit
    for other in canon.values():
        if other.kind != item.kind:
            continue
        names = [other.name, *other.aliases]
        if any(norm_name(name) == norm_name(item.name) for name in names):
            if prefix_compatible(other.name, item.name):
                return other
    return None


def names_should_merge(left: str, right: str, *, aliases: list[str] | None = None) -> bool:
    """Public helper for tests: exact/alias match with the prefix veto."""
    if not prefix_compatible(left, right):
        return False
    if norm_name(left) == norm_name(right):
        return True
    blob = {norm_name(item) for item in (aliases or [])}
    return norm_name(left) in blob or norm_name(right) in blob
