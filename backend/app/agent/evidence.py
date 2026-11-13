"""Evidence merge and global culling (ranking, not veto)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from app.agent.state import AgentState
from app.generate.prompt import EvidenceItem


def merge_evidence(existing: list[EvidenceItem], fresh: list[EvidenceItem]) -> list[EvidenceItem]:
    merged: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in existing + fresh:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        merged.append(replace(item, index=len(merged) + 1))
    return merged


def select_global_evidence(state: AgentState, *, cap: int) -> dict[str, Any]:
    """Keep a global Top-K. Each retrieved node keeps its best hit; the rest compete by score.

    Multi-node hits get a small rank bonus. Neighbors follow their parent hit and are
    dropped before any hit when the budget is tight.
    """
    cap = max(1, int(cap))
    items = list(state.evidence)
    if len(items) <= cap:
        return {"kept": len(items), "dropped": 0, "note": ""}

    hits = [item for item in items if getattr(item, "role", "hit") not in {"neighbor", "graph"}]
    neighbors = [item for item in items if getattr(item, "role", "hit") == "neighbor"]
    graphs = [item for item in items if getattr(item, "role", "hit") == "graph"]
    membership: Counter[str] = Counter()
    for node in state.nodes:
        for chunk_id in node.chunk_ids:
            membership[chunk_id] += 1

    guaranteed: list[EvidenceItem] = []
    guaranteed_ids: set[str] = set()
    for node in state.nodes:
        if node.status != "retrieved":
            continue
        best: EvidenceItem | None = None
        best_key = (-1.0, 0)
        for item in hits:
            if item.chunk_id not in node.chunk_ids:
                continue
            key = (float(item.score or 0.0), -int(item.index or 0))
            if key > best_key:
                best = item
                best_key = key
        if best is not None and best.chunk_id not in guaranteed_ids:
            guaranteed.append(best)
            guaranteed_ids.add(best.chunk_id)

    ranked = [item for item in hits if item.chunk_id not in guaranteed_ids]
    ranked.sort(
        key=lambda item: (
            -(float(item.score or 0.0) + 0.05 * max(0, membership[item.chunk_id] - 1)),
            item.index,
        )
    )
    kept_hits = list(guaranteed)
    kept_ids = set(guaranteed_ids)
    for item in ranked:
        if len(kept_hits) >= cap:
            break
        kept_hits.append(item)
        kept_ids.add(item.chunk_id)

    parent_kept = kept_ids
    ranked_nb = [item for item in neighbors if item.parent_chunk_id in parent_kept or item.chunk_id not in kept_ids]
    ranked_nb.sort(
        key=lambda item: (
            0 if item.parent_chunk_id in parent_kept else 1,
            item.page,
            item.start_offset,
        )
    )
    kept_nb: list[EvidenceItem] = []
    total = len(kept_hits)
    for item in ranked_nb:
        if item.chunk_id in kept_ids:
            continue
        if total >= cap:
            break
        kept_nb.append(item)
        kept_ids.add(item.chunk_id)
        total += 1

    ranked_graph = [item for item in graphs if item.chunk_id not in kept_ids]
    ranked_graph.sort(key=lambda item: -float(item.score or 0.0))
    kept_graph: list[EvidenceItem] = []
    for item in ranked_graph:
        if total >= cap:
            break
        kept_graph.append(item)
        kept_ids.add(item.chunk_id)
        total += 1

    combined = kept_hits + kept_nb + kept_graph
    combined.sort(
        key=lambda item: (
            item.doc_id,
            item.page,
            item.start_offset,
            0 if getattr(item, "role", "hit") == "hit" else 1 if getattr(item, "role", "hit") == "neighbor" else 2,
        )
    )
    rebuilt: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in combined:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        rebuilt.append(replace(item, index=len(rebuilt) + 1))

    dropped = len(items) - len(rebuilt)
    state.evidence = rebuilt
    chunk_to_index = {item.chunk_id: item.index for item in rebuilt}
    remapped: dict[str, list[int]] = {}
    for node in state.nodes:
        node.chunk_ids = [cid for cid in node.chunk_ids if cid in chunk_to_index]
        remapped[node.id] = [chunk_to_index[cid] for cid in node.chunk_ids]
    state.evidence_by_node = remapped
    state.hit_count = len(rebuilt)
    note = f"全局取舍：{len(items)} 条证据保留 {len(rebuilt)} 条" if dropped else ""
    return {"kept": len(rebuilt), "dropped": dropped, "note": note}
