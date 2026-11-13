"""Benchmark-only passage projection. Never changes production hits or scoring input."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy


def mapped_ids(hits, mapping):
    """Preserve the existing Du first-occurrence projection, including one-to-many maps."""
    seen, ids = set(), []
    for hit in hits:
        for passage in mapping[str(hit.chunk_id)]:
            if passage not in seen:
                seen.add(passage)
                ids.append(passage)
    return ids


def passage_diagnostics(stages: list[dict], mapping: dict) -> list[dict]:
    """Annotate copies with real mappings only; missing mappings stay explicit.

    duplicated_passage_count counts surplus chunk→passage associations, not the
    number of distinct repeated passages (reported separately). One chunk can map
    to multiple official passages when production ingest reused identical content.
    """
    try:
        result = deepcopy(stages)
        for stage in result:
            counts: Counter = Counter()
            missing = 0
            for row in stage["candidates"]:
                passages = list(dict.fromkeys(mapping.get(row["chunk_id"], [])))
                if passages:
                    row["passage_ids"] = passages
                    counts.update(passages)
                else:
                    missing += 1
            stage.update(
                returned_chunk_count=len(stage["candidates"]),
                unique_passage_count=len(counts),
                duplicated_passage_count=sum(counts.values()) - len(counts),
                repeated_passage_count=sum(count > 1 for count in counts.values()),
                mapped_passage_association_count=sum(counts.values()),
                unmapped_chunk_count=missing,
            )
        return result
    except Exception:  # noqa: BLE001 — diagnostics must not fail the scored retrieval
        return []
