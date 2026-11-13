"""Promotion, stale, retract. No extra control flow."""

from __future__ import annotations

from typing import Any

from app.memory.extractor import PROMOTION_RULE_VERSION, extractor_allows_run
from app.memory.healthy import can_promote_gap, healthy

GAP_KINDS = frozenset({"gap_confirmed", "empty_query"})


def assert_promotable_run(run: dict[str, Any]) -> None:
    assert not run.get("rolled_back")
    assert str(run.get("status") or "") != "crashed"
    assert extractor_allows_run(run)
    assert healthy(run)


def promote_gap(
    run: dict[str, Any],
    retrieve_obs: dict[str, Any] | None,
    draft: dict[str, Any],
) -> dict[str, Any] | None:
    if not extractor_allows_run(run):
        return None
    if draft.get("kind") in GAP_KINDS and not can_promote_gap(run, retrieve_obs):
        return None
    if not can_promote_machine(run, retrieve_obs, draft):
        return None
    item = dict(draft)
    item["stale"] = False
    item["retracted"] = False
    item["rule_version"] = int(draft.get("rule_version") or PROMOTION_RULE_VERSION)
    item["source_run_id"] = run.get("id")
    return item


def can_promote_machine(
    run: dict[str, Any],
    retrieve_obs: dict[str, Any] | None,
    draft: dict[str, Any],
) -> bool:
    if draft.get("kind") in GAP_KINDS:
        return can_promote_gap(run, retrieve_obs)
    return extractor_allows_run(run)


def mark_stale_on_ingest(
    items: list[dict[str, Any]],
    *,
    ingest_id: str,
    ingest_time,
    run_time_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        if row.get("kind") in GAP_KINDS and not row.get("retracted"):
            run_id = str(row.get("source_run_id") or "")
            source_time = run_time_by_id.get(run_id)
            if source_time is None or source_time <= ingest_time:
                row["stale"] = True
                row["stale_ingest_id"] = ingest_id
        out.append(row)
    return out


def should_skip_retrieve_for_gap(item: dict[str, Any], *, digest_unchanged: bool) -> bool:
    if item.get("retracted") or item.get("stale"):
        return False
    if item.get("kind") not in GAP_KINDS:
        return False
    return digest_unchanged


def retract_stale_on_hit(item: dict[str, Any], *, answered: bool) -> dict[str, Any]:
    row = dict(item)
    if row.get("stale") and answered:
        row["retracted"] = True
        row["stale"] = False
        row["version"] = int(row.get("version") or 1) + 1
    return row
