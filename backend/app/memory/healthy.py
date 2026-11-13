"""Whole-run health window. Intentionally conservative: one mid-run circuit fails the run."""

from __future__ import annotations

from typing import Any

UNHEALTHY_MARKERS = frozenset(
    {
        "circuit_open",
        "degraded",
        "stub",
        "embed_circuit",
        "rerank_circuit",
    }
)


def _event_types(events: list[dict[str, Any]] | None) -> set[str]:
    out: set[str] = set()
    for item in events or []:
        out.add(str(item.get("type") or item.get("action") or "").strip())
        obs = item.get("observation") if isinstance(item.get("observation"), dict) else {}
        if obs.get("circuit_open") or obs.get("degraded"):
            out.add("circuit_open")
        if str(obs.get("provider") or "") == "stub":
            out.add("stub")
    return out


def healthy(run: dict[str, Any]) -> bool:
    """Entire run. Mid-turn circuit ⇒ unhealthy. Do not change to 'look at the end only'."""
    if run.get("rolled_back") or str(run.get("status") or "") in {"cancelled", "crashed"}:
        return False
    types = _event_types(list(run.get("events") or run.get("steps") or []))
    if types & UNHEALTHY_MARKERS:
        return False
    return True


def index_versions_match(retrieve_obs: dict[str, Any] | None) -> bool:
    if not isinstance(retrieve_obs, dict):
        return False
    left = retrieve_obs.get("index_version")
    right = retrieve_obs.get("committed_index_version")
    if left is None or right is None:
        return False
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return False


def retrieve_informative(retrieve_obs: dict[str, Any] | None) -> bool:
    if not isinstance(retrieve_obs, dict):
        return False
    if not retrieve_obs.get("search_executed"):
        return False
    if retrieve_obs.get("downstream_ok") is False:
        return False
    return True


def can_promote_gap(run: dict[str, Any], retrieve_obs: dict[str, Any] | None) -> bool:
    if run.get("rolled_back") or str(run.get("status") or "") == "crashed":
        return False
    if not healthy(run):
        return False
    if not retrieve_informative(retrieve_obs):
        return False
    if not index_versions_match(retrieve_obs):
        return False
    return True
