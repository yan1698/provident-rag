"""Event projection only. Every field maps to a source_event_id."""

from __future__ import annotations

from typing import Any

PROMOTION_RULE_VERSION = 1

ALLOWED_KINDS = frozenset({"gap_confirmed", "empty_query", "route_failure", "user_pref", "session_distill"})
ALLOWED_PAYLOAD_KEYS = frozenset({"query", "coverage_rate", "reason_code"})


def extractor_allows_run(run: dict[str, Any]) -> bool:
    """Defensive filter. Async batch extractors must use this, not control-flow luck."""
    from app.memory.healthy import healthy

    status = str(run.get("status") or "")
    if run.get("rolled_back") or status in {"cancelled", "crashed", "error"}:
        return False
    return healthy(run)


def events_from_steps(run_id: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audit projection of planner steps. Not a second control-flow log."""
    events: list[dict[str, Any]] = []
    for index, step in enumerate(steps or []):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "")
        obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        etype = action
        if action == "retrieve" and int(obs.get("count") or 0) == 0:
            etype = "empty_retrieve"
        elif action == "refuse":
            etype = "refuse"
        events.append(
            {
                "id": f"{run_id}:{index}",
                "type": etype,
                "action": action,
                "input": step.get("input"),
                "observation": obs,
            }
        )
    return events


def project(
    events: list[dict[str, Any]],
    *,
    model_extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map events to memory drafts. model_extra may only supply kind + source_event_ids."""
    extra = dict(model_extra or {})
    overflow = {key: extra[key] for key in extra if key not in {"kind", "source_event_ids"}}
    drafts: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        kind = _kind_from_event(event)
        if extra.get("source_event_ids") == [event_id] and extra.get("kind") in ALLOWED_KINDS:
            kind = str(extra["kind"])
        if kind not in ALLOWED_KINDS:
            continue
        payload, origins = _payload_from_event(event, event_id)
        draft = {
            "kind": kind,
            "key": _key_from_event(event),
            "payload": payload,
            "source_event_ids": [event_id],
            "origins": origins,
            "rule_version": PROMOTION_RULE_VERSION,
            "extractor_overflow": overflow,
        }
        drafts.append(draft)
    return drafts


def assert_field_origins(draft: dict[str, Any], events: list[dict[str, Any]]) -> None:
    ids = {str(item.get("id") or "") for item in events}
    for event_id in draft.get("source_event_ids") or []:
        if str(event_id) not in ids:
            raise AssertionError(f"source_event_id missing: {event_id}")
    origins = draft.get("origins") or {}
    for field, event_id in origins.items():
        if str(event_id) not in ids:
            raise AssertionError(f"origin[{field}] not in events")
    why = str((draft.get("payload") or {}).get("why") or "")
    if why:
        raise AssertionError("free-text why is not a projected field")


def _kind_from_event(event: dict[str, Any]) -> str:
    etype = str(event.get("type") or event.get("action") or "")
    if etype in {"refuse", "empty_retrieve"}:
        return "empty_query"
    if etype == "gap":
        return "gap_confirmed"
    if etype in {"guard", "route_failure"}:
        return "route_failure"
    return ""


def _key_from_event(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    obs = event.get("observation") if isinstance(event.get("observation"), dict) else {}
    return str(
        payload.get("key") or payload.get("topic") or obs.get("query") or event.get("input") or event.get("id") or ""
    )


def _payload_from_event(event: dict[str, Any], event_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    obs = event.get("observation") if isinstance(event.get("observation"), dict) else {}
    payload: dict[str, Any] = {}
    origins: dict[str, str] = {"kind": event_id, "key": event_id}
    if "query" in obs or event.get("input"):
        payload["query"] = str(obs.get("query") or event.get("input") or "")
        origins["payload.query"] = event_id
    if obs.get("coverage_rate") is not None:
        payload["coverage_rate"] = obs.get("coverage_rate")
        origins["payload.coverage_rate"] = event_id
    if obs.get("reason_code") or event.get("guard"):
        payload["reason_code"] = str(obs.get("reason_code") or event.get("guard") or "")
        origins["payload.reason_code"] = event_id
    return payload, origins
