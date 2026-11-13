"""LTM read/write whitelist. Read scope == write scope."""

from __future__ import annotations

from typing import Any

from app.agent.memory import TURN_CLARIFY, TURN_OFFTOPIC, TURN_RETRIEVE, TURN_SUMMARIZE, is_corpus_ask

TURN_EXPAND = "expand_section"
TURN_INVENTORY = "inventory"
TURN_NEW_TOPIC = "new_topic"

# kind → turns that may read/write it into decision inputs (not retrieve slots).
KIND_TURNS: dict[str, frozenset[str]] = {
    "gap_confirmed": frozenset({TURN_NEW_TOPIC, TURN_RETRIEVE}),
    "empty_query": frozenset({TURN_NEW_TOPIC, TURN_RETRIEVE}),
    "route_failure": frozenset({TURN_CLARIFY}),
    "user_pref": frozenset({TURN_NEW_TOPIC, TURN_RETRIEVE, TURN_SUMMARIZE, TURN_CLARIFY}),
    "session_distill": frozenset(),  # vector aux only; never a new_topic decision input
}

RETRIEVE_SLOT_KEYS = frozenset({"goal_surface", "subject_slots", "question_slots"})
LTM_HINT_KEY = "gap_hints"


def resolve_turn_kind(goal: str, brief: str, *, default: str = TURN_RETRIEVE) -> str:
    from app.agent.memory import classify_turn_kind, is_scope_meta

    if is_corpus_ask(goal):
        return TURN_INVENTORY
    text = (goal or "").strip()
    if default == TURN_EXPAND or ("展开" in text and "节" in text):
        kind = classify_turn_kind(goal, brief)
        return TURN_EXPAND if kind == TURN_RETRIEVE and brief else kind
    return classify_turn_kind(goal, brief) if brief else TURN_NEW_TOPIC


def ltm_readable(kind: str, turn_kind: str) -> bool:
    allowed = KIND_TURNS.get(kind, frozenset())
    return turn_kind in allowed


def apply_ltm_to_state(
    *,
    turn_kind: str,
    retrieve_slots: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Inject LTM without touching retrieve slots. Returns extra state fields."""
    hints: list[dict[str, Any]] = []
    compose_mode = ""
    for item in items:
        if item.get("stale") or item.get("retracted"):
            continue
        kind = str(item.get("kind") or "")
        if not ltm_readable(kind, turn_kind):
            continue
        if turn_kind in {TURN_INVENTORY, TURN_NEW_TOPIC, TURN_RETRIEVE, TURN_EXPAND}:
            if kind in {"gap_confirmed", "empty_query"}:
                hints.append(
                    {
                        "key": item.get("key"),
                        "kind": kind,
                        "stale": False,
                    }
                )
    if turn_kind == TURN_INVENTORY:
        retrieve_slots["subject_slots"] = []
        retrieve_slots["question_slots"] = []
    # Never copy hints into retrieve_slots.
    from app.memory.query import assert_retrieve_slots

    assert_retrieve_slots(retrieve_slots)
    del compose_mode
    return {LTM_HINT_KEY: hints}
