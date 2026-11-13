"""Retrieve query builder. Slots only — never memory / gap / summary."""

from __future__ import annotations

from typing import Any

ALLOWED_RETRIEVE_SLOT_KEYS = frozenset({"goal_surface", "subject_slots", "question_slots"})
_BANNED = ("gap", "gap_hint", "memory")


def retrieve_slots_from_user(
    goal: str,
    *,
    subjects: list[str] | None = None,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "goal_surface": (goal or "").strip(),
        "subject_slots": [str(item).strip() for item in (subjects or []) if str(item).strip()],
        "question_slots": [str(item).strip() for item in (questions or []) if str(item).strip()],
    }


def assert_retrieve_slots(slots: dict[str, Any]) -> None:
    keys = set(slots)
    extra = keys - ALLOWED_RETRIEVE_SLOT_KEYS
    if extra:
        raise AssertionError(f"illegal retrieve slot keys: {sorted(extra)}")
    for banned in _BANNED:
        if banned in slots:
            raise AssertionError(f"{banned} must not enter retrieve slots")


def build_retrieve_query(retrieve_slots: dict[str, Any]) -> str:
    assert_retrieve_slots(retrieve_slots)
    questions = list(retrieve_slots.get("question_slots") or [])
    if questions:
        return str(questions[0]).strip()
    subjects = list(retrieve_slots.get("subject_slots") or [])
    if subjects:
        return f"说明{subjects[0]}"
    return str(retrieve_slots.get("goal_surface") or "").strip()
