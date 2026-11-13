"""Pure graph eval helpers. Keep this file free of SQLAlchemy / pydantic_settings."""

from __future__ import annotations


def graph_recall_lift(*, hits_off: list[str], hits_on: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    gold_set = set(gold)
    off = len(set(hits_off) & gold_set) / len(gold_set)
    on = len(set(hits_on) & gold_set) / len(gold_set)
    return round(on - off, 4)


def compare_dimension_hit(predicted: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    got = {str(item) for item in predicted}
    return sum(1 for item in expected if item in got) / len(expected)
