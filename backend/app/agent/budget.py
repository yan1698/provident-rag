"""Per-request step / retrieve budgets. Factoid caps stay at Day 13 values."""

from __future__ import annotations

from app.agent.classify import LONGFORM_KINDS, TASK_KIND_SYNTHESIS

FACTOID_STEP_CAP = 12


def resolve_step_limit(
    task_kind: str,
    max_steps: int | None = None,
    *,
    factoid_default: int = 8,
    synthesis_default: int = 20,
    factoid_cap: int = FACTOID_STEP_CAP,
    synthesis_cap: int | None = None,
) -> int:
    if task_kind in LONGFORM_KINDS:
        default = synthesis_default
        cap = synthesis_cap if synthesis_cap is not None else synthesis_default
    else:
        default = factoid_default
        cap = factoid_cap
    limit = default if max_steps is None else max_steps
    return max(1, min(int(limit), max(1, int(cap))))


def resolve_retrieve_limit(
    task_kind: str,
    *,
    factoid_default: int = 2,
    synthesis_default: int = 8,
) -> int:
    if task_kind in LONGFORM_KINDS:
        return max(1, int(synthesis_default))
    return max(1, int(factoid_default))


def next_failure_streak(streak: int, observation: dict | None, *, limit: int) -> tuple[int, bool]:
    """Count consecutive tool observations that carry ``error``. ``limit <= 0`` never trips."""
    failed = isinstance(observation, dict) and bool(observation.get("error"))
    if not failed:
        return 0, False
    streak += 1
    return streak, int(limit) > 0 and streak >= int(limit)


def past_deadline(deadline: float | None, now: float) -> bool:
    return deadline is not None and now >= deadline
