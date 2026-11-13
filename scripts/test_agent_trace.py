"""Refactoring safety net: freeze the agent's action trajectory, not its answer quality.

Runs fixed goals with ``LLM_PROVIDER=stub`` and records ``[action, source, phase]`` per
step. This is **not** a quality gate — it says nothing about whether the answer is any
good, and Agent deliberately stays out of the golden gate. It exists so that a refactor
of the planner / guards / tool dispatch shows up as a diff instead of as silence.

    python scripts/test_agent_trace.py             # compare against the snapshot
    python scripts/test_agent_trace.py --update    # re-record after an intended change

A diff means one of two things: you changed agent behaviour, or the corpus underneath
changed. Check which before reaching for ``--update`` — retrieval hit counts steer
``rule_plan``, so uploading or deleting documents can legitimately move a trajectory.
"""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

SNAPSHOT = ROOT / "data" / "eval" / "agent_trace.json"

# Scenario-specific notes are maintained with local materials.
# Scenario-specific notes are maintained with local materials.
os.environ["LLM_PROVIDER"] = "stub"
os.environ["ANSWER_LLM_PROVIDER"] = "stub"
os.environ["TASK_LLM_PROVIDER"] = "qwen"
os.environ["ALLOW_LLM_STUB"] = "true"
os.environ["AGENT_PLANNER"] = "rules"
# Scenario-specific notes are maintained with local materials.
os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["AGENT_DIGEST_MAX_DOCS"] = "2"

# Scenario-specific notes are maintained with local materials.
# Scenario-specific notes are maintained with local materials.
CASES = (
    {"id": "factoid_grounded", "goal": _TEST_MATERIALS["text_005"], "task_kind": "factoid"},
    {"id": "synthesis_compare", "goal": _TEST_MATERIALS["text_006"], "task_kind": "synthesis"},
    {"id": "digest_docs", "goal": _TEST_MATERIALS["text_007"], "task_kind": "digest"},
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _owner_id(db, settings):
    """Regression helper; scenario text is supplied locally."""
    if not settings.auth_enabled:
        return None
    from sqlalchemy import select

    from app.models import User

    pinned = (getattr(settings, "eval_gate_user_id", None) or "").strip()
    if pinned:
        from uuid import UUID

        try:
            return UUID(pinned)
        except ValueError:
            return None
    row = db.execute(select(User).where(User.username == settings.eval_gate_username)).scalar_one_or_none()
    return row.id if row is not None else None


def _record(db, case: dict, settings, owner_id) -> dict:
    from app.agent.service import delete_agent_run, run_agent

    row = run_agent(
        db,
        case["goal"],
        task_kind=case["task_kind"],
        planner="rules",
        owner_id=owner_id,
        settings=settings,
    )
    trace = [[step.get("action"), step.get("source"), step.get("phase")] for step in (row.steps or [])]
    result = {
        "id": case["id"],
        "goal": case["goal"],
        "task_kind": case["task_kind"],
        "status": row.status,
        "trace": trace,
    }
    # Scenario-specific notes are maintained with local materials.
    delete_agent_run(db, row.id, owner_id=owner_id)
    return result


def _collect() -> list[dict]:
    from app.core.config import get_settings
    from app.db.session import SessionLocal

    settings = get_settings()
    _assert(
        settings.llm_provider == "stub",
        f"{_TEST_MATERIALS['text_001']}{settings.llm_provider!r}{_TEST_MATERIALS['text_002']}",
    )
    recorded: list[dict] = []
    with SessionLocal() as db:
        owner_id = _owner_id(db, settings)
        for case in CASES:
            recorded.append(_record(db, case, settings, owner_id))
    return recorded


def _diff(expected: list[dict], actual: list[dict]) -> list[str]:
    by_id = {item["id"]: item for item in expected}
    problems: list[str] = []
    for item in actual:
        want = by_id.pop(item["id"], None)
        if want is None:
            problems.append(f"{item['id']}{_TEST_MATERIALS['text_004']}")
            continue
        if want.get("status") != item["status"]:
            problems.append(f"{item['id']}: status {want.get('status')} -> {item['status']}")
        want_trace = [list(step) for step in want.get("trace") or []]
        if want_trace != item["trace"]:
            problems.append(
                f"{item['id']}: trajectory changed\n"
                f"    was: {[s[0] for s in want_trace]}\n"
                f"    now: {[s[0] for s in item['trace']]}"
            )
    for missing in by_id:
        problems.append(f"{missing}{_TEST_MATERIALS['text_003']}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=_TEST_MATERIALS["text_008"])
    parser.add_argument("--update", action="store_true", help=_TEST_MATERIALS["text_009"])
    parser.add_argument(
        "--snapshot",
        default=str(SNAPSHOT),
        help=_TEST_MATERIALS["text_010"],
    )
    args = parser.parse_args()

    target = Path(args.snapshot)
    actual = _collect()
    for item in actual:
        print(f"{item['id']:22} status={item['status']:9} {[step[0] for step in item['trace']]}")

    if args.update or not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {target}")
        return

    expected = json.loads(target.read_text(encoding="utf-8"))
    problems = _diff(expected, actual)
    if problems:
        for line in problems:
            print(f"  !! {line}")
        raise SystemExit(_TEST_MATERIALS["text_011"])
    print("\nagent trajectories unchanged")


if __name__ == "__main__":
    main()
