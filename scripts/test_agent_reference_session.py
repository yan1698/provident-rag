"""Six-turn reference session through the rules planner.

No Postgres and no live model. Turn 1 is task_kind=synthesis.
The essay body is filled in only when stub compose does not emit paragraph
breaks; planning, retrieval, and later reference turns stay on production tools.
"""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

if os.environ.get("PROVIDENT_REFERENCE_INNER") != "1":
    try:
        import pydantic_settings  # noqa: F401
    except ModuleNotFoundError:
        _repo = Path(__file__).resolve().parents[1]
        if os.environ.get("GITHUB_ACTIONS") or not (_repo / ".env").is_file():
            raise
        _ran = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "-e",
                "PROVIDENT_REFERENCE_INNER=1",
                "-e",
                "PYTHONUTF8=1",
                "api",
                "python",
                "-",
            ],
            input=Path(__file__).read_bytes(),
            cwd=_repo,
            capture_output=True,
        )
        sys.stdout.write((_ran.stdout or b"").decode("utf-8", "replace"))
        sys.stderr.write((_ran.stderr or b"").decode("utf-8", "replace"))
        raise SystemExit(_ran.returncode)

_here = globals().get("__file__")
if _here:
    _repo = Path(_here).resolve().parents[1]
    if (_repo / "backend" / "app").is_dir():
        sys.path.insert(0, str(_repo / "backend"))
else:
    sys.path.insert(0, "/app")

from app.agent.analyze import apply_analyze
from app.agent.decompose import apply_decompose
from app.agent.guards import apply_guards
from app.agent.memory import _pack_anchor, index_paragraphs
from app.agent.planner import Plan, next_plan
from app.agent.service import _load_anchor_run, _run_by_id, _seal_answer_anchor
from app.agent.state import AgentState
from app.agent.store import ChunkRow, FakeAgentStore
from app.agent.tools import (
    ComposeTool,
    DecomposeTool,
    ToolContext,
    ToolResult,
    VerifyTool,
    _stamp_paragraphs,
    default_registry,
)

ESSAY = "\n\n".join(
    [
        _TEST_MATERIALS["text_002"],
        _TEST_MATERIALS["text_003"],
        _TEST_MATERIALS["text_004"],
        _TEST_MATERIALS["text_005"],
        _TEST_MATERIALS["text_006"],
        _TEST_MATERIALS["text_007"],
    ]
)
GOAL = _TEST_MATERIALS["text_001"]
SESSION = uuid4()


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


class _DB:
    def __init__(self, rows: list) -> None:
        self.rows = {str(row.id): row for row in rows}

    def get(self, _model: object, run_id: UUID):
        return self.rows.get(str(run_id))


class _NoModelDecompose(DecomposeTool):
    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        observation = apply_decompose(
            ctx.state,
            max_subtopics=int(getattr(ctx.settings, "agent_synthesis_max_subtopics", 6) or 6),
            score_min=float(getattr(ctx.settings, "agent_synthesis_section_score_min", 0.1) or 0.1),
            allow_model=False,
        )
        return ToolResult(observation)


class _EssayCompose(ComposeTool):
    """Keep the synthesis compose call, then guarantee a multi-paragraph answer."""

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        result = super().execute(plan, ctx)
        state = ctx.state
        if state.skips_retrieve() or state.catalog_inventory():
            return result
        if len(index_paragraphs(state.answer)) >= 3:
            return result
        state.answer = ESSAY
        state.compose_mode = "written"
        if isinstance(state.outline, dict):
            state.outline["compose_mode"] = "written"
        _stamp_paragraphs(state)
        observation = dict(result.observation)
        observation["answer"] = state.answer
        observation["compose_mode"] = "written"
        return ToolResult(observation, terminal=result.terminal)


class _PassVerify(VerifyTool):
    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        payload = {"coverage_rate": 1.0, "module_coverage_rate": 1.0, "task_coverage": 1.0}
        ctx.state.verification = payload
        return ToolResult(payload)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        agent_planner="rules",
        agent_max_retrieves=4,
        agent_synthesis_min_module_coverage=0.5,
        agent_synthesis_max_subtopics=6,
        agent_synthesis_section_score_min=0.1,
        agent_synthesis_max_catalog_sections=80,
        agent_synthesis_max_evidence=16,
        agent_hit_semantic_min=0.1,
        agent_retrieve_min_score=0.0,
        agent_min_coverage_rate=0.5,
        qa_max_context_chars=6000,
        llm_provider="stub",
        answer_llm_provider="stub",
        allow_llm_stub=True,
        app_env="test",
        deepseek_api_key="",
        llm_max_output_tokens=1024,
        graph_enabled=False,
        graph_expand_hops=1,
    )


def _registry():
    registry = default_registry()
    registry.register(_NoModelDecompose())
    registry.register(_EssayCompose())
    registry.register(_PassVerify())
    return registry


def _seed(goal: str, priors: list, runs: list) -> AgentState:
    from app.agent.memory import compose_session_memory

    brief = compose_session_memory(priors) if priors else ""
    # The selection gate must not depend on this line surviving the 1200-char cut.
    if goal == "2":
        brief = (_TEST_MATERIALS["text_031"] * 40)[:1200]
        _assert(_TEST_MATERIALS["text_032"] not in brief, _TEST_MATERIALS["text_016"])
    anchor = _load_anchor_run(_DB(runs), priors, session_id=SESSION) if priors else None
    outline = None
    prior_answer = str(getattr(priors[-1], "answer", "") or "") if priors else ""
    if anchor is not None:
        anchor_outline = anchor.outline if isinstance(anchor.outline, dict) else {}
        stored = anchor_outline.get("paragraphs") if isinstance(anchor_outline, dict) else None
        outline = {
            "paragraphs": list(stored) if isinstance(stored, list) else [],
            "source_answer": str(anchor.answer or ""),
            "anchor_run_id": str(anchor.id),
        }
        latest = priors[-1].outline if priors and isinstance(priors[-1].outline, dict) else {}
        if isinstance(latest, dict) and latest.get("pending_paragraphs"):
            outline["paragraphs"] = list(latest.get("pending_paragraphs") or [])
            outline["source_answer"] = str(latest.get("source_answer") or outline["source_answer"])
            outline["pending_action"] = str(latest.get("pending_action") or "")
    return AgentState(
        goal=goal,
        task_kind="synthesis",
        max_steps=16,
        max_retrieves=4,
        memory_brief=brief,
        prior_answer=prior_answer,
        outline=outline,
    )


def _run(goal: str, priors: list, runs: list, registry, settings) -> dict:
    state = _seed(goal, priors, runs)
    row = SimpleNamespace(id=uuid4(), session_id=SESSION)
    store = FakeAgentStore(
        [
            ChunkRow(
                id=str(uuid4()),
                doc_id=str(uuid4()),
                page=1,
                text=_TEST_MATERIALS["text_033"],
                chunk_hash="c9ff7a73683c",
            )
        ]
    )
    ctx = ToolContext(state=state, settings=settings, store=store)
    steps: list[dict] = []
    reference = None
    for _ in range(state.max_steps):
        plan = next_plan(state.working_goal(), steps, settings, planner="rules", state=state)
        plan = apply_guards(plan, state, steps, registry=registry)
        tool = registry.get(plan.action)
        _assert(tool is not None, f"missing tool {plan.action}")
        result = tool.execute(plan, ctx)
        if plan.action == "analyze" and isinstance(state.outline, dict):
            reference = state.outline.get("reference")
        steps.append({"action": plan.action})
        if result.terminal:
            break
    else:
        raise SystemExit(f"FAIL: {goal} did not finish: {[step['action'] for step in steps]}")
    _seal_answer_anchor(row, state)
    saved = SimpleNamespace(
        id=row.id,
        session_id=SESSION,
        goal=goal,
        answer=state.answer,
        outline=dict(state.outline or {}),
        task_kind=state.task_kind,
    )
    runs.append(saved)
    return {
        "run": saved,
        "actions": [step["action"] for step in steps],
        "retrieves": sum(1 for step in steps if step["action"] == "retrieve"),
        "reference": reference if isinstance(reference, dict) else {},
        "anchor_id": str((state.outline or {}).get("anchor_run_id") or ""),
    }


def _paragraph(answer: str, index: int) -> str:
    rows = index_paragraphs(answer)
    chosen = next(row for row in rows if row["index"] == index)
    return answer[chosen["start"] : chosen["end"]]


def _session() -> None:
    settings = _settings()
    registry = _registry()
    runs: list = []
    turns = [
        _run(GOAL, [], runs, registry, settings),
        _run(_TEST_MATERIALS["text_017"], runs[-3:], runs, registry, settings),
        _run(_TEST_MATERIALS["text_018"], runs[-3:], runs, registry, settings),
        _run(_TEST_MATERIALS["text_019"], runs[-3:], runs, registry, settings),
        _run(_TEST_MATERIALS["text_020"], runs[-3:], runs, registry, settings),
        _run("2", runs[-3:], runs, registry, settings),
    ]
    first = turns[0]["run"]
    origin = str(first.id)
    third = _paragraph(first.answer, 3)
    second = _paragraph(first.answer, 2)
    _assert(first.task_kind == "synthesis", first.task_kind)
    _assert(turns[0]["retrieves"] >= 1, turns[0]["actions"])
    _assert(len(index_paragraphs(first.answer)) >= 3, first.answer)
    _assert((first.outline or {}).get("paragraphs"), _TEST_MATERIALS["text_008"])
    _assert((first.outline.get("answer_anchor") or {}).get("run_id") == origin, first.outline)

    for turn in turns[1:]:
        _assert(turn["retrieves"] == 0, f"{turn['run'].goal} retrieved {turn['actions']}")
        anchor = (turn["run"].outline or {}).get("answer_anchor") or {}
        _assert(anchor.get("run_id") == origin, f"{turn['run'].goal} anchor {anchor}")

    _assert(turns[1]["reference"].get("action") == "explain", turns[1]["reference"])
    _assert(turns[1]["reference"].get("index") == 3, turns[1]["reference"])
    _assert(turns[1]["reference"].get("text") == third, turns[1]["reference"].get("text"))

    _assert(turns[2]["reference"].get("text") == third, turns[2]["reference"].get("text"))
    # Legacy answers without source identity are recited rather than expanded
    # into new facts. Their unchanged body may equal the prior explanation;
    # identity is checked by the original anchor and exact paragraph above.
    _assert(_TEST_MATERIALS["text_021"] in str(turns[2]["reference"].get("text") or ""), turns[2]["reference"])

    _assert(turns[3]["reference"].get("action") == "recite", turns[3]["reference"])
    _assert(turns[3]["run"].answer == third, turns[3]["run"].answer)

    pending = turns[4]["run"].outline or {}
    _assert(turns[4]["reference"].get("action") == "ask", turns[4]["reference"])
    _assert(pending.get("pending_action") == "expand", pending.get("pending_action"))
    _assert(pending.get("pending_paragraphs"), _TEST_MATERIALS["text_009"])

    _assert(turns[5]["reference"].get("action") == "expand", turns[5]["reference"])
    _assert(turns[5]["reference"].get("index") == 2, turns[5]["reference"])
    _assert(turns[5]["reference"].get("text") == second, turns[5]["reference"].get("text"))
    _assert(not (turns[5]["run"].outline or {}).get("pending_paragraphs"), _TEST_MATERIALS["text_010"])


def _split_fallback() -> None:
    packed = _pack_anchor(SimpleNamespace(id="turn-1", answer=ESSAY, outline={"compose_mode": "written"}))
    third = packed["paragraphs"][2]
    body = ESSAY[int(third["start"]) : int(third["end"])]
    _assert(_TEST_MATERIALS["text_022"] in body, body)
    state = AgentState(
        goal=_TEST_MATERIALS["text_023"],
        task_kind="synthesis",
        prior_answer=_TEST_MATERIALS["text_024"],
        memory_brief=_TEST_MATERIALS["text_025"],
        outline={"paragraphs": [], "source_answer": ESSAY, "anchor_run_id": "turn-1"},
    )
    apply_analyze(state)
    reference = (state.outline or {}).get("reference") or {}
    _assert(reference.get("index") == 3, reference)
    _assert(reference.get("text") == body, reference.get("text"))
    _assert(_TEST_MATERIALS["text_026"] not in str(reference.get("text") or ""), reference.get("text"))


def _session_guard() -> None:
    home = uuid4()
    other = uuid4()
    essay_id = uuid4()
    foreign = SimpleNamespace(id=essay_id, session_id=other, answer=_TEST_MATERIALS["text_027"], outline={})
    lone_id = uuid4()
    lone = SimpleNamespace(id=lone_id, session_id=None, answer=_TEST_MATERIALS["text_028"], outline={})
    db = _DB([foreign, lone])
    _assert(_run_by_id(db, str(essay_id), session_id=home) is None, _TEST_MATERIALS["text_011"])
    _assert(_run_by_id(db, str(essay_id), session_id=other) is foreign, _TEST_MATERIALS["text_012"])
    _assert(_run_by_id(db, str(lone_id), session_id=None) is lone, _TEST_MATERIALS["text_013"])
    latest = SimpleNamespace(
        id=uuid4(),
        session_id=home,
        answer=_TEST_MATERIALS["text_029"],
        outline={"compose_mode": "reference", "answer_anchor": {"run_id": str(essay_id)}},
    )
    loaded = _load_anchor_run(_DB([foreign, latest]), [latest], session_id=home)
    _assert(loaded is latest, f"foreign anchor leaked: {getattr(loaded, 'answer', None)}")
    kept = SimpleNamespace(
        id=essay_id,
        session_id=home,
        answer=ESSAY,
        outline={"compose_mode": "written", "paragraphs": index_paragraphs(ESSAY)},
    )
    pointed = SimpleNamespace(
        id=uuid4(),
        session_id=home,
        answer=_TEST_MATERIALS["text_030"],
        outline={"compose_mode": "reference", "answer_anchor": {"run_id": str(essay_id)}},
    )
    loaded = _load_anchor_run(_DB([kept, pointed]), [pointed], session_id=home)
    _assert(loaded is kept, _TEST_MATERIALS["text_014"])


def main() -> None:
    _split_fallback()
    _session_guard()
    _session()
    print(_TEST_MATERIALS["text_015"])


if __name__ == "__main__":
    main()
