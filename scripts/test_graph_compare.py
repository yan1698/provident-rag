"""G3 compare dimensions + skip when the graph is empty. No SQLAlchemy."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.agent.guards import apply_guards
from app.agent.planner import Plan
from app.agent.state import AgentState
from app.graph.metrics import compare_dimension_hit, graph_recall_lift
from app.graph.walk import compare_dimensions, questions_for_compare
from app.verify.graph import relation_grounded_rate


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def test_dimension_intersection() -> None:
    dims = compare_dimensions({"requires", "defines", "part_of"}, {"requires", "measured_by", "part_of"})
    _assert(dims == ["part_of", "requires"], f"intersection: {dims}")
    qs = questions_for_compare(["Dense", "BM25"], dims)
    _assert(len(qs) == 2 and "Dense" in qs[0] and _TEST_MATERIALS["text_002"] in qs[0], f"questions: {qs}")
    print("dimensions ok")


def test_skip_without_graph_guard() -> None:
    state = AgentState(goal=_TEST_MATERIALS["text_001"], task_kind="synthesis")
    plan = apply_guards(Plan("walk", "graph_walk", "", "rules"), state, [])
    _assert(plan.action == "analyze", f"must analyze first: {plan}")
    history = [{"action": "analyze"}]
    plan = apply_guards(Plan("walk", "graph_walk", "", "rules"), state, history)
    _assert(plan.action == "survey" and plan.source == "guard", f"G7: {plan}")
    print("G7 ok")


def test_grounding_not_hallucination() -> None:
    meta = relation_grounded_rate(
        [
            {"relation": "requires", "quote": _TEST_MATERIALS["text_003"], "text": _TEST_MATERIALS["text_004"]},
            {"relation": "causes", "quote": _TEST_MATERIALS["text_005"], "text": _TEST_MATERIALS["text_006"]},
        ]
    )
    _assert(meta["checked"] == 2, meta)
    _assert(abs(meta["relation_grounded_rate"] - 0.5) < 1e-9, meta)
    _assert(len(meta["relation_unbacked"]) == 1, meta)
    print("relation observe ok")


def test_eval_helpers() -> None:
    lift = graph_recall_lift(hits_off=["c1"], hits_on=["c1", "c2"], gold=["c1", "c2"])
    _assert(lift == 0.5, f"lift: {lift}")
    hit = compare_dimension_hit(["requires", "part_of"], ["requires", "measured_by"])
    _assert(hit == 0.5, f"dim hit: {hit}")
    print("eval helpers ok")


def main() -> None:
    test_dimension_intersection()
    test_skip_without_graph_guard()
    test_grounding_not_hallucination()
    test_eval_helpers()
    print("graph compare ok")


if __name__ == "__main__":
    main()
