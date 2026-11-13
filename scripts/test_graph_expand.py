"""G2 expand: hits are never dropped for graph neighbors."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.generate.prompt import EvidenceItem
from app.graph.expand import attach_graph_candidates
from app.graph.query import walk_ids
from app.agent.evidence import select_global_evidence
from app.agent.state import AgentState, SynthesisNode


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _hit(cid: str, text: str, *, role: str = "hit", score: float = 0.5) -> EvidenceItem:
    return EvidenceItem(
        index=0,
        chunk_id=cid,
        doc_id="d",
        page=1,
        text=text,
        role=role,
        score=score,
    )


def test_walk_hops() -> None:
    ids = walk_ids(["a"], [("a", "b"), ("b", "c"), ("c", "d")], hops=2)
    _assert("a" in ids and "b" in ids and "c" in ids, f"2 hops: {ids}")
    _assert("d" not in ids, f"3rd hop leaked: {ids}")
    print("walk hops ok")


def test_tight_budget_keeps_hits() -> None:
    hits = [_hit("h1", _TEST_MATERIALS["text_005"] * 20)]
    cands = [_hit("g1", _TEST_MATERIALS["text_006"] * 40, role="graph", score=0.9)]
    out, note = attach_graph_candidates(hits, cands, scores={"g1": 0.9}, min_score=0.02, max_chars=len(hits[0].text))
    kept = {item.chunk_id for item in out}
    _assert("h1" in kept, f"hit dropped: {kept} {note}")
    _assert("g1" not in kept, f"{_TEST_MATERIALS['text_001']}{kept}")
    _assert(note["graph_dropped"] >= 1, f"should record drop: {note}")
    print("tight budget ok")


def test_cull_drops_graph_first() -> None:
    state = AgentState(goal=_TEST_MATERIALS["text_004"])
    state.nodes = [
        SynthesisNode(
            id="n1",
            title=_TEST_MATERIALS["text_007"],
            query=_TEST_MATERIALS["text_008"],
            status="retrieved",
            chunk_ids=["h1"],
        )
    ]
    state.evidence = [
        _hit("h1", "hit", score=0.8),
        _hit("n1", "nb", role="neighbor", score=0.1),
        _hit("g1", "graph", role="graph", score=0.99),
    ]
    select_global_evidence(state, cap=1)
    kept = {item.chunk_id for item in state.evidence}
    _assert(kept == {"h1"}, f"{_TEST_MATERIALS['text_002']}{kept}")
    state.evidence = [
        _hit("h1", "hit", score=0.8),
        _hit("n1", "nb", role="neighbor", score=0.1),
        _hit("g1", "graph", role="graph", score=0.99),
    ]
    state.nodes = [
        SynthesisNode(
            id="n1",
            title=_TEST_MATERIALS["text_009"],
            query=_TEST_MATERIALS["text_010"],
            status="retrieved",
            chunk_ids=["h1"],
        )
    ]
    select_global_evidence(state, cap=2)
    kept = [item.chunk_id for item in state.evidence]
    _assert("h1" in kept and "g1" not in kept, f"{_TEST_MATERIALS['text_003']}{kept}")
    print("cull order ok")


def main() -> None:
    test_walk_hops()
    test_tight_budget_keeps_hits()
    test_cull_drops_graph_first()
    print("graph expand ok")


if __name__ == "__main__":
    main()
