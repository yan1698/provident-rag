"""Tool contract + FakeAgentStore: no Postgres, no live API."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
if (_REPO / "backend" / "app").is_dir():
    sys.path.insert(0, str(_REPO / "backend"))
    sys.path.insert(0, str(_REPO / "scripts"))
else:
    sys.path.insert(0, "/app")


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _run() -> None:
    from app.agent.guards import apply_guards
    from app.agent.planner import Plan
    from app.agent.state import AgentState
    from app.agent.store import ChunkRow, FakeAgentStore
    from app.agent.tools import AnalyzeTool, FinishTool, RefuseTool, SurveyTool, ToolContext, default_registry

    def _ctx(state: AgentState, store: FakeAgentStore | None = None) -> ToolContext:
        settings = SimpleNamespace(
            agent_synthesis_max_catalog_sections=80,
            graph_enabled=True,
            graph_expand_hops=1,
            agent_min_coverage_rate=0.5,
            deepseek_api_key="",
            llm_max_output_tokens=1024,
        )
        return ToolContext(state=state, settings=settings, store=store or FakeAgentStore())

    registry = default_registry()
    _assert("analyze" in registry.names(), registry.names())
    _assert("decompose" in registry.names(), _TEST_MATERIALS["text_002"])
    _assert("digest" in registry.names(), _TEST_MATERIALS["text_003"])
    _assert("answer" in registry.names_for("factoid"), registry.names_for("factoid"))
    _assert("answer" not in registry.names_for("synthesis"), _TEST_MATERIALS["text_004"])

    syn = AgentState(goal=_TEST_MATERIALS["text_011"], task_kind="synthesis")
    unknown = apply_guards(Plan("x", "web_search", "", "llm"), syn, [], registry=registry)
    _assert(unknown.action == "error" and unknown.source == "guard", unknown)

    store = FakeAgentStore(
        [
            ChunkRow(
                id="11111111-1111-1111-1111-111111111111",
                doc_id="22222222-2222-2222-2222-222222222222",
                page=1,
                text=_TEST_MATERIALS["text_020"],
                chunk_hash="abc",
            )
        ]
    )
    state = AgentState(goal=_TEST_MATERIALS["text_012"], task_kind="factoid")
    surveyed = SurveyTool().execute(Plan("scan", "survey"), _ctx(state, store))
    _assert(state.catalog, f"{_TEST_MATERIALS['text_001']}{surveyed.observation}")
    _assert(store.load_catalog_calls == 1, store.load_catalog_calls)

    from app.corpus.catalog import CatalogDocument

    cached = AgentState(
        goal=_TEST_MATERIALS["text_013"],
        task_kind="synthesis",
        catalog=[CatalogDocument("d1", "invoice.txt", [_TEST_MATERIALS["text_021"]])],
    )
    cached_obs = SurveyTool().execute(Plan("scan", "survey"), _ctx(cached, store))
    _assert(store.load_catalog_calls == 1, _TEST_MATERIALS["text_005"])
    _assert(cached_obs.observation.get("cached") is True, cached_obs.observation)

    analyzed = AnalyzeTool().execute(Plan("think", "analyze"), _ctx(syn, store))
    _assert(syn.task_spec is not None, analyzed.observation)
    _assert(analyzed.terminal is False, analyzed)

    refused = RefuseTool().execute(Plan("stop", "refuse"), _ctx(state, store))
    _assert(refused.terminal is True, refused)
    _assert(state.answer, _TEST_MATERIALS["text_006"])
    kept = state.answer

    finished = FinishTool().execute(
        Plan("done", "finish", _TEST_MATERIALS["text_014"]),
        _ctx(state, store),
    )
    _assert(finished.terminal is True, finished)
    _assert(state.answer == kept, _TEST_MATERIALS["text_007"])
    _assert(finished.observation.get("payload_ignored") is True, finished.observation)

    # Regression: a factoid after a long-form run must expose only its own
    # generator-fed sources, not an old outline or every retrieved candidate.
    from unittest.mock import patch
    from app.agent.tools import AnswerTool
    from app.generate.prompt import EvidenceItem
    from app.generate.service import AnswerResult

    fed = EvidenceItem(
        index=1, chunk_id="fed", doc_id="current", page=1, text=_TEST_MATERIALS["text_015"], chunk_hash="current-hash"
    )
    unused = EvidenceItem(index=2, chunk_id="unused", doc_id="current", page=1, text=_TEST_MATERIALS["text_016"])
    follow = AgentState(
        goal=_TEST_MATERIALS["text_017"],
        task_kind="factoid",
        evidence=[fed, unused],
        outline={"evidence": [{"chunk_id": "old"}]},
    )
    generated = AnswerResult(
        query=follow.goal, answer=_TEST_MATERIALS["text_018"], provider="stub", citations=[], evidence=[fed]
    )
    with patch("app.generate.service.answer_from_evidence", return_value=generated):
        AnswerTool().execute(Plan("answer", "answer"), _ctx(follow))
    _assert([e["chunk_id"] for e in follow.outline["evidence"]] == ["fed"], _TEST_MATERIALS["text_008"])
    _assert(follow.outline["evidence"][0]["chunk_hash"] == "current-hash", _TEST_MATERIALS["text_009"])

    print(_TEST_MATERIALS["text_010"])


def main() -> None:
    try:
        _run()
        return
    except (ModuleNotFoundError, ImportError):
        if Path("/app/app").is_dir():
            raise
        from lib.offline import can_use_local_api, run_script_in_api

        if not can_use_local_api():
            raise
        run_script_in_api(_HERE, _TEST_MATERIALS["text_019"])


if __name__ == "__main__":
    main()
