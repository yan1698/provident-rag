"""Deterministic delivery reconciliation; no provider calls or oracle IDs."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.agent.state import AgentState, TaskSpec
from app.agent.store import ChunkRow, FakeAgentStore
from app.generate.prompt import EvidenceItem
from app.generate.synthesis import compose_synthesis
from app.verify.service import reconcile_delivery


def fixture():
    text = _TEST_MATERIALS["text_001"]
    sha = hashlib.sha256(text.encode()).hexdigest()
    row = ChunkRow(id="one", doc_id="doc", text=text, page=1, chunk_hash="hash", section_path="Manual / Backup")
    proof = {
        "chunk_id": "one",
        "doc_id": "doc",
        "hash": "hash",
        "text_sha256": sha,
        "section_id": "section-one",
        "quote": text,
    }
    requirement = {"id": "connection", "text": _TEST_MATERIALS["text_004"], "sources": [proof]}
    paragraph = {"text": text, "sources": [1]}
    state = AgentState(
        goal=_TEST_MATERIALS["text_006"], task_kind="synthesis", task_spec=TaskSpec(questions=["Connection?"])
    )
    state.compose_mode = "written"
    state.evidence = [EvidenceItem(index=1, chunk_id="one", doc_id="doc", page=1, text=text, chunk_hash="hash")]
    state.reading = {
        "mode": "bounded",
        "receipts": [
            {"chunk_id": "one", "hash": "hash", "text_sha256": sha, "section_ids": ["section-one"], "status": "read"}
        ],
    }
    state.outline = {"modules": [{"paragraphs": [paragraph]}]}
    state.evidence_selection = {
        "compose": {"provider": "qwen", "provider_response_received": True},
        "delivery": {"origin": "model", "raw_content": copy.deepcopy(state.outline["modules"])},
    }
    state.verification = {"claims": [{"text": text, "status": "supported", "citation_indices": [1]}]}
    reviews = {"connection": {"conclusion": text}}
    return state, FakeAgentStore([row]), [requirement], reviews


def check(change, expected, *, full=False):
    state, store, reqs, reviews = fixture()
    change(state, store, reqs, reviews)
    result = reconcile_delivery(state, store, requirements=reqs, reviews=reviews, exhaustive=full)
    assert result["requirements"][0]["stage"] == expected, result
    return result


def main():
    assert check(lambda *a: None, "delivered", full=True)["status"] == "complete"
    assert check(lambda *a: None, "delivered")["status"] == "partial", _TEST_MATERIALS["text_002"]
    check(lambda s, st, r, v: st._chunks.clear(), "source_unverifiable")
    check(lambda s, st, r, v: r[0]["sources"][0].update(hash="changed"), "source_unverifiable")
    check(lambda s, st, r, v: s.reading["receipts"].clear(), "not_read")
    check(lambda s, st, r, v: r[0]["sources"][0].update(section_id="wrong-section"), "not_read")
    check(lambda s, st, r, v: s.evidence.clear(), "not_fed")
    check(
        lambda s, st, r, v: s.evidence_selection["compose"].update(provider_response_received=False),
        "input_unconfirmed",
    )
    check(lambda s, st, r, v: s.evidence_selection["delivery"].update(origin="extractive"), "generation_unavailable")
    check(lambda s, st, r, v: s.outline.update(modules=[]), "removed_by_cleaning")
    check(lambda s, st, r, v: s.verification["claims"][0].update(status="unsupported"), "citation_unsupported")
    check(lambda s, st, r, v: s.outline["modules"][0]["paragraphs"][0].update(sources=[10]), "citation_unsupported")
    check(
        lambda s, st, r, v: (
            s.evidence_selection["delivery"].update(raw_content=[]),
            s.outline.update(modules=[]),
            v["connection"].update(absent_from_generated=True),
        ),
        "not_generated",
    )
    check(lambda s, st, r, v: v.clear(), "delivery_unassessed")

    def host_step_label(s, st, r, v):
        original = v["connection"]["conclusion"]
        s.outline["modules"][0]["paragraphs"][0]["text"] = _TEST_MATERIALS["text_007"] + original
        v["connection"].update(raw_conclusion=original, conclusion=_TEST_MATERIALS["text_009"] + original)

    check(host_step_label, "delivered")
    assert (
        check(
            lambda s, st, r, v: s.evidence_selection["compose"].update(finish_reason="length"), "delivered", full=True
        )["status"]
        == "partial"
    )
    s, st, _, _ = fixture()
    assert reconcile_delivery(s, st)["status"] == "partial"
    s.verification = None
    assert reconcile_delivery(s, st)["status"] == "partial", "compose precedes verification"
    s.compose_mode = "extractive"
    assert reconcile_delivery(s, st)["status"] == "unverifiable"
    original, _, _, _ = fixture()
    s = AgentState(goal=_TEST_MATERIALS["text_008"], evidence=original.evidence)
    bad = {
        "modules": [{"heading": "Connection", "paragraphs": [{"text": _TEST_MATERIALS["text_010"], "sources": [10]}]}]
    }
    with patch("app.generate.synthesis._llm_compose", return_value=bad):
        draft = compose_synthesis(s, SimpleNamespace(answer_llm_provider="qwen", qa_max_context_chars=6000))
    assert draft.compose_mode == "refused" and not draft.modules
    assert s.evidence_selection["delivery"]["raw_paragraphs"] == 1
    assert s.evidence_selection["delivery"]["visible_paragraphs"] == 0
    assert s.evidence_selection["delivery"]["lost_paragraphs"] == 1
    assert s.evidence_selection["delivery"]["raw_content"] == bad["modules"], _TEST_MATERIALS["text_003"]
    assert s.evidence_selection["compose"]["source_contract_error"]
    print(_TEST_MATERIALS["text_005"])


if __name__ == "__main__":
    main()
