"""Source identity and specificity gates; no provider calls or fixed oracle IDs."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import copy
import hashlib
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.agent.state import AgentState, TaskSpec
from app.agent.store import ChunkRow, FakeAgentStore
from app.agent.batch_reading import eligible, read_batch, pending, revalidate
from app.agent.reading import reading_plan, read_for_synthesis, reference_rows
from app.corpus.catalog import CatalogDocument
from app.corpus.sections import section_records
from app.generate.prompt import EvidenceItem
from app.generate.synthesis import prepare_synthesis_prompt, compose_synthesis
from app.verify.claims import Claim
from app.verify.service import _judge_claim, _aggregate, reconcile_delivery
from app.api.verify import ClaimCheckOut, to_verification_out

SETTINGS = SimpleNamespace(
    agent_reading_max_batches=3,
    agent_reading_batch_chars=3000,
    agent_reading_total_chars=9000,
    agent_reading_generation_chars=6000,
    agent_reading_timeout_seconds=30,
    answer_llm_provider="stub",
    qa_max_context_chars=6000,
    verify_entail_threshold=0.7,
    verify_lexical_gate=0.3,
    verify_lexical_threshold=0.6,
)


def fixture():
    texts = [
        "# Manual\n\nIntroduction.",
        _TEST_MATERIALS["text_003"],
        _TEST_MATERIALS["text_004"],
        _TEST_MATERIALS["text_005"],
        _TEST_MATERIALS["text_006"],
        _TEST_MATERIALS["text_007"],
    ]
    rows = [
        ChunkRow(id=f"c{i}", doc_id="d", text=t, page=1, chunk_hash=f"h{i}", start_offset=i * 200)
        for i, t in enumerate(texts)
    ]
    sections = section_records("d", rows)
    target = next(s for s in sections if s["title"] == "Copy Procedure")
    proof = {
        "doc_id": "d",
        "chunk_id": "c2",
        "chunk_hash": "h2",
        "text_sha256": hashlib.sha256(rows[2].text.encode()).hexdigest(),
    }
    state = AgentState(
        goal=_TEST_MATERIALS["text_014"],
        task_kind="synthesis",
        task_spec=TaskSpec(intent="explain"),
        catalog=[CatalogDocument("d", "manual.md", [s["title"] for s in sections], section_entries=sections)],
        reference={
            "kind": "step",
            "retrieve": True,
            "chunk_ids": ["c2"],
            "source_proofs": [proof],
            "text": _TEST_MATERIALS["text_016"],
        },
    )
    return state, FakeAgentStore(rows), target


def reading_contracts():
    state, store, target = fixture()
    assert eligible(state), _TEST_MATERIALS["text_001"]
    plan = reading_plan(state, state.catalog[0], state.catalog[0].section_entries, rows=store._chunks)
    assert plan["target_id"] == target["id"]
    # Legacy catalog without section metadata still enters the tool, where
    # identity is rebuilt from currently owner/scope-filtered original rows.
    state.catalog[0].section_entries = []
    with patch("app.agent.digest._chunk_scores", return_value=[0] * 20):
        read_batch(state, store, SETTINGS)
        while pending(state):
            read_batch(state, store, SETTINGS)
    selected = {e.chunk_id for e in state.evidence}
    assert {"c1", "c2", "c3"} <= selected and "c4" not in selected
    assert state.reading["documents"]["d"]["target_basis"] == _TEST_MATERIALS["text_008"]
    prepared = prepare_synthesis_prompt(state, max_chars=6000)
    assert {"c1", "c2", "c3"} <= {e.chunk_id for e in prepared.evidence}
    checkpoint = replace(state, reading=copy.deepcopy(state.reading), evidence=[])
    revalidate(checkpoint, store)
    assert {e.chunk_id for e in checkpoint.evidence} == selected
    store._chunks = [replace(r, text=r.text + " changed") if r.id == "c2" else r for r in store._chunks]
    revalidate(checkpoint, store)
    assert checkpoint.reading["status"] == "invalid_source" and not checkpoint.evidence
    # Existing compose fallback also resolves the chapter, never anchor-only.
    state, store, _ = fixture()
    read_for_synthesis(state, store, SETTINGS)
    assert {"c1", "c2", "c3"} <= {e.chunk_id for e in state.evidence}
    # Repeated Note identities must be explicit; wrong IDs cannot fall back to
    # a title or silently pick the first occurrence.
    state, store, _ = fixture()
    state.reference["source_proofs"][0]["section_id"] = "missing"
    read_batch(state, store, SETTINGS)
    assert state.reading["status"] == "invalid_source"
    assert compose_synthesis(state, SETTINGS).compose_mode == "refused"
    state, store, _ = fixture()
    second_note = next(s for s in state.catalog[0].section_entries if s["title"] == "Note" and s["chunk_ids"] == ["c5"])
    row = store._chunks[5]
    state.reference.update(
        chunk_ids=[row.id],
        source_proofs=[
            {
                "chunk_id": row.id,
                "doc_id": "d",
                "chunk_hash": row.chunk_hash,
                "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                "section_id": second_note["id"],
            }
        ],
    )
    read_batch(state, store, SETTINGS)
    assert state.reading["documents"]["d"]["target_id"] == second_note["id"]
    # A continuation's bytes may be unchanged even when its heading occurrence
    # changes; a persisted section identity must be rechecked on resume.
    state, store, target = fixture()
    continuation = ChunkRow(
        id="continued", doc_id="d", text=_TEST_MATERIALS["text_015"], page=1, chunk_hash="hc", start_offset=450
    )
    store._chunks.append(continuation)
    state.reference.update(
        chunk_ids=[continuation.id],
        source_proofs=[
            {
                "chunk_id": continuation.id,
                "doc_id": "d",
                "chunk_hash": continuation.chunk_hash,
                "text_sha256": hashlib.sha256(continuation.text.encode()).hexdigest(),
                "section_id": target["id"],
            }
        ],
    )
    assert not reference_rows(state, store)[1]
    store._chunks = [
        replace(r, text=r.text.replace("Copy Procedure", "Different Procedure")) if r.id == "c2" else r
        for r in store._chunks
    ]
    assert _TEST_MATERIALS["text_009"] in reference_rows(state, store)[1]
    for missing in [True, False]:
        state, store, _ = fixture()
        store._chunks = (
            [r for r in store._chunks if r.id != "c2"]
            if missing
            else [replace(r, chunk_hash="new") if r.id == "c2" else r for r in store._chunks]
        )
        read_batch(state, store, SETTINGS)
        assert state.reading["status"] == "invalid_source" and not state.evidence
    state, store, _ = fixture()
    read_batch(state, store, SETTINGS, stop=lambda: True)
    assert state.reading["status"] == "cancelled" and not state.reading["receipts"]
    state, store, _ = fixture()
    read_batch(state, store, SETTINGS, deadline=0)
    assert state.reading["status"] == "timeout"
    state, store, _ = fixture()
    tiny = SimpleNamespace(**{**vars(SETTINGS), "agent_reading_max_batches": 1, "agent_reading_batch_chars": 1})
    read_batch(state, store, tiny)
    assert state.reading["status"] == "budget_exhausted" and not state.evidence


def judge(text, source):
    return _judge_claim(
        Claim(text=text, raw=text, citation_indices=[1]),
        evidence_by_index={1: EvidenceItem(index=1, chunk_id="c", doc_id="d", page=1, text=source)},
        valid_indices=[1],
        entailment_score=0.99,
        settings=SETTINGS,
    )


def verifier_contracts():
    missing = judge(_TEST_MATERIALS["text_010"], _TEST_MATERIALS["text_011"])
    assert missing.status == "unverified" and not missing.hallucination and missing.source_gaps
    assert (
        judge(
            _TEST_MATERIALS["text_017"],
            _TEST_MATERIALS["text_018"],
        ).status
        == "unverified"
    )
    assert (
        judge(
            _TEST_MATERIALS["text_019"],
            _TEST_MATERIALS["text_020"],
        ).status
        == "supported"
    )
    assert judge("Version 2.0 introduced modern_call.", "Section 2.0: modern_call copies data.").status == "unverified"
    assert judge("Version 2.0 introduced modern_call.", "Version 2.0 introduced modern_call.").status == "supported"
    assert judge(_TEST_MATERIALS["text_021"], _TEST_MATERIALS["text_022"]).status == "unverified"
    assert judge(_TEST_MATERIALS["text_023"], _TEST_MATERIALS["text_024"]).status == "supported"
    assert judge(_TEST_MATERIALS["text_025"], _TEST_MATERIALS["text_026"]).status == "unverified"
    assert judge(_TEST_MATERIALS["text_027"], _TEST_MATERIALS["text_028"]).status == "unverified"
    assert judge(_TEST_MATERIALS["text_029"], _TEST_MATERIALS["text_030"]).status == "supported"
    assert judge(_TEST_MATERIALS["text_031"], "Copy files safely.").status == "unverified"
    assert judge(_TEST_MATERIALS["text_032"], "doBackup copies files.").status == "supported"
    assert judge(_TEST_MATERIALS["text_033"], "Call modern_call.").status == "unverified"
    assert judge(_TEST_MATERIALS["text_034"], "Version 1.0 supports modern_call.").status == "unverified"
    assert judge(_TEST_MATERIALS["text_035"], _TEST_MATERIALS["text_036"]).status == "supported"
    assert judge(_TEST_MATERIALS["text_037"], "Step 1 calls modern_call.").status == "supported"
    assert judge(_TEST_MATERIALS["text_038"], _TEST_MATERIALS["text_039"]).status == "supported"
    assert judge(_TEST_MATERIALS["text_040"], _TEST_MATERIALS["text_041"]).status == "supported"
    assert judge(_TEST_MATERIALS["text_042"], "Step 1 runs modern_call.").status == "supported"
    result = _aggregate([missing])
    out = to_verification_out(result)
    assert out.unverified_count == 1 and out.supported_count == 0 and out.hallucination_count == 0
    assert out.claims[0].source_gaps
    old = {k: v for k, v in asdict(missing).items() if k != "source_gaps"}
    assert ClaimCheckOut(**old).source_gaps == [], _TEST_MATERIALS["text_002"]
    # Even an exhaustive set of requested facts cannot conceal an additional
    # unverified statement in the visible answer.
    state, store, _ = fixture()
    state.compose_mode = "written"
    state.verification = {"claims": [asdict(missing)]}
    state.evidence_selection = {"compose": {"provider": "qwen", "provider_response_received": True}}
    state.outline = {"modules": [{"paragraphs": [{"text": _TEST_MATERIALS["text_043"], "sources": [1]}]}]}
    state.evidence = [EvidenceItem(index=1, chunk_id="c2", doc_id="d", page=1, text=store._chunks[2].text)]
    contract = reconcile_delivery(state, store, requirements=[], exhaustive=True)
    assert contract["status"] != "complete" and contract["unverified_claims"]
    print(_TEST_MATERIALS["text_012"])


if __name__ == "__main__":
    reading_contracts()
    verifier_contracts()
    print(_TEST_MATERIALS["text_013"])
