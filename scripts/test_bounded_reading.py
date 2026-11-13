"""Controlled read/delivery contracts. No paid calls, even when replaying Qwen."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.agent.state import AgentState, TaskSpec, SynthesisNode
from app.agent.store import FakeAgentStore, ChunkRow
from app.agent.batch_reading import eligible, read_batch, pending, revalidate, record_delivery
from app.agent.run_budget import bind, reserve, settle
from app.corpus.catalog import CatalogDocument
from app.corpus.sections import section_records, norm_title
from app.generate.prompt import EvidenceItem
from app.generate.synthesis import compose_synthesis, prepare_synthesis_prompt, finalize_draft, generation_chars
from app.ingest.parser import parse_document
from app.ingest.chunker import chunk_pages
from app.llm.base import LLMError, LLMUsage, LLMResponse
from app.llm.meter import MeteredClient

AUDIT = ROOT / _TEST_MATERIALS["text_001"]
OUT = Path(os.environ.get("ACCEPTANCE_OUTPUT", tempfile.gettempdir()))
SETTINGS = SimpleNamespace(
    answer_llm_provider="stub",
    agent_reading_max_batches=3,
    agent_reading_batch_chars=6000,
    agent_reading_total_chars=18000,
    agent_reading_generation_chars=12000,
    agent_reading_timeout_seconds=30,
    qa_max_context_chars=6000,
)


def fixture(path, goal, doc_id):
    drafts = chunk_pages(parse_document(path.name, path.read_bytes()), filename=path.name)
    rows = [
        ChunkRow(
            id=f"{doc_id}-{n}",
            doc_id=doc_id,
            page=d.page,
            text=d.text,
            chunk_hash=d.chunk_hash,
            start_offset=d.start_offset,
            section_path=d.section_path,
        )
        for n, d in enumerate(drafts)
    ]
    sections = section_records(doc_id, rows)
    doc = CatalogDocument(doc_id, path.name, [s["title"] for s in sections], section_entries=sections)
    state = AgentState(
        goal=goal,
        task_kind="synthesis",
        catalog=[doc],
        task_spec=TaskSpec(intent="plan", subjects=[goal]),
        nodes=[SynthesisNode(id="n", title=goal, query=goal, status="retrieved")],
    )
    assert eligible(state)
    return state, rows


def run_fixture(path, goal, gold, doc_id):
    state, rows = fixture(path, goal, doc_id)
    store = FakeAgentStore(rows)
    read_batch(state, store, SETTINGS)
    assert all(g["scope_chunk_ids"] for d in state.reading["documents"].values() for g in d["goals"].values())
    first = set(state.reading["documents"][doc_id]["read_chunk_ids"])
    checkpoint = json.loads(json.dumps(state.reading))
    restored = replace(state, reading=checkpoint, evidence=[])
    revalidate(restored, store)
    while pending(restored):
        read_batch(restored, store, SETTINGS)
    state = restored
    assert len(state.reading["batches"]) <= 3
    ids = [cid for b in state.reading["batches"] for cid in b["chunk_ids"]]
    assert len(ids) == len(set(ids)), _TEST_MATERIALS["text_002"]
    assert first.issubset(ids)
    assert state.reading["budget"]["used_chars"] <= 18000
    for receipt in state.reading["receipts"]:
        assert receipt["hash"] and receipt["section_ids"] and receipt["text_sha256"] and receipt["status"] == "read"
    revalidate(state, store)
    prepared = prepare_synthesis_prompt(state, max_chars=generation_chars(state, SETTINGS))
    read_ids = {e.chunk_id for e in state.evidence}
    state.evidence = prepared.evidence
    assert prepared.selection["evidence_chars"] <= 12000
    # Simulate complete omission by the generator. Reading is unchanged, answer
    # delivery must not turn green merely because all evidence was supplied.
    record_delivery(state, [])
    assert all(
        not g["answer_cited_chunk_ids"] for d in state.reading["documents"].values() for g in d["goals"].values()
    )
    assert state.reading["quality"]["answer_semantic_coverage"] == "not_assessed"
    points = []
    for name, quote in gold.items():
        source = {r.id for r in rows if norm_title(quote) in norm_title(r.text)}
        assert source, (name, _TEST_MATERIALS["text_009"])
        points.append(
            {
                "key": name,
                "source_chunk_ids": sorted(source),
                "read": bool(source & read_ids),
                "prepared_input": bool(source & {e.chunk_id for e in state.evidence}),
                "new_real_model_input": None,
                "new_real_answer": None,
                "reason": _TEST_MATERIALS["text_025"],
            }
        )
    result = {
        "goal": goal,
        "sections": state.catalog[0].section_entries,
        "reading": state.reading,
        "prepared_evidence": [vars(e) for e in state.evidence],
        "gold": points,
        "new_provider_calls": 0,
    }
    assert all(p["read"] and p["prepared_input"] for p in points), points
    # Digest follows the same source contracts without an intermediate LLM.
    digest = replace(state, task_kind="digest", nodes=[], digests=[], evidence=list(state.evidence))
    from app.agent.tools import ComposeTool, ToolContext
    from app.agent.planner import Plan

    composed = ComposeTool().execute(Plan("test", "compose"), ToolContext(digest, SETTINGS, store))
    assert digest.compose_mode == "extractive" and digest.outline["modules"] and not composed.terminal
    (OUT / (doc_id + "-coverage.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(
        doc_id,
        "chunks",
        len(rows),
        "read",
        len(read_ids),
        "input",
        len(state.evidence),
        "gold-read",
        sum(p["read"] for p in points),
        "gold-input",
        sum(p["prepared_input"] for p in points),
    )
    # Deleted and changed rows cannot remain in the writer whitelist on resume.
    victim = state.reading["receipts"][0]["chunk_id"]
    store._chunks = [replace(r, text=r.text + " changed") if r.id == victim else r for r in store._chunks]
    revalidate(state, store)
    assert victim not in {e.chunk_id for e in state.evidence}
    assert next(x for x in state.reading["receipts"] if x["chunk_id"] == victim)["status"] == "invalid"
    return result


def contracts():
    # A high CE score for a different operation's Note must not displace the
    # prerequisite Note attached to the target's own parent section.
    parts = [
        "# Manual\n\nOverview.",
        _TEST_MATERIALS["text_004"],
        _TEST_MATERIALS["text_005"],
        "## Recovery\n\nRecover files.",
        _TEST_MATERIALS["text_006"],
    ]
    note_rows = [
        ChunkRow(id=f"n{i}", doc_id="notes", page=1, text=text, chunk_hash=f"h{i}", start_offset=i * 100)
        for i, text in enumerate(parts)
    ]
    note_sections = section_records("notes", note_rows)
    note_state = AgentState(
        goal=_TEST_MATERIALS["text_010"],
        task_kind="synthesis",
        task_spec=TaskSpec(intent="plan"),
        catalog=[
            CatalogDocument("notes", "notes.md", [s["title"] for s in note_sections], section_entries=note_sections)
        ],
    )
    caps = SimpleNamespace(**{**vars(SETTINGS), "agent_reading_max_batches": 1, "agent_reading_batch_chars": 220})
    with patch(
        "app.agent.digest._chunk_scores",
        side_effect=lambda goal, texts, settings: [100 if "unrelated" in t else 0 for t in texts],
    ):
        read_batch(note_state, FakeAgentStore(note_rows), caps)
    assert "n1" in {r["chunk_id"] for r in note_state.reading["receipts"]}
    assert "n4" not in {r["chunk_id"] for r in note_state.reading["receipts"]}
    # CI-safe source fixture independent of the ignored audit documents.
    rows = [
        ChunkRow(
            id=str(i),
            doc_id="d",
            page=1,
            text=f"## Procedure\n\nStep {i} must keep file {i}. " + "x " * 100,
            chunk_hash=f"h{i}",
            start_offset=i * 300,
        )
        for i in range(8)
    ]
    sections = section_records("d", rows)
    state = AgentState(
        goal=_TEST_MATERIALS["text_011"],
        task_kind="synthesis",
        task_spec=TaskSpec(intent="plan"),
        catalog=[CatalogDocument("d", "d.md", [s["title"] for s in sections], section_entries=sections)],
    )
    tiny = SimpleNamespace(**{**vars(SETTINGS), "agent_reading_max_batches": 1, "agent_reading_batch_chars": 500})
    store = FakeAgentStore(rows)
    read_batch(state, store, tiny)
    assert state.reading["status"] == "budget_exhausted" and state.reading["documents"]["d"]["unread_chunk_ids"]
    cancelled = replace(state, reading={}, evidence=[])
    read_batch(cancelled, store, SETTINGS, stop=lambda: True)
    assert cancelled.reading["status"] == "cancelled" and not cancelled.reading["receipts"]
    timed = replace(state, reading={}, evidence=[])
    read_batch(timed, store, SETTINGS, deadline=0)
    assert timed.reading["status"] == "timeout"
    item = EvidenceItem(index=1, chunk_id="c", doc_id="d", page=1, text="a" * 1000)
    too_big = AgentState(
        goal=_TEST_MATERIALS["text_012"],
        evidence=[item],
        reading={"mode": "bounded", "budget": {"generation_chars": 10}},
    )
    assert not prepare_synthesis_prompt(too_big, max_chars=10).evidence
    # Output can be structurally malformed, but never silently disappear.
    state = AgentState(goal=_TEST_MATERIALS["text_013"], evidence=[item])
    bad = {"modules": [{"heading": "bad", "paragraphs": "not an array"}]}
    draft = finalize_draft(bad, state)
    assert not draft.modules and any("paragraphs" in g["reason"] for g in draft.gaps)


def budget_contracts():
    from app.llm.qwen import QwenClient

    requests = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=_TEST_MATERIALS["text_038"]), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    client = object.__new__(QwenClient)
    client._model = "mock-qwen"
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: requests.append(kw) or response))
    )
    assert (
        client.generate(
            system="s", prompt=_TEST_MATERIALS["text_029"], max_output_tokens=99, total_output_cap=60, temperature=0
        ).text
        == "ok"
    )
    assert requests[0]["max_completion_tokens"] == 60 and "max_tokens" not in requests[0]
    client.generate(
        system="s",
        prompt=_TEST_MATERIALS["text_014"],
        max_output_tokens=99,
        total_output_cap=60,
        request_timeout=240,
        temperature=0,
    )
    assert requests[1]["timeout"] == 240
    client.generate(
        system="s",
        prompt=_TEST_MATERIALS["text_015"],
        max_output_tokens=99,
        total_output_cap=60,
        thinking_budget=20,
        temperature=0,
    )
    assert requests[2]["extra_body"] == {"thinking_budget": 20}
    truncated = AgentState(
        goal=_TEST_MATERIALS["text_016"],
        evidence=[EvidenceItem(index=1, chunk_id="c", doc_id="d", page=1, text=_TEST_MATERIALS["text_033"])],
    )
    raw = {"modules": [{"heading": "answer", "paragraphs": [{"text": _TEST_MATERIALS["text_034"], "sources": [1]}]}]}
    with patch(
        "app.llm.get_llm_client",
        return_value=SimpleNamespace(generate=lambda **kw: LLMResponse(text=json.dumps(raw), finish_reason="length")),
    ):
        draft = compose_synthesis(
            truncated,
            SimpleNamespace(answer_llm_provider="qwen", qa_max_context_chars=6000, llm_max_output_tokens=1200),
        )
    assert truncated.partial and draft.modules and any(g["topic"] == _TEST_MATERIALS["text_030"] for g in draft.gaps)
    state = AgentState(goal=_TEST_MATERIALS["text_017"], task_kind="synthesis")
    settings = SimpleNamespace(
        agent_longform_output_tokens=100,
        agent_longform_final_output_reserve=60,
        agent_longform_input_tokens=10000,
        agent_longform_max_model_calls=3,
    )
    saved = []
    with bind(state, settings, checkpoint=lambda: saved.append(copy.deepcopy(state.run_budget)) or True):
        first = reserve("s", "p", 80, "analyze", "qwen")
        assert first["wire_output_cap"] == 28
        settle(first, status="failed")
        final = reserve("s", "p", 1, "compose", "qwen")
        assert final["wire_output_cap"] == 48
        settle(final, LLMUsage(prompt_tokens=10, completion_tokens=20), status="success")
    assert saved[0]["calls"][0]["status"] == "reserved"
    assert state.run_budget["calls"][0]["output_reserved"] == 40, _TEST_MATERIALS["text_003"]
    assert "unpriced" in state.run_budget["currency_status"]
    denied = AgentState(goal=_TEST_MATERIALS["text_018"], task_kind="synthesis")
    with bind(denied, settings, checkpoint=lambda: False):
        try:
            reserve("s", "p", 10, "compose", "qwen")
            raise AssertionError("lost lease accepted")
        except LLMError as e:
            assert e.kind == "cancelled"
    attempts = []

    class Boom:
        def generate(self, **kw):
            attempts.append(kw)
            raise LLMError("timeout", kind="timeout")

    state = AgentState(goal=_TEST_MATERIALS["text_019"], task_kind="synthesis")
    with (
        bind(
            state,
            SimpleNamespace(
                agent_longform_max_model_calls=2,
                agent_longform_output_tokens=300,
                agent_longform_final_output_reserve=80,
            ),
        ),
        patch("app.llm.meter._retry_policy", return_value=(5, 0, 0)),
    ):
        try:
            MeteredClient(Boom(), provider="qwen").generate(
                system="s", prompt=_TEST_MATERIALS["text_031"], max_output_tokens=1, temperature=0, purpose="compose"
            )
        except LLMError as e:
            assert e.kind == "budget"
    assert len(attempts) == 2 and len(state.run_budget["calls"]) == 2
    assert all(c["status"] == "failed" for c in state.run_budget["calls"])
    assert all(k["total_output_cap"] == 80 for k in attempts)
    assert all(k["request_timeout"] == 240 for k in attempts)
    assert all(k["thinking_budget"] <= k["total_output_cap"] // 3 for k in attempts)


def replay():
    paid = AUDIT / _TEST_MATERIALS["text_007"]
    if not paid.exists():
        print(_TEST_MATERIALS["text_020"])
        return
    outputs = []
    for n, call in [(2, 4), (3, 6), (4, 7), (6, 10)]:
        saved = json.loads((paid / f"turn-{n}.json").read_text(encoding="utf8"))["result"]
        o = saved["outline"]
        items = [
            EvidenceItem(**{k: v for k, v in e.items() if k in EvidenceItem.__dataclass_fields__})
            for e in o["evidence"]
        ]
        sources = []
        for d in o.get("reading", {}).get("documents", {}).values():
            sources.extend(
                {"doc_id": d["doc_id"], "filename": d["filename"], "section": s["title"]}
                for s in d["required_sections"]
            )
        catalog = [
            CatalogDocument(items[0].doc_id, d["filename"], d["sections"]) for d in saved.get("catalog", []) if items
        ]
        state = AgentState(
            goal=saved["goal"],
            task_kind="synthesis",
            task_spec=TaskSpec(intent=saved["task_spec"]["intent"]),
            evidence=items,
            catalog=catalog,
            reading=copy.deepcopy(o.get("reading") or {}),
            nodes=[
                SynthesisNode(
                    id="r",
                    title=_TEST_MATERIALS["text_035"],
                    query=_TEST_MATERIALS["text_036"],
                    source_sections=sources,
                )
            ],
            partial=bool(o.get("reading")),
        )
        raw = json.loads(json.loads((paid / f"call-{call:02d}.json").read_text(encoding="utf8"))["response"])
        with patch("app.generate.synthesis._llm_compose", return_value=raw):
            draft = compose_synthesis(state, SimpleNamespace(answer_llm_provider="qwen", qa_max_context_chars=6000))
        diagnostic = state.evidence_selection["delivery"]
        assert diagnostic["raw_modules"] == diagnostic["visible_modules"], (n, diagnostic)
        assert diagnostic["raw_paragraphs"] == diagnostic["visible_paragraphs"], (n, diagnostic)
        assert draft.compose_mode == "written" and draft.markdown.strip()
        if n == 4:
            assert (
                len(draft.modules) == 3 and "pg_wal" in draft.markdown and _TEST_MATERIALS["text_032"] in draft.markdown
            )
        outputs.append(
            {
                "turn": n,
                "call": call,
                "goal": saved["goal"],
                "provider": "recorded_response_replay",
                "new_provider_calls": 0,
                "old_user_answer": saved["answer"],
                "new_user_answer": draft.markdown,
                "modules": draft.modules,
                "evidence": [vars(e) for e in state.evidence],
                "delivery": diagnostic,
                "task_spec": saved["task_spec"],
                "operation_anchor": o.get("operation_anchor"),
                "reference": o.get("reference"),
            }
        )
    (OUT / _TEST_MATERIALS["text_026"]).write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf8")


if __name__ == "__main__":
    contracts()
    budget_contracts()
    replay()
    source = AUDIT / "acceptance-be"
    if (source / "fixtures/postgresql16-backup.pdf").exists():
        gold = json.loads((source / _TEST_MATERIALS["text_037"]).read_text(encoding="utf8"))["gold"]
        run_fixture(
            source / "fixtures/postgresql16-backup.pdf",
            _TEST_MATERIALS["text_021"],
            gold,
            "postgresql",
        )
    else:
        print(_TEST_MATERIALS["text_022"])
    control = AUDIT / "reading-round2"
    if (control / _TEST_MATERIALS["text_027"]).exists():
        gold = json.loads((control / "sources/manifest.json").read_text(encoding="utf8"))["gold"]
        run_fixture(
            control / _TEST_MATERIALS["text_028"],
            _TEST_MATERIALS["text_023"],
            gold,
            "sqlite",
        )
    else:
        print(_TEST_MATERIALS["text_024"])
    print(_TEST_MATERIALS["text_008"])
