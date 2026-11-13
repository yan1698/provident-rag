"""Deterministic B+E regressions; real saved traces, no external model calls."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.agent.analyze import apply_analyze
from app.agent.memory import compose_session_memory
from app.agent.state import AgentState, SynthesisNode, TaskSpec
from app.generate.prompt import EvidenceItem
from app.generate.synthesis import apply_g3, compose_synthesis, reindex_evidence_state, finalize_draft
from app.ingest.parser import parse_document, _lines_from_dict
from app.ingest.chunker import chunk_pages
from app.agent.store import ChunkRow
from app.corpus.sections import section_records, norm_title, required_sections, section_chunk_ids
from app.corpus.catalog import CatalogDocument
from app.agent.store import FakeAgentStore
from app.agent.reading import select_reading, mark_read, finalize_reading, reference_rows
from app.agent.tools import _stamp_paragraphs, ComposeTool, ToolContext
from app.agent.planner import Plan
from dataclasses import replace

BE = ROOT / _TEST_MATERIALS["text_008"]


def evidence_contract():
    fixture = BE / _TEST_MATERIALS["text_009"]
    if fixture.exists():
        trace = json.loads(fixture.read_text(encoding="utf-8"))["result"]
        items = [
            EvidenceItem(**{k: v for k, v in item.items() if k in EvidenceItem.__dataclass_fields__})
            for item in trace["outline"]["evidence"]
        ]
        source = trace["outline"]["modules"][0]["source_sections"][0]
    else:
        trace = {"goal": _TEST_MATERIALS["text_019"]}
        items = [EvidenceItem(index=i, chunk_id=f"c{i}", doc_id="manual", page=1, text="x" * 600) for i in range(1, 13)]
        source = {"doc_id": "manual", "filename": _TEST_MATERIALS["text_020"], "section": "operation"}
    state = AgentState(
        goal=trace["goal"],
        task_kind="synthesis",
        evidence=items,
        task_spec=TaskSpec(),
        nodes=[
            SynthesisNode(
                id="one",
                title=_TEST_MATERIALS["text_057"],
                query=_TEST_MATERIALS["text_058"],
                status="retrieved",
                source_sections=[source],
            )
        ],
    )
    captured = []
    state.digests = [
        {"filename": _TEST_MATERIALS["text_021"], "bullets": [{"text": _TEST_MATERIALS["text_047"], "sources": [10]}]}
    ]

    def generate(**kw):
        captured.append(kw["prompt"])
        return SimpleNamespace(
            text=json.dumps(
                {
                    "modules": [
                        {
                            "heading": "probe",
                            "source_sections": [source],
                            "paragraphs": [{"text": _TEST_MATERIALS["text_077"], "sources": [1, 10]}],
                        }
                    ]
                }
            )
        )

    with patch("app.llm.get_llm_client", return_value=SimpleNamespace(generate=generate)):
        draft = compose_synthesis(state, SimpleNamespace(answer_llm_provider="qwen", qa_max_context_chars=6000))
    sent = state.evidence_selection["compose"]["sent_indices"]
    assert sent == list(range(1, 10)), sent
    assert [e.index for e in state.evidence] == sent
    assert draft.compose_mode == "refused" and "Misleading" not in draft.markdown
    assert len(captured) == 1 and '<evidence id="S10"' not in captured[0]
    assert "Unfed digest summary" not in captured[0]
    blocked = AgentState(
        goal=trace["goal"],
        task_kind="synthesis",
        evidence=list(items),
        task_spec=TaskSpec(),
        nodes=[
            SynthesisNode(
                id="one",
                title=_TEST_MATERIALS["text_059"],
                query=_TEST_MATERIALS["text_060"],
                status="retrieved",
                source_sections=[source],
            )
        ],
    )
    with patch("app.llm.get_llm_client", return_value=SimpleNamespace(generate=generate)):
        result = ComposeTool().execute(
            Plan("test", "compose", "", "rules"),
            ToolContext(
                blocked,
                SimpleNamespace(answer_llm_provider="qwen", qa_max_context_chars=6000, agent_synthesis_max_evidence=16),
                FakeAgentStore(),
            ),
        )
    assert (
        result.terminal
        and result.observation["status"] == "refused"
        and _TEST_MATERIALS["text_022"] in blocked.answer
        and "[S10]" not in blocked.answer
    )
    assert len(state.evidence_selection["compose"]["candidate_evidence"]) == 12
    kept, _ = apply_g3(
        [
            {
                "heading": "probe",
                "source_sections": [source],
                "paragraphs": [{"text": _TEST_MATERIALS["text_061"], "sources": [1, 10]}],
            }
        ],
        [source],
        valid_indices=set(sent),
    )
    assert not kept
    remap = AgentState(
        goal=_TEST_MATERIALS["text_023"],
        evidence=[replace(items[0], index=3), replace(items[1], index=7)],
        digests=[{"bullets": [{"text": _TEST_MATERIALS["text_071"], "sources": [7]}]}],
    )
    reindex_evidence_state(remap)
    assert remap.digests[0]["bullets"][0]["sources"] == [2]
    assert "[S7]" not in remap.digests[0]["bullets"][0]["text"]
    from app.agent.digest import _read_document

    with (
        patch("app.llm.router_llm_available", return_value=True),
        patch(
            "app.agent.digest._llm_digest",
            return_value={"bullets": [{"text": _TEST_MATERIALS["text_072"], "sources": [1, 10]}]},
        ),
    ):
        bullets, _, mode, note = _read_document(
            "safe source", [replace(items[0], text=_TEST_MATERIALS["text_062"])], SimpleNamespace()
        )
    assert (
        mode == "extractive"
        and _TEST_MATERIALS["text_024"] in note
        and all("Injected" not in b["text"] for b in bullets)
    )


def task_transitions():
    fixture = BE / _TEST_MATERIALS["text_010"]
    turns = (
        json.loads(fixture.read_text(encoding="utf-8"))["turns"][:6]
        if fixture.exists()
        else [
            {"goal": goal}
            for goal in [
                _TEST_MATERIALS["text_048"],
                _TEST_MATERIALS["text_049"],
                _TEST_MATERIALS["text_050"],
                _TEST_MATERIALS["text_051"],
                _TEST_MATERIALS["text_052"],
                _TEST_MATERIALS["text_053"],
            ]
        ]
    )
    history = []
    for index, turn in enumerate(turns):
        state = AgentState(goal=turn["goal"], task_kind="synthesis", memory_brief=compose_session_memory(history))
        apply_analyze(state)
        if index == 1:
            assert state.task_spec.update == "new"
            assert "LowLevelAPI" in "".join(state.task_spec.subjects)
        if index in {1, 2, 3, 5}:
            assert all(
                _TEST_MATERIALS["text_073"] not in q and _TEST_MATERIALS["text_074"] not in q
                for q in state.task_spec.questions
            )
        if index in {3, 4}:
            assert state.task_spec.update == "clarify"  # no fabricated operation anchor
        history.append({"goal": state.goal, "answer": "", "outline": {"task_spec": state.task_spec.to_dict()}})
    state = AgentState(
        goal=_TEST_MATERIALS["text_025"], task_kind="synthesis", memory_brief=compose_session_memory(history)
    )
    apply_analyze(state)
    assert state.task_spec.update == "new" and "Redis" in " ".join(state.task_spec.questions)
    assert not any("PostgreSQL" in q for q in state.task_spec.questions)
    switched = AgentState(
        goal=_TEST_MATERIALS["text_026"], task_kind="synthesis", memory_brief=compose_session_memory(history)
    )
    apply_analyze(switched)
    assert switched.task_spec.update == "new" and any("Redis" in q for q in switched.task_spec.questions)


def chapter_contracts():
    if not (BE / "fixtures/postgresql16-backup.pdf").exists():
        print(_TEST_MATERIALS["text_027"])
        return
    manifest = json.loads((BE / "sources/manifest.json").read_text(encoding="utf-8"))
    pages = parse_document("manual.pdf", (BE / "fixtures/postgresql16-backup.pdf").read_bytes())
    drafts = chunk_pages(pages, filename=_TEST_MATERIALS["text_028"])
    rows = [
        ChunkRow(
            id=str(i),
            doc_id="manual",
            page=c.page,
            text=c.text,
            chunk_hash=c.chunk_hash,
            start_offset=c.start_offset,
            section_path=c.section_path,
        )
        for i, c in enumerate(drafts)
    ]
    sections = section_records("manual", rows)
    missing = [
        t["title"]
        for t in manifest["toc"]
        if not any(norm_title(t["title"]) == norm_title(s["title"]) for s in sections)
    ]
    assert not missing, missing
    connection = next(r for r in rows if "connection calling pg_backup_start" in r.text)
    assert " / Note / " not in connection.section_path and "26.3.3. Making" in connection.section_path
    notes = [s for s in sections if s["title"] == "Note"]
    assert len(notes) == 2 and notes[0]["id"] != notes[1]["id"], notes
    assert (notes[0]["path"], notes[0]["label"]) != (notes[1]["path"], notes[1]["label"])
    plan = required_sections(_TEST_MATERIALS["text_011"], sections)
    assert plan["target_id"] and not plan["missing_categories"], plan
    selected = set(section_chunk_ids(sections, plan["target_id"]))
    assert selected and any("pg_backup_start" in r.text for r in rows if r.id in selected)
    assert any(_TEST_MATERIALS["text_044"] in s["title"] for s in plan["required"])
    assert any(s["reason"] == "risks" for s in plan["required"])
    doc = CatalogDocument("manual", "manual.pdf", [s["title"] for s in sections], section_entries=sections)
    state = AgentState(goal=_TEST_MATERIALS["text_029"], task_kind="digest", catalog=[doc])
    selected = select_reading(state, doc, rows, 6000, SimpleNamespace())
    state.evidence = [
        EvidenceItem(index=i + 1, chunk_id=r.id, doc_id=r.doc_id, page=r.page, text=r.text, chunk_hash=r.chunk_hash)
        for i, r in enumerate(selected)
    ]
    mark_read(state, doc.doc_id, state.evidence)
    finalize_reading(state)
    assert state.partial and state.gaps and state.reading["documents"]["manual"]["unread_chunk_ids"]
    gold = json.loads((BE / _TEST_MATERIALS["text_063"]).read_text(encoding="utf-8"))["gold"]
    record = state.reading["documents"]["manual"]
    coverage = []
    for key, quote in gold.items():
        ids = {r.id for r in rows if norm_title(quote) in norm_title(r.text)}
        assert ids, (key, quote)
        coverage.append(
            {
                "key": key,
                "candidate": True,
                "required": bool(ids.intersection(record["required_chunk_ids"])),
                "read": bool(ids.intersection(record["read_chunk_ids"])),
                "prompt": bool(ids.intersection(record["prompt_chunk_ids"])),
                "answer": None,
                "uncovered": "" if ids.intersection(record["prompt_chunk_ids"]) else _TEST_MATERIALS["text_054"],
            }
        )
    generous = AgentState(goal=state.goal, task_kind="digest", catalog=[doc])
    all_required = select_reading(generous, doc, rows, 1000000, SimpleNamespace())
    assert all(any(norm_title(q) in norm_title(r.text) for r in all_required) for q in gold.values())
    conditions = AgentState(
        goal=_TEST_MATERIALS["text_030"],
        task_kind="digest",
        catalog=[doc],
        task_spec=TaskSpec(subjects=[_TEST_MATERIALS["text_064"]]),
    )
    condition_rows = select_reading(conditions, doc, rows, 6000, SimpleNamespace())
    assert {r.id for r in condition_rows} != {r.id for r in selected}
    condition_points = [k for k, q in gold.items() if any(norm_title(q) in norm_title(r.text) for r in condition_rows)]
    assert "G12_archive_config" in condition_points and "G13_tablespace_risk" in condition_points, condition_points
    anchor = next(r for r in rows if norm_title(gold["G04_filesystem_step"]) in norm_title(r.text))
    followup = AgentState(
        goal=_TEST_MATERIALS["text_031"],
        task_kind="synthesis",
        task_spec=TaskSpec(subjects=[_TEST_MATERIALS["text_065"]]),
        reference={"retrieve": True, "chunk_ids": [anchor.id]},
        catalog=[doc],
    )
    from app.generate.fence import format_evidence_block

    follow_rows = select_reading(
        followup,
        doc,
        rows,
        6000 - len(format_evidence_block(anchor.text, index=1, page=anchor.page)) - 1,
        SimpleNamespace(),
    )
    follow_points = [
        k for k, q in gold.items() if any(norm_title(q) in norm_title(r.text) for r in [anchor] + follow_rows)
    ]
    assert all(any(k.startswith(prefix) for k in follow_points) for prefix in ["G09", "G10", "G14"]), follow_points
    output = Path(os.environ["ACCEPTANCE_OUTPUT"]) if os.environ.get("ACCEPTANCE_OUTPUT") else None
    if output and output.is_dir():
        (output / "deterministic-reading.json").write_text(
            json.dumps(
                {
                    "sections": sections,
                    "reading": record,
                    "gold": coverage,
                    "condition_followup": {"covered_points": condition_points, "reading": conditions.reading},
                    "file_followup": {"covered_points": follow_points, "reading": followup.reading},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf8",
        )

    def span(text):
        return {"text": text, "size": 14, "flags": 16}

    wrapped = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [span(_TEST_MATERIALS["text_075"])]},
                    {"spans": [span("Recovery (PITR)")]},
                ],
            },
            {"type": 0, "lines": [{"spans": [{"text": _TEST_MATERIALS["text_076"], "size": 10, "flags": 0}]}]},
        ]
    }
    assert _lines_from_dict(wrapped)[0] == _TEST_MATERIALS["text_012"]
    print(f"{_TEST_MATERIALS['text_001']}{len(rows)}{_TEST_MATERIALS['text_002']}")


def step_contracts():
    items = [
        EvidenceItem(
            index=i,
            chunk_id=f"c{i}",
            doc_id="d",
            page=1,
            text=f"{_TEST_MATERIALS['text_003']}{i}{_TEST_MATERIALS['text_004']}",
            chunk_hash=f"hash{i}",
        )
        for i in range(1, 6)
    ]
    modules = [
        {
            "heading": "operation",
            "paragraphs": [{"text": _TEST_MATERIALS["text_066"], "sources": [1]}]
            + [
                {
                    "text": f"{_TEST_MATERIALS['text_005']}{i}{_TEST_MATERIALS['text_006']}{i}{_TEST_MATERIALS['text_007']}",
                    "sources": [i],
                    "step_number": i,
                }
                for i in range(1, 6)
            ],
        }
    ]
    state = AgentState(
        goal=_TEST_MATERIALS["text_032"],
        evidence=items,
        answer="\n\n".join(p["text"] + f" [S{p['sources'][0]}]" for p in modules[0]["paragraphs"]),
        outline={"modules": modules},
        compose_mode="written",
    )
    _stamp_paragraphs(state)
    assert len(state.outline["steps"]) == 5
    state.operation_anchor = "original-operation"
    state.operation_steps = state.outline["steps"]
    state.task_spec = TaskSpec(intent="explain", update="continue")
    _stamp_paragraphs(state)
    assert state.outline["operation_anchor"] == {"run_id": "original-operation"}
    from app.agent.memory import resolve_answer_reference

    assert (
        resolve_answer_reference(_TEST_MATERIALS["text_033"], answer=state.answer, steps=state.outline["steps"]) is None
    )
    seed = {**state.outline, "source_answer": state.answer, "operation_anchor": "origin"}
    step = AgentState(goal=_TEST_MATERIALS["text_034"], memory_brief=_TEST_MATERIALS["text_035"], outline=seed.copy())
    apply_analyze(step)
    assert step.reference["kind"] == "step" and step.reference["chunk_ids"] == ["c3"] and not step.skips_retrieve()
    paragraph = AgentState(
        goal=_TEST_MATERIALS["text_036"], memory_brief=_TEST_MATERIALS["text_037"], outline=seed.copy()
    )
    apply_analyze(paragraph)
    assert (
        paragraph.reference["kind"] == "paragraph"
        and paragraph.reference["chunk_ids"] == ["c2"]
        and not paragraph.skips_retrieve()
    )
    store = FakeAgentStore(
        [ChunkRow(id=i.chunk_id, doc_id=i.doc_id, page=1, text=i.text, chunk_hash=i.chunk_hash) for i in items]
    )
    assert reference_rows(step, store)[0][0].id == "c3"
    store._chunks = [
        replace(r, text=_TEST_MATERIALS["text_055"], chunk_hash="hash3") if r.id == "c3" else r for r in store._chunks
    ]
    assert reference_rows(step, store)[1]
    store._chunks = [r for r in store._chunks if r.id != "c3"]
    assert reference_rows(step, store)[1]
    ambiguous = AgentState(
        goal=_TEST_MATERIALS["text_038"], memory_brief=_TEST_MATERIALS["text_039"], outline=seed.copy()
    )
    apply_analyze(ambiguous)
    assert (
        ambiguous.skips_retrieve()
        and ambiguous.reference["kind"] == "condition"
        and _TEST_MATERIALS["text_040"] in ambiguous.reference["message"]
    )
    print(_TEST_MATERIALS["text_013"])


def real_gap_replay():
    path = ROOT / _TEST_MATERIALS["text_014"]
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf8"))["result"]
        raw_gaps = saved["gaps"]
        reading = saved["outline"]["reading"]
    else:
        raw_gaps = [{"topic": "conditions", "reason": _TEST_MATERIALS["text_045"]}]
        reading = {"documents": {"d": {"complete": False}}}
    state = AgentState(goal=_TEST_MATERIALS["text_041"], reading=reading)
    draft = finalize_draft({"modules": [], "gaps": raw_gaps}, state)
    assert not any(_TEST_MATERIALS["text_056"] in g["reason"] for g in draft.gaps)
    assert any(_TEST_MATERIALS["text_046"] in g["reason"] for g in draft.gaps)
    print(_TEST_MATERIALS["text_015"])


def chapter_string_contract():
    source = {"doc_id": "d", "filename": _TEST_MATERIALS["text_016"], "section": "26.3.3. Operation"}
    state = AgentState(
        goal=_TEST_MATERIALS["text_042"],
        evidence=[EvidenceItem(index=1, chunk_id="c", doc_id="d", page=1, text=_TEST_MATERIALS["text_067"])],
        nodes=[
            SynthesisNode(
                id="n", title=_TEST_MATERIALS["text_068"], query=_TEST_MATERIALS["text_069"], source_sections=[source]
            )
        ],
    )
    raw = {
        "modules": [
            {
                "heading": "User heading",
                "source_sections": ["manual.pdf / 26.3.3. Operation"],
                "paragraphs": [{"text": _TEST_MATERIALS["text_070"], "sources": [1]}],
            }
        ]
    }
    assert finalize_draft(raw, state).modules
    raw["modules"][0]["source_sections"] = [_TEST_MATERIALS["text_017"]]
    with patch(
        "app.llm.get_llm_client",
        return_value=SimpleNamespace(generate=lambda **kw: SimpleNamespace(text=json.dumps(raw))),
    ):
        draft = compose_synthesis(state, SimpleNamespace(answer_llm_provider="qwen", qa_max_context_chars=6000))
    assert draft.compose_mode == "refused" and _TEST_MATERIALS["text_043"] in draft.markdown and not draft.modules


if __name__ == "__main__":
    evidence_contract()
    task_transitions()
    chapter_contracts()
    step_contracts()
    real_gap_replay()
    chapter_string_contract()
    print(_TEST_MATERIALS["text_018"])
