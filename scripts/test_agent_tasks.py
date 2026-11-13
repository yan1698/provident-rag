"""Offline contract for agent-tasks-v1.

Questions and section titles come from the ingested interview notes.
Does not call /query and does not load an embedding model.
"""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
if (_REPO / "backend" / "app").is_dir():
    ROOT = _REPO
    sys.path.insert(0, str(ROOT / "backend"))
else:
    ROOT = Path("/tmp")
    sys.path.insert(0, "/app")

from types import SimpleNamespace

from app.agent.analyze import accept_questions, analyze_task, apply_analyze
from app.agent.classify import classify_task_kind
from app.agent.decompose import count_section_hits, hint_for_question
from app.agent.memory import index_paragraphs, merge_followup_spec, resolve_answer_reference, select_answer_anchor
from app.agent.state import AgentState
from app.corpus.catalog import CatalogDocument
from app.generate.prompt import EvidenceItem

TASKS = ROOT / "data" / "eval" / "agent" / "agent-tasks-v1.json"
GOLDEN = ROOT / "data" / "eval" / "golden" / _TEST_MATERIALS["text_005"]


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"FAIL: {path.name}{_TEST_MATERIALS['text_003']}")
    return payload


def _blob(values) -> str:
    return " ".join(str(item) for item in values)


def _forbid(text: str, words: list[str], label: str) -> None:
    for word in words:
        _assert(word not in text, f"{label} contains {word!r}: {text}")


def _catalog(rows: list[dict]) -> list[CatalogDocument]:
    return [
        CatalogDocument(
            str(row.get("doc_id") or ""),
            str(row.get("filename") or ""),
            [str(item) for item in (row.get("sections") or [])],
            preview=str(row.get("preview") or ""),
        )
        for row in rows
    ]


def _check_route(item: dict) -> None:
    goal = item["goal"]
    expect = item["expect"]
    kind = classify_task_kind(goal, settings=None)
    spec = analyze_task(goal)
    if "task_kind" in expect:
        _assert(kind == expect["task_kind"], f"{item['id']} task_kind {kind}")
    if "intent" in expect:
        _assert(spec.intent == expect["intent"], f"{item['id']} intent {spec.intent}")
    if expect.get("subjects_empty"):
        _assert(not spec.subjects, f"{item['id']} subjects {spec.subjects}")
    if expect.get("questions_empty"):
        _assert(not spec.questions, f"{item['id']} questions {spec.questions}")
    blob = _blob(spec.subjects)
    for word in expect.get("subjects_include") or []:
        _assert(word in blob, f"{item['id']} missing subject {word}: {spec.subjects}")
    _forbid(blob, expect.get("subjects_exclude") or [], item["id"])
    qblob = _blob(spec.questions)
    for word in expect.get("questions_include_substr") or []:
        _assert(word in qblob, f"{item['id']} missing {word} in {spec.questions}")
    _forbid(qblob, expect.get("questions_forbid_substr") or [], item["id"])
    if "questions_max" in expect:
        _assert(len(spec.questions) <= int(expect["questions_max"]), f"{item['id']} {spec.questions}")
    if "questions_count" in expect:
        _assert(len(spec.questions) == int(expect["questions_count"]), f"{item['id']} {spec.questions}")
    if expect.get("hold_short"):
        held = classify_task_kind(goal, settings=SimpleNamespace(deepseek_api_key="present"))
        _assert(held == "factoid", f"{item['id']} short question upgraded: {held}")


def _check_accept(item: dict) -> None:
    accepted = accept_questions(item["intent"], item["goal"], item.get("subjects") or [], item.get("raw_questions"))
    expect = item["expect"]
    if expect.get("accepted") is False:
        _assert(accepted is None, f"{item['id']} should reject, got {accepted}")
        return
    _assert(accepted is not None, f"{item['id']} should accept")
    questions = accepted or []
    qblob = _blob(questions)
    for word in expect.get("questions_include_substr") or []:
        _assert(word in qblob, f"{item['id']} missing {word} in {questions}")
    _forbid(qblob, expect.get("questions_forbid_substr") or [], item["id"])
    if "questions_count" in expect:
        _assert(len(questions) == int(expect["questions_count"]), f"{item['id']} {questions}")


def _check_followup(item: dict) -> None:
    merged = merge_followup_spec(dict(item["incoming"]), user_goal=item["goal"], brief=item.get("brief") or "")
    expect = item["expect"]
    if "turn_kind" in expect:
        _assert(merged.get("turn_kind") == expect["turn_kind"], f"{item['id']} {merged}")
    if "intent" in expect:
        _assert(merged.get("intent") == expect["intent"], f"{item['id']} {merged}")
    subjects = [str(part) for part in (merged.get("subjects") or [])]
    if "subjects" in expect:
        _assert(subjects == list(expect["subjects"]), f"{item['id']} subjects {subjects}")
    if expect.get("subjects_empty"):
        _assert(not subjects, f"{item['id']} subjects {subjects}")
    questions = [str(part) for part in (merged.get("questions") or [])]
    if expect.get("questions_empty"):
        _assert(not questions, f"{item['id']} questions {questions}")
    qblob = _blob(questions)
    for word in expect.get("questions_include_substr") or []:
        _assert(word in qblob, f"{item['id']} missing {word} in {questions}")
    for exact in expect.get("questions_exclude_exact") or []:
        _assert(exact not in questions, f"{item['id']} kept template {exact}: {questions}")


def _check_hint(item: dict) -> None:
    catalog = _catalog(item.get("catalog") or [])
    vectors = item.get("vectors") or {}
    if vectors:
        query = [float(value) for value in vectors["query"]]
        passages = [[float(value) for value in row] for row in vectors["passages"]]

        def encode_query(_text: str) -> list[float]:
            return query

        def encode_passages(_texts: list[str]) -> list[list[float]]:
            return passages
    else:

        def encode_query(_text: str) -> list[float]:
            raise RuntimeError("lexical-only")

        def encode_passages(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("lexical-only")

    hinted = hint_for_question(
        item["question"],
        catalog,
        encode_query=encode_query,
        encode_passages=encode_passages,
        hint_min=item.get("hint_min"),
    )
    section = hinted.section if hinted is not None else ""
    _assert(section == item["expect"]["section"], f"{item['id']} section {section!r}")


def _check_refer(item: dict) -> None:
    answer = str(item.get("answer") or "")
    paragraphs = item.get("paragraphs")
    if paragraphs is None and item.get("use_indexed"):
        paragraphs = index_paragraphs(answer)
    outline = None
    if paragraphs is not None or item.get("pending_action"):
        outline = {
            "paragraphs": paragraphs or [],
            "source_answer": str(item.get("source_answer") or answer),
            "pending_action": str(item.get("pending_action") or ""),
        }
    state = AgentState(
        goal=str(item.get("goal") or ""),
        memory_brief=str(item.get("brief") or _TEST_MATERIALS["text_010"]),
        prior_answer=answer,
        outline=outline,
    )
    spec = apply_analyze(state)
    expect = item["expect"]
    reference = (state.outline or {}).get("reference") if isinstance(state.outline, dict) else None
    if expect.get("missed"):
        _assert(reference is None, f"{item['id']} paragraph rule captured {reference}")
        _assert(state.turn_kind == expect.get("turn_kind", "retrieve"), f"{item['id']} {state.turn_kind}")
        return
    _assert(isinstance(reference, dict) and reference.get("hit"), f"{item['id']} missed reference {state.turn_kind}")
    _assert(state.skips_retrieve(), f"{item['id']} still retrieves")
    _assert(not spec.get("questions"), f"{item['id']} questions {spec.get('questions')}")
    _assert(spec.get("intent") != "plan", f"{item['id']} upgraded to plan")
    if "action" in expect:
        _assert(reference.get("action") == expect["action"], f"{item['id']} action {reference.get('action')}")
    if "index" in expect:
        _assert(reference.get("index") == expect["index"], f"{item['id']} index {reference.get('index')}")
    if expect.get("text_exact"):
        rows = index_paragraphs(answer)
        chosen = next(row for row in rows if row["index"] == reference.get("index"))
        _assert(reference.get("text") == answer[chosen["start"] : chosen["end"]], f"{item['id']} text drifted")
    if expect.get("message_has"):
        _assert(
            expect["message_has"] in str(reference.get("message") or ""), f"{item['id']} {reference.get('message')}"
        )


def _check_anchor(item: dict) -> None:
    prepared = []
    for row in item.get("runs") or []:
        outline = dict(row.get("outline") or {})
        answer = str(row.get("answer") or "")
        if outline.get("compose_mode") == "written" and not outline.get("paragraphs"):
            outline["paragraphs"] = index_paragraphs(answer)
            outline.setdefault("answer_anchor", {"run_id": row.get("id")})
        prepared.append({"id": row.get("id"), "answer": answer, "outline": outline})
    anchor = select_answer_anchor(prepared)
    expect = item["expect"]
    _assert(anchor.get("run_id") == expect["anchor_run_id"], f"{item['id']} anchor {anchor.get('run_id')}")
    if not item.get("goal"):
        return
    reference = resolve_answer_reference(
        str(item["goal"]),
        answer=str(anchor.get("answer") or ""),
        paragraphs=list(anchor.get("paragraphs") or []),
    )
    _assert(isinstance(reference, dict) and reference.get("hit"), f"{item['id']} missed {reference}")
    _assert(reference.get("retrieve") is False, f"{item['id']} would retrieve")
    if "index" in expect:
        _assert(reference.get("index") == expect["index"], f"{item['id']} index {reference.get('index')}")
    if "action" in expect:
        _assert(reference.get("action") == expect["action"], f"{item['id']} action {reference.get('action')}")
    if expect.get("text_has"):
        _assert(expect["text_has"] in str(reference.get("text") or ""), f"{item['id']} text {reference.get('text')}")
    if expect.get("text_lacks"):
        _assert(
            expect["text_lacks"] not in str(reference.get("text") or ""), f"{item['id']}{_TEST_MATERIALS['text_004']}"
        )


def _check_counts(item: dict) -> None:
    catalog = _catalog(item.get("catalog") or [])
    fresh = [
        EvidenceItem(
            index=index,
            chunk_id=f"c{index}",
            doc_id=str(row.get("doc_id") or ""),
            page=1,
            text=str(row.get("text") or ""),
            role=str(row.get("role") or "hit"),
        )
        for index, row in enumerate(item.get("hits") or [], start=1)
    ]
    counted = count_section_hits(item.get("section"), fresh, catalog)
    expect = item["expect"]
    _assert(counted["corpus_hit_count"] == expect["corpus_hit_count"], f"{item['id']} {counted}")
    _assert(counted["section_hit_count"] == expect["section_hit_count"], f"{item['id']} {counted}")


def _prompt_text(tasks: dict) -> str:
    parts: list[str] = []
    for item in tasks.get("items") or []:
        for key in ("goal", "question", "brief"):
            if item.get(key):
                parts.append(str(item[key]))
        parts.extend(str(value) for value in (item.get("raw_questions") or []))
        parts.extend(str(value) for value in (item.get("subjects") or []))
        for row in item.get("catalog") or []:
            parts.append(str(row.get("filename") or ""))
            parts.extend(str(value) for value in (row.get("sections") or []))
    return "\n".join(parts)


def _shares_corpus(tasks: dict) -> None:
    golden = _load(GOLDEN)
    golden_files = [str(name) for name in golden.get("corpus") or []]
    task_files = [str(name) for name in tasks.get("corpus") or []]
    _assert(task_files == golden_files, f"{_TEST_MATERIALS['text_001']}{task_files}")
    blob = _prompt_text(tasks)
    _assert(_TEST_MATERIALS["text_008"] not in blob, _TEST_MATERIALS["text_006"])
    _assert(_TEST_MATERIALS["text_009"] not in blob, _TEST_MATERIALS["text_007"])


def main() -> None:
    tasks = _load(TASKS)
    _assert(tasks.get("schema") == "provident.agent.tasks.v1", tasks.get("schema"))
    _assert(tasks.get("name") == "agent-tasks-v1", tasks.get("name"))
    items = tasks.get("items") or []
    _assert(len(items) >= 12, f"{_TEST_MATERIALS['text_002']}{len(items)}")
    _shares_corpus(tasks)
    handlers = {
        "route": _check_route,
        "accept": _check_accept,
        "followup": _check_followup,
        "hint": _check_hint,
        "counts": _check_counts,
        "refer": _check_refer,
        "anchor": _check_anchor,
    }
    seen: set[str] = set()
    fixture = str(tasks.get("reference_answer") or "")
    for item in items:
        if item.get("mode") == "refer" and not item.get("answer"):
            item["answer"] = fixture
        item_id = str(item.get("id") or "")
        _assert(item_id and item_id not in seen, f"duplicate id {item_id}")
        seen.add(item_id)
        mode = str(item.get("mode") or "")
        _assert(mode in handlers, f"{item_id} unknown mode {mode}")
        handlers[mode](item)
    print(f"agent-tasks-v1 ok ({len(items)})")


if __name__ == "__main__":
    main()
