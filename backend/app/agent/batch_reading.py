"""Finite chapter reads over existing source identities, not a new retriever.

Receipts mean original text was loaded and validated; they do not mean that an
LLM saw it or that the answer delivered its facts. No model is called here.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict

from app.agent.reading import is_step_read, reading_plan, reference_rows
from app.corpus.sections import section_records
from app.generate.fence import format_evidence_block
from app.generate.prompt import EvidenceItem

HINTS = {
    "steps": "procedure steps backup start stop files directory tar exclude",
    "conditions": "prerequisites requirements before must configuration enabled continuous connection",
    "risks": "risk warning caveats overwrite incomplete failure cannot restored abort",
}


def instruction_weight(text):
    return len(
        set(
            re.findall(
                r"\b(?:must|essential|should|include|exclude|omit|omitted|unless|ensure|required|cannot)\b|必须|不得|确保|禁止",
                text,
                re.I,
            )
        )
    )


def eligible(state, settings=None):
    if state.task_kind not in {"synthesis", "digest"} or state.skips_retrieve() or state.catalog_inventory():
        return False
    if state.task_spec and state.task_spec.intent == "compare":
        return False
    if is_step_read(state):
        # Identity resolution and owner/scope/hash checks occur inside the tool,
        # including legacy catalogs that have no persisted section entries.
        return True
    return any(reading_plan(state, d, d.section_entries)["target_id"] for d in state.catalog if d.section_entries)


def pending(state):
    r = state.reading
    return r.get("mode") == "bounded" and r.get("status") == "reading"


def _goals(state):
    goal = state.goal
    if (state.task_spec and state.task_spec.intent == "plan") or re.search(r"完整|全部|complete|all steps", goal, re.I):
        return ["steps", "conditions", "risks"]
    result = []
    if re.search(r"步骤|文件|操作|step|file|procedure", goal, re.I):
        result.append("steps")
    if re.search(r"条件|前置|配置|为什么|prerequi|condition|why", goal, re.I):
        result.append("conditions")
    if re.search(r"风险|注意|risk|warning|caveat", goal, re.I):
        result.append("risks")
    return result or ["steps", "conditions", "risks"]


def _invalid_source(state, settings, error):
    from app.agent.state import GapItem

    state.evidence = []
    state.reading = {
        "mode": "bounded",
        "status": "invalid_source",
        "documents": {},
        "batches": [],
        "receipts": [],
        "error": error,
        "budget": {"generation_chars": max(1, int(getattr(settings, "agent_reading_generation_chars", 12000)))},
    }
    state.gaps.append(GapItem(topic="步骤来源", reason=error))
    state.partial = True


def initialize(state, store, settings):
    if state.reading.get("mode") == "bounded":
        return
    documents = {}
    goals = _goals(state)
    anchor_docs = set()
    if state.reference and state.reference.get("retrieve"):
        rows, error = reference_rows(state, store)
        if error:
            _invalid_source(state, settings, error)
            return
        anchor_docs = {r.doc_id for r in rows}
    for doc in state.catalog:
        if anchor_docs and doc.doc_id not in anchor_docs:
            continue
        current_rows = store.load_doc_chunks(doc.doc_id) if anchor_docs else None
        sections = section_records(doc.doc_id, current_rows) if current_rows is not None else doc.section_entries
        plan = reading_plan(state, doc, sections, rows=current_rows)
        if plan.get("error"):
            _invalid_source(state, settings, plan["error"])
            return
        if not plan["target_id"]:
            continue
        required = plan["required"]
        ids = list(dict.fromkeys(cid for s in required for cid in s["chunk_ids"]))
        documents[doc.doc_id] = {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "target_id": plan["target_id"],
            "target_basis": plan.get("target_basis", "named section"),
            "anchor_section_ids": plan.get("anchor_section_ids", []),
            "required_sections": required,
            "required_chunk_ids": ids,
            "selected_chunk_ids": [],
            "read_chunk_ids": [],
            "prompt_chunk_ids": [],
            "unread_chunk_ids": ids[:],
            "budget_pruned_chunk_ids": [],
            "missing_categories": [g for g in goals if g in plan["missing_categories"]],
            "total_document_chunks": len({cid for s in doc.section_entries for cid in s["chunk_ids"]}),
            "full_document_read": False,
            "complete": False,
            "goals": {
                g: {
                    "scope_chunk_ids": list(
                        dict.fromkeys(cid for s in required if s["reason"] in {g, "context"} for cid in s["chunk_ids"])
                    )
                    or ids[:],
                    "scope_basis": "named role/context sections"
                    if any(s["reason"] in {g, "context"} for s in required)
                    else "unlabelled target prose/code fallback; semantic role not certified",
                    "read_chunk_ids": [],
                    "sent_chunk_ids": [],
                    "answer_cited_chunk_ids": [],
                    "semantic_answer_coverage": "not_assessed",
                    "completeness_proof": "unknown: chapter roles are reading hints, not a semantic proof",
                }
                for g in goals
            },
        }
    if anchor_docs - set(documents):
        _invalid_source(state, settings, "步骤来源无法在当前可读章节中定位，请指定原文章节。")
        return
    state.evidence_selection["retrieval_candidates"] = [asdict(e) for e in state.evidence]
    state.evidence = []
    state.reading = {
        "mode": "bounded",
        "status": "reading",
        "documents": documents,
        "batches": [],
        "receipts": [],
        "budget": {
            "max_batches": max(1, min(6, int(getattr(settings, "agent_reading_max_batches", 3)))),
            "batch_chars": max(1, int(getattr(settings, "agent_reading_batch_chars", 6000))),
            "total_chars": max(1, int(getattr(settings, "agent_reading_total_chars", 18000))),
            "generation_chars": max(1, int(getattr(settings, "agent_reading_generation_chars", 12000))),
            "used_chars": 0,
            "elapsed_seconds": 0.0,
            "timeout_seconds": max(0.0, float(getattr(settings, "agent_reading_timeout_seconds", 30))),
        },
        "resume_policy": "explicit source revalidation; never automatically replay a model request",
    }


def _refresh(state):
    for d in state.reading.get("documents", {}).values():
        read = set(d["read_chunk_ids"])
        d["unread_chunk_ids"] = [cid for cid in d["required_chunk_ids"] if cid not in read]
        d["complete"] = not d["unread_chunk_ids"] and not d["missing_categories"]
        d["full_document_read"] = len(read) == d["total_document_chunks"]
        required_ids = {s["id"]: set(s["chunk_ids"]) for s in d["required_sections"]}
        d["read_section_ids"] = [sid for sid, ids in required_ids.items() if ids <= read]
        d["unread_section_ids"] = [sid for sid, ids in required_ids.items() if not ids <= read]
        for g in d["goals"].values():
            g["read_chunk_ids"] = [cid for cid in g["scope_chunk_ids"] if cid in read]


def read_batch(state, store, settings, *, stop=None, deadline=None):
    initialize(state, store, settings)
    r = state.reading
    if r["status"] != "reading":
        return {"status": r["status"], "done": True}
    b = r["budget"]
    started = time.monotonic()

    def interrupted():
        if stop and stop():
            return "cancelled"
        if deadline is not None and time.perf_counter() >= deadline:
            return "timeout"
        if b["timeout_seconds"] and b["elapsed_seconds"] + time.monotonic() - started >= b["timeout_seconds"]:
            return "timeout"
        return ""

    why = interrupted()
    if why:
        r["status"] = why
        return {"status": why, "done": True}
    cap = min(b["batch_chars"], b["total_chars"] - b["used_chars"])
    if len(r["batches"]) >= b["max_batches"] or cap <= 0:
        r["status"] = "budget_exhausted"
        return {"status": r["status"], "done": True}
    from app.agent.digest import _chunk_scores

    ranked = []
    core = []
    required_notes = []
    for d in r["documents"].values():
        # Existing catalog/source API determines the scope. Direct re-read below
        # checks current owner/scope/hash instead of trusting old in-memory text.
        rows = store.load_doc_chunks(d["doc_id"])
        unread = set(d["unread_chunk_ids"])
        candidates = [row for row in rows if str(row.id) in unread]
        target = next(s for s in d["required_sections"] if s["id"] == d["target_id"])
        target_ids = set(target["chunk_ids"])
        core.extend((d, row) for row in candidates if str(row.id) in target_ids)
        parent_note_ids = {
            cid
            for s in d["required_sections"]
            if s["title"] in {"Note", "Important", "Warning", "Caution", "注意", "警告"}
            and target["path"][: len(s["path"]) - 1] == s["path"][:-1]
            for cid in s["chunk_ids"]
        }
        required_notes.extend((d, row) for row in candidates if str(row.id) in parent_note_ids)
        if d.get("priority_chunk_ids"):
            by_id = {str(row.id): row for row in candidates}
            ranked.extend((d, by_id[cid]) for cid in d["priority_chunk_ids"] if cid in by_id)
            continue
        rank_start = len(ranked)
        groups = []
        for goal, record in d["goals"].items():
            direct = {cid for s in d["required_sections"] if s["reason"] == goal for cid in s["chunk_ids"]}
            own = direct or set(record["scope_chunk_ids"])
            group = [row for row in candidates if str(row.id) in own]
            opening = (
                {cid for s in d["required_sections"] if s["reason"] == goal for cid in s["chunk_ids"][:3]}
                if goal == "conditions"
                else set()
            )
            scores = (
                _chunk_scores(state.goal + " " + HINTS[goal], [row.text for row in group], settings) if group else []
            )
            group = [
                row
                for row, score in sorted(
                    zip(group, scores),
                    key=lambda p: (
                        p[0].id not in opening,
                        p[0].start_offset if p[0].id in opening else 0,
                        not bool(re.match(r"^#{1,6}\s", p[0].text)),
                        -instruction_weight(p[0].text),
                        -p[1],
                        p[0].start_offset,
                    ),
                )
            ]
            groups.append(group)
        context = {cid for s in d["required_sections"] if s["reason"] == "context" for cid in s["chunk_ids"]}
        context_rows = [row for row in candidates if str(row.id) in context]
        if context_rows:
            scores = _chunk_scores(
                state.goal + " background rationale not cannot requirements",
                [row.text for row in context_rows],
                settings,
            )
            parent_notes = {
                cid
                for s in d["required_sections"]
                if s["title"] in {"Note", "Important", "Warning", "Caution", "注意", "警告"}
                and target["path"][: len(s["path"]) - 1] == s["path"][:-1]
                for cid in s["chunk_ids"]
            }
            groups.append(
                [
                    row
                    for row, score in sorted(
                        zip(context_rows, scores), key=lambda p: (p[0].id not in parent_notes, -p[1], p[0].start_offset)
                    )
                ]
            )
        for i in range(max((len(g) for g in groups), default=0)):
            ranked.extend((d, g[i]) for g in groups if i < len(g))
        ranked.extend((d, row) for row in candidates)
        d["priority_chunk_ids"] = list(dict.fromkeys(str(row.id) for _, row in ranked[rank_start:]))
    # Preserve a named procedure's sequence (including continuations) before
    # optional adjacent material. Category rotation then fills other obligations.
    ranked = sorted(core, key=lambda pair: (pair[0]["doc_id"], pair[1].start_offset)) + required_notes + ranked
    # Original step anchors must be read even if another category scores higher.
    anchors = set((state.reference or {}).get("chunk_ids") or [])
    ranked.sort(key=lambda pair: 0 if str(pair[1].id) in anchors else 1)
    selected = []
    seen = set()
    used = 0
    for d, row in ranked:
        if str(row.id) in seen:
            continue
        seen.add(str(row.id))
        size = len(format_evidence_block(row.text, index=len(state.evidence) + len(selected) + 1, page=row.page)) + 1
        if used + size > cap:
            continue
        selected.append((d, row, size))
        used += size
    fresh = store.load_chunks([str(row.id) for _, row, _ in selected])
    batch = {"index": len(r["batches"]) + 1, "chunk_ids": [], "section_ids": [], "chars": 0, "status": "complete"}
    for d, row, size in selected:
        why = interrupted()
        if why:
            batch["status"] = why
            r["status"] = why
            break
        current = fresh.get(str(row.id))
        if current is None or current.chunk_hash != row.chunk_hash or current.text != row.text:
            r.setdefault("invalid_sources", []).append(str(row.id))
            continue
        section_ids = [s["id"] for s in d["required_sections"] if str(row.id) in s["chunk_ids"]]
        receipt = {
            "chunk_id": str(row.id),
            "doc_id": d["doc_id"],
            "hash": row.chunk_hash,
            "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
            "section_ids": section_ids,
            "batch": batch["index"],
            "status": "read",
            "text": row.text,
            "page": row.page,
            "start_offset": row.start_offset,
            "section_path": row.section_path,
        }
        r["receipts"].append(receipt)
        state.evidence.append(
            EvidenceItem(
                index=len(state.evidence) + 1,
                chunk_id=str(row.id),
                doc_id=d["doc_id"],
                page=row.page,
                text=row.text,
                chunk_hash=row.chunk_hash,
                start_offset=row.start_offset,
                section_path=row.section_path,
                role="read",
                section_id=section_ids[-1] if section_ids else "",
            )
        )
        d["read_chunk_ids"].append(str(row.id))
        d["selected_chunk_ids"].append(str(row.id))
        batch["chunk_ids"].append(str(row.id))
        batch["section_ids"].extend(s for s in section_ids if s not in batch["section_ids"])
        batch["chars"] += size
    b["used_chars"] += batch["chars"]
    b["elapsed_seconds"] += time.monotonic() - started
    r["batches"].append(batch)
    _refresh(state)
    if r["status"] == "reading":
        if all(not d["unread_chunk_ids"] for d in r["documents"].values()):
            r["status"] = "scope_loaded"
        elif len(r["batches"]) >= b["max_batches"] or not batch["chunk_ids"] or b["used_chars"] >= b["total_chars"]:
            r["status"] = "budget_exhausted"
    if r["status"] != "reading":
        for d in r["documents"].values():
            d["budget_pruned_chunk_ids"] = d["unread_chunk_ids"][:]
    for node in state.nodes:
        node.chunk_ids = [e.chunk_id for e in state.evidence if not node.doc_id or e.doc_id == node.doc_id]
        node.status = "retrieved" if node.chunk_ids else "gap"
    if not state.nodes and state.evidence:
        from app.agent.state import SynthesisNode

        state.nodes = [
            SynthesisNode(
                id="bounded-read",
                title=state.goal,
                query=state.goal,
                status="retrieved",
                chunk_ids=[e.chunk_id for e in state.evidence],
            )
        ]
    for node in state.nodes:
        node.source_sections = [
            {"doc_id": d["doc_id"], "filename": d["filename"], "section": s["title"]}
            for d in r["documents"].values()
            for s in d["required_sections"]
            if not node.doc_id or node.doc_id == d["doc_id"]
        ]
    return {
        "status": r["status"],
        "done": r["status"] != "reading",
        "batch": batch,
        "remaining_goals": {
            d["doc_id"]: {g: len(set(x["scope_chunk_ids"]) - set(x["read_chunk_ids"])) for g, x in d["goals"].items()}
            for d in r["documents"].values()
        },
    }


def revalidate(state, store):
    """Before writing/resuming, drop changed/deleted/out-of-scope receipts."""
    r = state.reading
    if r.get("mode") != "bounded":
        return
    if state.reference and state.reference.get("retrieve"):
        _, error = reference_rows(state, store)
        if error:
            from app.agent.state import GapItem

            state.evidence = []
            r["status"] = "invalid_source"
            r["error"] = error
            state.partial = True
            state.gaps.append(GapItem(topic="步骤来源", reason=error))
            return
    current = store.load_chunks([x["chunk_id"] for x in r["receipts"]])
    valid = []
    for receipt in r["receipts"]:
        row = current.get(receipt["chunk_id"])
        ok = bool(
            row
            and row.chunk_hash == receipt["hash"]
            and hashlib.sha256(row.text.encode()).hexdigest() == receipt["text_sha256"]
        )
        receipt["status"] = "read" if ok else "invalid"
        if ok:
            valid.append(
                EvidenceItem(
                    index=len(valid) + 1,
                    chunk_id=str(row.id),
                    doc_id=str(row.doc_id),
                    page=row.page,
                    text=row.text,
                    chunk_hash=row.chunk_hash,
                    section_path=row.section_path,
                    start_offset=row.start_offset,
                    role="read",
                    section_id=receipt["section_ids"][-1] if receipt["section_ids"] else "",
                )
            )
    state.evidence = valid
    if valid and not state.nodes:
        from app.agent.state import SynthesisNode

        state.nodes = [
            SynthesisNode(
                id="bounded-resume",
                title=state.goal,
                query=state.goal,
                status="retrieved",
                chunk_ids=[e.chunk_id for e in valid],
                source_sections=[
                    {"doc_id": d["doc_id"], "filename": d["filename"], "section": s["title"]}
                    for d in r["documents"].values()
                    for s in d["required_sections"]
                ],
            )
        ]
    for d in r["documents"].values():
        d["read_chunk_ids"] = [e.chunk_id for e in valid if e.doc_id == d["doc_id"]]
    _refresh(state)


def record_delivery(state, modules):
    r = state.reading
    if r.get("mode") != "bounded":
        return
    cited = {int(n) for m in modules for p in m.get("paragraphs", []) for n in p.get("sources", []) if str(n).isdigit()}
    sent = {e.chunk_id for e in state.evidence}
    user = {e.chunk_id for e in state.evidence if e.index in cited}
    for d in r["documents"].values():
        for g in d["goals"].values():
            g["sent_chunk_ids"] = [cid for cid in g["read_chunk_ids"] if cid in sent]
            g["answer_cited_chunk_ids"] = [cid for cid in g["sent_chunk_ids"] if cid in user]
            g["read_but_not_sent"] = [cid for cid in g["read_chunk_ids"] if cid not in sent]
            g["sent_but_not_cited"] = [cid for cid in g["sent_chunk_ids"] if cid not in user]
    r["quality"] = {
        "reading": "see task goal ledger; all chunks is not the success metric",
        "answer_semantic_coverage": "not_assessed",
        "citation_support": "existing verifier + current source integrity, separate from coverage",
    }


def writer_order(state):
    """Reserve the procedure sequence, then diversify the finite writer input.

    Heading/imperative cues allocate space, never certify semantic completeness.
    The writer still sees original whole chunks and uses the same source IDs.
    """
    items = state.evidence
    if state.reading.get("mode") != "bounded":
        return items
    ordered = []
    for d in state.reading.get("documents", {}).values():
        target = next(s for s in d["required_sections"] if s["id"] == d["target_id"])
        core = set(target["chunk_ids"])
        ordered.extend(sorted((e for e in items if e.chunk_id in core), key=lambda e: e.start_offset))
        # Preserve the prerequisite opening across page/chunk continuations.
        # Three chunks is a finite opening window, not a proof of completeness.
        opening = {cid for s in d["required_sections"] if s["reason"] == "conditions" for cid in s["chunk_ids"][:3]}
        ordered.extend(sorted((e for e in items if e.chunk_id in opening), key=lambda e: e.start_offset))
        parent_note_ids = {
            cid
            for s in d["required_sections"]
            if s["title"] in {"Note", "Important", "Warning", "Caution", "注意", "警告"}
            and target["path"][: len(s["path"]) - 1] == s["path"][:-1]
            for cid in s["chunk_ids"]
        }
        ordered.extend(e for e in items if e.chunk_id in parent_note_ids)
        groups = []
        for role in ["conditions", "risks", "context", "steps"]:
            ids = {cid for s in d["required_sections"] if s["reason"] == role for cid in s["chunk_ids"]}
            group = [e for e in items if e.chunk_id in ids and e.chunk_id not in core]
            if role == "context":
                group.sort(key=lambda e: e.index)  # already task-ranked during reading
            elif role == "steps":
                group.sort(key=lambda e: (-instruction_weight(e.text), e.start_offset))
            else:
                group.sort(
                    key=lambda e: (
                        not bool(re.match(r"^#{1,6}\s", e.text)),
                        -instruction_weight(e.text),
                        e.start_offset,
                    )
                )
            groups.append(group)
        ordered.extend(g[i] for i in range(max((len(g) for g in groups), default=0)) for g in groups if i < len(g))
    ordered.extend(items)
    seen = set()
    return [e for e in ordered if not (e.chunk_id in seen or seen.add(e.chunk_id))]
