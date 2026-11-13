"""Small, bounded chapter reader shared by synthesis and digest.

It reuses ordered chunks and existing prompt budgets. It never promises full
reading when only a ranked/budgeted subset has been inspected.
"""

from __future__ import annotations

import re
from typing import Any

from app.corpus.sections import required_sections, section_records
from app.generate.fence import format_evidence_block
from app.generate.prompt import EvidenceItem


def evidence_identity_error(items, store):
    """Store reads already enforce owner, hard scope and explicit document target."""
    rows = store.load_chunks([item.chunk_id for item in items])
    for item in items:
        row = rows.get(str(item.chunk_id))
        if (
            row is None
            or str(row.doc_id) != str(item.doc_id)
            or row.text != item.text
            or (item.chunk_hash and row.chunk_hash != item.chunk_hash)
        ):
            return "原文来源缺失、身份改变或不在当前授权范围内，已拒绝交付该回答。"
    return ""


def reference_rows(state, store):
    import hashlib

    reference = state.reference or {}
    if reference.get("anchors"):
        from dataclasses import replace

        combined = {}
        for anchor in reference["anchors"]:
            rows, error = reference_rows(replace(state, reference=anchor), store)
            if error:
                return [], error  # A valid first step never rescues an invalid second.
            combined.update((str(row.id), row) for row in rows)
        if set(combined) != set(reference.get("chunk_ids") or []):
            return [], "多个步骤的来源范围不一致，无法核验。"
        return list(combined.values()), ""
    wanted = list(reference.get("chunk_ids") or [])
    rows = store.load_chunks(wanted)
    if set(rows) != set(wanted) or not rows:
        return [], "步骤来源已删除、缺失或不在当前访问范围内，无法核验。"
    proofs = reference.get("source_proofs") or []
    if {p.get("chunk_id") for p in proofs} != set(wanted):
        return [], "历史步骤缺少完整的原文身份凭据，请重新指定原文。"
    current_sections = {}
    for proof in proofs:
        row = rows.get(proof["chunk_id"])
        if (
            row is None
            or not proof.get("chunk_hash")
            or not proof.get("text_sha256")
            or (proof.get("doc_id") and str(row.doc_id) != str(proof["doc_id"]))
            or row.chunk_hash != proof.get("chunk_hash")
            or hashlib.sha256(row.text.encode()).hexdigest() != proof.get("text_sha256")
        ):
            return [], "步骤来源的哈希或文本已改变，无法用历史步骤说明新增事实。"
        if proof.get("section_id"):
            if row.doc_id not in current_sections:
                current_sections[row.doc_id] = section_records(row.doc_id, store.load_doc_chunks(row.doc_id))
            if not any(
                s["id"] == proof["section_id"] and row.id in s["chunk_ids"] for s in current_sections[row.doc_id]
            ):
                return [], "步骤原文章节身份已改变或无法核验，请重新指定章节。"
    return sorted(rows.values(), key=lambda r: (r.start_offset, r.page)), ""


def reading_goal(state) -> str:
    context = "、".join(state.task_spec.subjects) if state.task_spec else ""
    if state.reference and state.reference.get("retrieve"):
        context += " " + str(state.reference.get("text") or "")
    return state.goal + " " + context


def is_step_read(state) -> bool:
    reference = state.reference or {}
    return bool(reference.get("kind") == "step" and reference.get("retrieve") and reference.get("chunk_ids"))


def step_context_rows(state, store):
    """Reread a validated operation and its scoped conditions/risks, never history text."""
    anchors, error = reference_rows(state, store)
    if error:
        return [], error
    selected = {str(row.id): row for row in anchors}
    for doc_id in {row.doc_id for row in anchors}:
        rows = store.load_doc_chunks(doc_id)
        doc = type("DocumentRef", (), {"doc_id": doc_id})()
        plan = reading_plan(state, doc, section_records(doc_id, rows), rows=rows)
        if plan.get("error"):
            return [], plan["error"]
        needed = {
            cid
            for section in plan["required"]
            if section["reason"] in {"conditions", "risks"}
            for cid in section["chunk_ids"]
        }
        selected.update((str(row.id), row) for row in rows if str(row.id) in needed)
    return list(selected.values()), ""


def reading_plan(state, doc, sections: list[dict], *, rows=None) -> dict:
    """Resolve a step through source identity, then reuse existing section scopes.

    Legacy anchors without section IDs resolve from current chunk membership.
    Ambiguous heading-boundary chunks ask for clarification rather than choosing
    the first repeated title. Ownership/hash validation stays on reference_rows.
    """
    if not is_step_read(state):
        return required_sections(reading_goal(state), sections)
    reference = state.reference
    proofs = {p["chunk_id"]: p for p in reference.get("source_proofs", [])}
    by_row = {str(r.id): r for r in rows or []}
    targets = {}
    for cid in reference["chunk_ids"]:
        proof = proofs.get(cid, {})
        if proof.get("doc_id") and str(proof["doc_id"]) != doc.doc_id:
            continue
        matches = [s for s in sections if cid in s["chunk_ids"]]
        if proof.get("section_id"):
            matches = [s for s in matches if s["id"] == proof["section_id"]]
        else:
            path = proof.get("section_path") or getattr(by_row.get(cid), "section_path", "")
            exact = [
                s for s in matches if path == " / ".join(s["path"]) or path.endswith(" / " + " / ".join(s["path"]))
            ]
            if exact:
                matches = exact
        if not matches or len(matches) != 1:
            return {
                "target_id": None,
                "required": [],
                "missing_categories": ["步骤章节身份无法唯一核验，请指定章节"],
                "error": "步骤章节身份已失效或存在歧义，请指定原文章节后重试。",
            }
        targets[matches[0]["id"]] = matches[0]
    if not targets:
        return {"target_id": None, "required": [], "missing_categories": ["当前文档不含步骤来源"]}
    primary = max(targets.values(), key=lambda s: (s["level"], s["ordinal"]))
    plan = required_sections(state.goal, sections, target_id=primary["id"])
    required = {s["id"]: s for s in plan["required"]}
    for identity in targets:
        for section in required_sections(state.goal, sections, target_id=identity)["required"]:
            required.setdefault(section["id"], section)
    plan.update(
        required=list(required.values()),
        anchor_section_ids=list(targets),
        target_basis="validated step source identity",
    )
    return plan


def select_reading(state, doc, rows: list[Any], cap: int, settings) -> list[Any]:
    from app.agent.digest import _select_doc_chunks, _chunk_scores

    sections = section_records(doc.doc_id, rows)
    plan = reading_plan(state, doc, sections, rows=rows)
    required = plan["required"]
    wanted = {cid for section in required for cid in section["chunk_ids"]}
    requested = [row for row in rows if str(row.id) in wanted] if wanted else rows
    already = set((state.reference or {}).get("chunk_ids") or []) if state.task_kind == "synthesis" else set()
    # Full procedures keep their ordered steps. Follow-ups spend the same budget
    # on the requested conditions/risks or file details instead of rereading step 1.
    if required:
        priorities = {"steps": 0, "conditions": 1, "risks": 2, "context": 3}
        priority = {cid: min(priorities[s["reason"]] for s in required if cid in s["chunk_ids"]) for cid in wanted}
        ranked = sorted(requested, key=lambda r: (priority[str(r.id)], r.start_offset))
        full_steps = bool(re.search(r"完整.*步骤|全部.*步骤|complete.*steps", state.goal, re.I))
        if not full_steps and re.search(r"前置|条件|风险|注意|prerequi|risk|caveat", state.goal, re.I):
            scores = _chunk_scores(
                state.goal + " prerequisites requirements conditions enable settings caution risk danger overwrite",
                [r.text for r in requested],
                settings,
            )
            by_id = {str(r.id): value for r, value in zip(requested, scores)}
            groups = [[r for r in requested if priority[str(r.id)] == p] for p in [1, 2, 3]]
            for group in groups:
                group.sort(key=lambda r: (-by_id[str(r.id)], r.start_offset))
            ranked = [
                group[i] for i in range(max((len(g) for g in groups), default=0)) for group in groups if i < len(group)
            ]
            ranked.extend(r for r in requested if priority[str(r.id)] == 0)
        elif not full_steps and state.reference and re.search(r"文件|排除|保留|file|exclude", state.goal, re.I):
            depth = {cid: max(len(s["path"]) for s in required if cid in s["chunk_ids"]) for cid in wanted}
            ranked.sort(key=lambda r: (priority[str(r.id)], -depth[str(r.id)], r.start_offset))
            if re.search(r"为什么|为何|why", state.goal, re.I):
                context = [r for r in requested if priority[str(r.id)] == 3 and str(r.id) not in already]
                if context:
                    scores = _chunk_scores(state.goal, [r.text for r in context], settings)
                    best = max(zip(context, scores), key=lambda pair: pair[1])[0]
                    ranked = [best] + [r for r in ranked if str(r.id) != str(best.id)]
        chosen = []
        used = 0
        for row in ranked:
            if str(row.id) in already:
                continue
            size = len(format_evidence_block(row.text, index=len(chosen) + 1, source=doc.filename, page=row.page)) + 1
            if used + size > cap:
                continue
            chosen.append(row)
            used += size
        ids = {str(row.id) for row in chosen}
        chosen = [row for row in rows if str(row.id) in ids]
    else:
        chosen = _select_doc_chunks(state.goal, rows, cap, settings)
        ids = {str(row.id) for row in chosen}
    state.reading.setdefault("documents", {})[doc.doc_id] = {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "target_id": plan["target_id"],
        "required_sections": required,
        "required_chunk_ids": [str(row.id) for row in requested],
        "selected_chunk_ids": [str(row.id) for row in chosen],
        "read_chunk_ids": [],
        "prompt_chunk_ids": [],
        "unread_chunk_ids": [str(row.id) for row in requested],
        "budget_pruned_chunk_ids": [
            str(row.id) for row in requested if str(row.id) not in ids and str(row.id) not in already
        ],
        "missing_categories": plan["missing_categories"],
        "max_chars": cap,
        "total_document_chunks": len(rows),
        "full_document_read": False,
        "complete": False,
    }
    return chosen


def mark_read(state, doc_id: str, items: list[EvidenceItem]) -> None:
    record = state.reading.get("documents", {}).get(str(doc_id))
    if record is None:
        return
    ids = {item.chunk_id for item in items}
    record["read_chunk_ids"] = list(dict.fromkeys(record["read_chunk_ids"] + [item.chunk_id for item in items]))
    read = set(record["read_chunk_ids"])
    record["unread_chunk_ids"] = [cid for cid in record["required_chunk_ids"] if cid not in read]
    record["complete"] = not record["unread_chunk_ids"] and not record["missing_categories"]
    record["full_document_read"] = len(read) == record["total_document_chunks"]
    # Keep phase truth: being loaded/read is distinct from a later prompt inclusion.
    record["selected_chunk_ids"] = list(dict.fromkeys(record["selected_chunk_ids"] + list(ids)))


def finalize_reading(state) -> None:
    from app.agent.state import GapItem

    sent = {item.chunk_id for item in state.evidence}
    for record in state.reading.get("documents", {}).values():
        record["prompt_chunk_ids"] = [cid for cid in record["read_chunk_ids"] if cid in sent]
        record["prompt_omitted_chunk_ids"] = [cid for cid in record["read_chunk_ids"] if cid not in sent]
        record["complete"] = record["complete"] and not record["prompt_omitted_chunk_ids"]
        if not record["complete"]:
            detail = (
                f"必要范围{len(record['required_chunk_ids'])}片段；已读{len(record['read_chunk_ids'])}，"
                f"准备写稿{len(record['prompt_chunk_ids'])}，未读{len(record['unread_chunk_ids'])}，"
                f"预算裁剪{len(record['budget_pruned_chunk_ids'])}。不能确认完整覆盖操作步骤、前置条件及风险。"
            )
            if record["missing_categories"]:
                detail += " 未识别范围：" + "、".join(record["missing_categories"])
            if not any(g.topic == f"阅读范围：{record['filename']}" for g in state.gaps):
                state.gaps.append(GapItem(topic=f"阅读范围：{record['filename']}", reason=detail))
            state.partial = True
    if state.reading.get("mode") == "bounded":
        for record in state.reading.get("documents", {}).values():
            record["coverage_claim"] = (
                "source-set accounting only; task completeness requires answer and citation review"
            )


def read_for_synthesis(state, store, settings) -> None:
    """When an operation names a real section, read it rather than trust old hit order."""
    if state.task_kind != "synthesis" or state.skips_retrieve() or not state.catalog:
        return
    if state.task_spec and state.task_spec.intent == "compare":
        return  # preserve the existing comparison retrieval lane
    from app.agent.state import GapItem

    selected: list[EvidenceItem] = []
    cap = int(getattr(settings, "qa_max_context_chars", 6000))
    if state.reference and state.reference.get("retrieve"):
        anchors, error = reference_rows(state, store)
        if error:
            state.evidence = []
            state.gaps.append(GapItem(topic="步骤来源", reason=error))
            state.partial = True
            return
        selected = [
            EvidenceItem(
                index=i + 1,
                chunk_id=r.id,
                doc_id=r.doc_id,
                page=r.page,
                text=r.text,
                chunk_hash=r.chunk_hash,
                start_offset=r.start_offset,
                section_path=r.section_path,
                role="reference",
            )
            for i, r in enumerate(anchors)
        ]
    remaining = cap - sum(len(format_evidence_block(i.text, index=i.index, page=i.page)) + 1 for i in selected)
    reference_docs = {item.doc_id for item in selected}
    for doc in state.catalog:
        if reference_docs and doc.doc_id not in reference_docs:
            continue
        rows = store.load_doc_chunks(doc.doc_id)
        plan = reading_plan(state, doc, section_records(doc.doc_id, rows), rows=rows)
        if plan.get("error"):
            state.gaps.append(GapItem(topic="步骤章节", reason=plan["error"]))
            state.partial = True
        if not plan["target_id"]:
            continue
        picked = select_reading(state, doc, rows, remaining, settings)
        existing_ids = {item.chunk_id for item in selected}
        picked = [r for r in picked if str(r.id) not in existing_ids]
        items = [
            EvidenceItem(
                index=len(selected) + i + 1,
                chunk_id=str(r.id),
                doc_id=doc.doc_id,
                page=r.page,
                text=r.text,
                chunk_hash=r.chunk_hash,
                start_offset=r.start_offset,
                section_path=getattr(r, "section_path", ""),
                role="section",
            )
            for i, r in enumerate(picked)
        ]
        selected.extend(items)
        remaining -= sum(len(format_evidence_block(i.text, index=i.index, page=i.page)) + 1 for i in items)
        mark_read(state, doc.doc_id, [i for i in selected if i.doc_id == doc.doc_id])
    if state.reading.get("documents") or selected:
        state.evidence_selection["retrieval_candidates"] = [vars(item).copy() for item in state.evidence]
        state.evidence = selected
        for node in state.nodes:
            node.chunk_ids = [item.chunk_id for item in selected if not node.doc_id or item.doc_id == node.doc_id]
            node.status = "retrieved" if node.chunk_ids else "gap"
            sections = [
                {"doc_id": d["doc_id"], "filename": d["filename"], "section": s["title"]}
                for d in state.reading.get("documents", {}).values()
                for s in d["required_sections"]
            ]
            if sections:
                node.source_sections = sections
            elif not node.source_sections:
                node.source_sections = [
                    {
                        "doc_id": i.doc_id,
                        "filename": next((d.filename for d in state.catalog if d.doc_id == i.doc_id), ""),
                        "section": i.section_path or "步骤来源",
                    }
                    for i in selected
                ]
            state.evidence_by_node[node.id] = [item.index for item in selected if item.chunk_id in node.chunk_ids]
