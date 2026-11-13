"""Unified entry adapter. The existing Agent classifier and executors stay separate."""

from __future__ import annotations

import re

from app.agent.classify import classify_task_kind

READING = re.compile(r"全文|整篇|完整.{0,6}(阅读|读完)|逐章|系统阅读|按篇阅读")


def decide(goal, requested="auto", *, answer_source="document_only", target_document_id=None, has_answer_context=False):
    if requested not in {"auto", "quick", "synthesis", "digest"}:
        raise ValueError("invalid analysis_mode")
    decision = dict(
        requested=requested,
        actual="quick",
        reason="safe_document_default",
        override=requested != "auto",
        fallback_reason="",
    )
    if answer_source == "general_explanation":
        return {**decision, "actual": "general_explanation", "reason": "explicit_source_policy"}
    if requested != "auto":
        if requested == "digest" and not target_document_id:
            raise ValueError("按篇阅读需要明确选择目标文档")
        return {**decision, "actual": requested, "reason": "user_override"}
    from app.agent.memory import is_answer_pointer

    if has_answer_context and is_answer_pointer(goal):
        return {**decision, "reason": "answer_reference_first"}
    if READING.search(goal):
        if target_document_id:
            return {**decision, "actual": "digest", "reason": "reading_target_confirmed"}
        return {**decision, "fallback_reason": "reading_target_missing", "reason": "请先选择阅读文档；本次使用快速问答"}
    try:
        # settings=None deliberately prevents optional provider classification.
        kind = classify_task_kind(goal, settings=None)
    except Exception:
        return {**decision, "fallback_reason": "classifier_failed"}
    if kind == "synthesis":
        return {**decision, "actual": kind, "reason": "existing_classifier_rules"}
    return decision


def validate_scope(db, owner_id, mode, collection_ids):
    """Fail closed for explicit scopes; legacy router silently drops unknown IDs."""
    if mode not in {"soft", "hard"}:
        return
    from app.corpus.library import list_collections

    allowed = {str(row.id) for row in list_collections(db, owner_id)}
    ids = {str(item) for item in collection_ids or []}
    if not ids or not ids <= allowed or mode == "hard" and len(ids) != 1:
        raise ValueError("所选知识范围不可用，请重新选择")


def validate_target(db, owner_id, target, scope):
    from app.models import Document

    row = db.get(Document, target)
    if row is None or row.owner_id != owner_id or not scope.allows(row.collection_id):
        raise ValueError("target document not found in selected scope")
    return row
