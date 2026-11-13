"""Structural citation checks. They do not change Verifier claim statuses."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.generate.prompt import EvidenceItem


def assess_citations(
    evidence: list[EvidenceItem],
    rows: dict[str, dict[str, Any]],
    *,
    owner_id: UUID | None,
) -> list[dict[str, Any]]:
    """Compare cited chunks with the current rows.

    A chunk the caller does not own is reported as ``missing``. That avoids
    confirming another tenant's id. Claim statuses stay on the Verifier.
    """
    findings: list[dict[str, Any]] = []
    for item in evidence:
        chunk_id = str(item.chunk_id or "")
        row = rows.get(chunk_id)
        status = "ok"
        if row is None:
            status = "missing"
        else:
            row_owner = row.get("owner_id")
            if owner_id is not None and str(row_owner or "") != str(owner_id):
                status = "missing"
                row = None
            elif item.chunk_hash and str(row.get("chunk_hash") or "") != item.chunk_hash:
                status = "hash_mismatch"
            elif (item.text or "") != str(row.get("text") or ""):
                status = "text_mismatch"
        payload: dict[str, Any] = {
            "index": int(item.index),
            "chunk_id": chunk_id,
            "status": status,
        }
        if row is not None and status != "missing":
            payload["doc_id"] = str(row.get("doc_id") or "")
            payload["ingest_version"] = int(row.get("ingest_version") or 0)
            payload["section_path"] = str(row.get("section_path") or "")
            page = int(row.get("page") or 0)
            start = int(row.get("start_offset") or 0)
            end = int(row.get("end_offset") or 0)
            payload["locator"] = f"p{page}:{start}-{end}"
        findings.append(payload)
    return findings


def load_citation_rows(
    db: Session,
    chunk_ids: list[str],
    *,
    owner_id: UUID | None = None,
) -> dict[str, dict[str, Any]]:
    ids: list[UUID] = []
    for raw in chunk_ids:
        try:
            ids.append(UUID(str(raw)))
        except (TypeError, ValueError):
            continue
    if not ids:
        return {}
    from sqlalchemy import select

    from app.models import Chunk, Document

    stmt = (
        select(
            Chunk.id,
            Chunk.doc_id,
            Chunk.page,
            Chunk.start_offset,
            Chunk.end_offset,
            Chunk.text,
            Chunk.chunk_hash,
            Chunk.section_path,
            Document.owner_id,
            Document.ingest_version,
        )
        .join(Document, Document.id == Chunk.doc_id)
        .where(Chunk.id.in_(ids))
    )
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    rows = db.execute(stmt).all()
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        found[str(row.id)] = {
            "doc_id": row.doc_id,
            "page": row.page,
            "start_offset": row.start_offset,
            "end_offset": row.end_offset,
            "text": row.text or "",
            "chunk_hash": row.chunk_hash or "",
            "section_path": row.section_path or "",
            "owner_id": row.owner_id,
            "ingest_version": row.ingest_version,
        }
    return found


def assess_snapshot(db: Session, evidence: list[dict], *, owner_id: UUID | None) -> list[dict[str, Any]]:
    """Adapt legacy JSON snapshots, then batch-check with the existing verifier.

    Missing identity/text in older records is unverifiable, never an implicit ok.
    Does not modify the historical snapshot or fetch each chunk separately.
    """
    findings: list[dict[str, Any]] = []
    items: list[EvidenceItem] = []
    positions: list[int] = []
    for position, raw in enumerate(evidence):
        raw = raw if isinstance(raw, dict) else {}
        index = raw.get("index")
        if not isinstance(index, int) or index < 1:
            index = position + 1
        chunk_id = str(raw.get("chunk_id") or "")
        findings.append({"index": index, "chunk_id": chunk_id, "status": "unverifiable"})
        if not chunk_id or not isinstance(raw.get("text"), str):
            continue
        items.append(
            EvidenceItem(
                index=index,
                chunk_id=chunk_id,
                doc_id=str(raw.get("doc_id") or ""),
                page=0,
                text=raw["text"],
                chunk_hash=str(raw.get("chunk_hash") or ""),
            )
        )
        positions.append(position)
    rows = load_citation_rows(db, [item.chunk_id for item in items], owner_id=owner_id)
    for position, finding in zip(positions, assess_citations(items, rows, owner_id=owner_id)):
        findings[position] = finding
    return findings
