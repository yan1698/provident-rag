"""Ingest one throwaway file through the live database, then delete it."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.ingest.service import ingest_file
from app.models import Document


def main() -> None:
    settings = get_settings()
    print(
        "chunk",
        settings.chunk_min_tokens,
        settings.chunk_max_tokens,
        settings.chunk_overlap_tokens,
    )
    body = (
        _TEST_MATERIALS["text_008"]
        + (_TEST_MATERIALS["text_014"] * 40)
        + _TEST_MATERIALS["text_003"]
        + (_TEST_MATERIALS["text_004"] * 40)
        + "\n"
    )
    data = body.encode("utf-8")
    db = SessionLocal()
    doc_id = None
    try:
        stale = (
            db.query(Document).filter(Document.filename == "prep-check.md", Document.owner_id.is_(None)).one_or_none()
        )
        if stale is not None:
            db.delete(stale)
            db.commit()
        first = ingest_file(
            db,
            filename=_TEST_MATERIALS["text_005"],
            data=data,
            content_type="text/markdown",
            settings=settings,
            owner_id=None,
        )
        print("first", first.reused, first.chunk_count, first.token_stats)
        doc_id = first.doc_id
        second = ingest_file(
            db,
            filename=_TEST_MATERIALS["text_006"],
            data=data,
            content_type="text/markdown",
            settings=settings,
            owner_id=None,
        )
        if not second.reused or second.doc_id != first.doc_id:
            raise SystemExit(f"{_TEST_MATERIALS['text_001']}{second.reused}")
        doc = db.get(Document, first.doc_id)
        if doc is None or int(doc.ingest_version) != 3:
            raise SystemExit(f"FAIL: version {getattr(doc, 'ingest_version', None)}")
        same = [
            chunk
            for chunk in doc.chunks
            if _TEST_MATERIALS["text_015"] in (chunk.text or "") and _TEST_MATERIALS["text_016"] in (chunk.text or "")
        ]
        if not same:
            raise SystemExit(_TEST_MATERIALS["text_007"])
        too_big = [chunk.token_count for chunk in doc.chunks if chunk.token_count > settings.chunk_max_tokens]
        if too_big:
            raise SystemExit(f"{_TEST_MATERIALS['text_002']}{too_big}")
        paths = sorted({chunk.section_path or "" for chunk in doc.chunks})
        print("paths", paths)
        if not any(_TEST_MATERIALS["text_017"] in path for path in paths) or not any(
            _TEST_MATERIALS["text_018"] in path for path in paths
        ):
            raise SystemExit(f"FAIL: section paths {paths}")
        for chunk in doc.chunks:
            path = chunk.section_path or ""
            if _TEST_MATERIALS["text_009"] in path and _TEST_MATERIALS["text_010"] in (chunk.text or ""):
                raise SystemExit("FAIL: section crossed")
            if _TEST_MATERIALS["text_011"] in path and _TEST_MATERIALS["text_012"] in (chunk.text or ""):
                raise SystemExit(_TEST_MATERIALS["text_013"])
        doc_id = doc.id
        print("db ingest ok", first.chunk_count)
    finally:
        if doc_id is not None:
            row = db.get(Document, doc_id)
            if row is not None:
                db.delete(row)
                db.commit()
        db.close()


if __name__ == "__main__":
    main()
