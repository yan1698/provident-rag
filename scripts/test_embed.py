"""Day 2 embedding acceptance: worker fills pgvector and reruns idempotently."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import func, select, text

from app.core.config import get_settings
from app.db.session import SessionLocal, init_db
from app.embed.service import count_pending, run_embedding_pipeline
from app.models import Chunk
from lib.accept import assert_ok
from lib.http import API_BASE, get_json, post_file

RAW = ROOT / "data" / "raw"


def _ensure_sample_ingested() -> None:
    md_path = RAW / "sample.md"
    assert_ok(md_path.is_file(), f"missing {md_path}")
    result = post_file("sample.md", md_path.read_bytes(), "text/markdown", timeout=120)
    assert_ok(result.get("chunk_count", 0) >= 1, _TEST_MATERIALS["text_004"])
    print(f"ingest ok doc_id={result['doc_id']} chunks={result['chunk_count']}")


def main() -> None:
    settings = get_settings()
    if settings.hf_home:
        os.environ.setdefault("HF_HOME", settings.hf_home)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    health = get_json(f"{API_BASE}/health", timeout=120)
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    print("health ok")

    init_db()
    _ensure_sample_ingested()

    with SessionLocal() as db:
        pending_before = count_pending(db)
        assert_ok(pending_before >= 1, _TEST_MATERIALS["text_006"])
        print(f"pending_before={pending_before}")

    with SessionLocal() as db:
        stats = run_embedding_pipeline(db, settings=settings)
        pending_after = count_pending(db)
    assert_ok(stats.embedded >= 1, f"worker embedded nothing: {stats}")
    assert_ok(pending_after == 0, f"{_TEST_MATERIALS['text_001']}{pending_after}")
    print(
        f"embed ok embedded={stats.embedded} copied={stats.copied} "
        f"batches={stats.batches} pending_after={pending_after}"
    )

    with SessionLocal() as db:
        total = int(db.scalar(select(func.count()).select_from(Chunk)) or 0)
        with_vec = int(db.scalar(select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))) or 0)
        dim = db.scalar(text("SELECT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL LIMIT 1"))
    assert_ok(total == with_vec, f"{_TEST_MATERIALS['text_002']}{with_vec}/{total}")
    assert_ok(dim == 384, f"unexpected embedding dim: {dim}")
    print(f"vectors ok count={with_vec} dim={dim}")

    with SessionLocal() as db:
        stats2 = run_embedding_pipeline(db, settings=settings)
        pending_final = count_pending(db)
    assert_ok(stats2.embedded == 0, f"{_TEST_MATERIALS['text_003']}{stats2}")
    assert_ok(pending_final == 0, f"pending reappeared: {pending_final}")
    print("idempotency ok")

    print(_TEST_MATERIALS["text_005"])


if __name__ == "__main__":
    main()
