from __future__ import annotations

import argparse
import os
import sys

from uuid import UUID

from app.core.config import get_settings
from app.db.session import SessionLocal, init_db
from app.embed.service import count_pending, run_embedding_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-embed chunks into pgvector (Day 2)")
    parser.add_argument("--doc-id", type=str, default=None, help="Only embed chunks for this doc UUID")
    parser.add_argument("--owner-id", type=str, default=None, help="Only embed chunks owned by this user UUID")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process this run")
    parser.add_argument("--dry-run", action="store_true", help="Count work without calling the model")
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.hf_home:
        os.environ.setdefault("HF_HOME", settings.hf_home)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    init_db()

    doc_id = UUID(args.doc_id) if args.doc_id else None
    owner_id = UUID(args.owner_id) if args.owner_id else None

    with SessionLocal() as db:
        pending_before = count_pending(db, doc_id=doc_id, owner_id=owner_id)
        stats = run_embedding_pipeline(
            db,
            settings=settings,
            doc_id=doc_id,
            owner_id=owner_id,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        pending_after = count_pending(db, doc_id=doc_id, owner_id=owner_id)

    if not args.dry_run:
        init_db()

    mode = "dry-run" if args.dry_run else "embed"
    print(
        f"[{mode}] pending_before={pending_before} embedded={stats.embedded} "
        f"copied={stats.copied} skipped={stats.skipped} batches={stats.batches} "
        f"pending_after={pending_after}"
    )
    return 0 if pending_after == 0 or stats.embedded > 0 or args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
