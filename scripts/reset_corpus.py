"""把库里的文档收敛回 golden set 声明的语料，保证评测可复现。

`scripts/test_ingest.py` 会上传 `generated.txt`、`sample.pdf` 这类合成夹具，
它们的填充文本和真实语料高度相似，会挤进 Top-K 候选，让 Recall@k / MRR 失真。
跑 Day 10 评测前执行本脚本，只保留当前评测集里 `corpus` 声明的文档（可多份）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop documents that are not part of the eval corpus")
    parser.add_argument("--keep", action="append", default=[], help="Extra filename to keep (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", dest="eval_set", default="", help="Eval set name (default: golden-agent-notes-v1)")
    args = parser.parse_args()

    from sqlalchemy import func, select

    from app.db.session import SessionLocal, init_db
    from app.eval.golden import DEFAULT_EVAL_SET, corpus_filenames, load_golden
    from app.models import Chunk, Document

    golden = load_golden(name=args.eval_set or DEFAULT_EVAL_SET)
    keep = set(corpus_filenames(golden.corpus)) | {Path(name).name for name in args.keep}
    keep.discard("")
    if not keep:
        raise SystemExit("golden set has no corpus and no --keep given; refusing to wipe everything")

    init_db()
    with SessionLocal() as db:
        rows = db.execute(
            select(Document, func.count(Chunk.id))
            .outerjoin(Chunk, Chunk.doc_id == Document.id)
            .group_by(Document.id)
            .order_by(Document.created_at)
        ).all()

        doomed = [(doc, count) for doc, count in rows if doc.filename not in keep]
        for doc, count in rows:
            flag = "keep  " if doc.filename in keep else "delete"
            print(f"{flag} {doc.filename:<20} chunks={count}")

        if not doomed:
            print("\ncorpus already clean")
            return
        if args.dry_run:
            print(f"\ndry-run: would delete {len(doomed)} documents")
            return

        for doc, _ in doomed:
            db.delete(doc)
        db.commit()
        print(f"\ndeleted {len(doomed)} documents; corpus now = {sorted(keep)}")


if __name__ == "__main__":
    main()
