"""Day 11: freeze an EvalReport into a one-shot, comparable snapshot.

The snapshot is the unit you keep: corpus file hashes, which traces were scored,
the Day 9/10 metrics, and whether quality gates passed. Replay and Dify compare
both produce this shape so numbers are always lined up against the same schema.

Only ``corpus_snapshot`` needs a DB session; the report shape and the gate
arithmetic live in ``app.eval.gates`` so CI can test them without SQLAlchemy.
The re-exports below keep the existing ``from app.eval.snapshot import ...``
call sites working.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.eval.gates import (
    CorpusFile,
    CorpusSnapshot,
    GateResult,
    TracesMeta,
    evaluate_gates,
    freeze_report,
    metrics_dict,
    score_dict,
)
from app.eval.golden import GoldenSet, corpus_filenames, forbidden_matches
from app.models import Chunk, Document

__all__ = [
    "CorpusFile",
    "CorpusSnapshot",
    "GateResult",
    "TracesMeta",
    "corpus_snapshot",
    "evaluate_gates",
    "freeze_report",
    "metrics_dict",
    "score_dict",
]


def corpus_snapshot(db: Session, golden: GoldenSet, *, owner_id=None) -> CorpusSnapshot:
    expected = set(corpus_filenames(golden.corpus))
    stmt = (
        select(Document.filename, Document.file_hash, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.doc_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at)
    )
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    rows = db.execute(stmt).all()
    documents = [
        CorpusFile(
            filename=row.filename,
            file_hash=row.file_hash or "",
            chunk_count=int(row[2] or 0),
            expected=bool(expected) and row.filename in expected,
        )
        for row in rows
    ]
    return CorpusSnapshot(
        path=golden.corpus,
        documents=documents,
        chunk_count=sum(item.chunk_count for item in documents),
        off_corpus_count=sum(1 for item in documents if expected and item.filename not in expected),
        forbidden_documents=forbidden_matches((item.filename for item in documents), golden.forbidden_files),
    )
