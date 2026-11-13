"""Day 11: replay golden traces without calling the generator LLM.

Uses the answer + evidence already stored on ``query_traces``. Retrieval
metrics are recomputed against current gold labels. Verifier is optional:
stored coverage/hallucination is the default so a report can be frozen on a
machine that only has Postgres; ``reverify=True`` re-runs the local
Cross-Encoder (still no online LLM) when thresholds or the model change.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.version import API_VERSION
from app.eval.golden import GoldenSet
from app.eval.service import latest_traces_for_golden, score_traces, with_chunk_hashes
from app.eval.snapshot import TracesMeta, corpus_snapshot, freeze_report
from app.generate.prompt import EvidenceItem
from app.models.trace import QueryTrace
from app.verify.service import verify_answer


def replay_snapshot(
    db: Session,
    golden: GoldenSet,
    *,
    reverify: bool = False,
    settings: Settings | None = None,
    api_version: str = API_VERSION,
    owner_id=None,
) -> dict[str, Any]:
    traces = latest_traces_for_golden(db, golden, owner_id=owner_id)
    overrides: dict[str, dict[str, Any]] | None = None
    if reverify:
        settings = settings or get_settings()
        overrides = {}
        for eval_id, row in traces.items():
            evidence = with_chunk_hashes(db, list(row.evidence or []))
            result = verify_answer(row.answer or "", to_evidence_items(evidence), settings=settings)
            overrides[eval_id] = asdict(result)

    report = score_traces(db, golden, traces, verification_by_id=overrides, owner_id=owner_id)
    corpus = corpus_snapshot(db, golden, owner_id=owner_id)
    meta = traces_meta(traces, reverify=reverify)
    return freeze_report(report, golden=golden, corpus=corpus, traces=meta, api_version=api_version)


def traces_meta(traces: dict[str, QueryTrace], *, reverify: bool) -> TracesMeta:
    rows = list(traces.values())
    providers = sorted({row.provider for row in rows if row.provider})
    models = sorted({row.llm_model for row in rows if row.llm_model})
    return TracesMeta(
        providers=providers,
        llm_models=models,
        trace_ids=[str(row.id) for row in rows],
        reverify=reverify,
        source="reverified" if reverify else "stored",
    )


def to_evidence_items(evidence: list[dict[str, Any]]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for position, raw in enumerate(evidence, start=1):
        items.append(
            EvidenceItem(
                index=int(raw.get("index") or position),
                chunk_id=str(raw.get("chunk_id") or ""),
                doc_id=str(raw.get("doc_id") or ""),
                page=int(raw.get("page") or 1),
                text=str(raw.get("text") or ""),
                chunk_hash=str(raw.get("chunk_hash") or ""),
            )
        )
    return items
