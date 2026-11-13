"""Knowledge-graph package. Keep imports lazy so unit tests do not need SQLAlchemy 2."""

from __future__ import annotations

__all__ = [
    "build_document_graph",
    "count_pending",
    "graph_counts",
    "graph_job_state",
    "norm_name",
    "prune_orphans",
    "quote_in_chunk",
    "run_graph_pipeline",
    "start_graph_job",
]


def __getattr__(name: str):
    if name in {"build_document_graph"}:
        from app.graph.rules import build_document_graph

        return build_document_graph
    if name in {"norm_name", "quote_in_chunk"}:
        from app.graph import schema

        return getattr(schema, name)
    if name in {"count_pending", "graph_counts", "prune_orphans", "run_graph_pipeline"}:
        from app.graph import service

        return getattr(service, name)
    if name in {"graph_job_state", "start_graph_job"}:
        from app.graph.jobs import graph_job_state, start_graph_job

        return {"graph_job_state": graph_job_state, "start_graph_job": start_graph_job}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
