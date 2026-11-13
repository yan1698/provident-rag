"""Enqueue graph jobs. State lives in Redis so API and worker agree."""

from __future__ import annotations

from app.queue.client import enqueue_graph, job_state


def graph_job_state(*, owner_id=None) -> dict[str, str | bool | int | None]:
    return job_state("graph", owner_id)


def start_graph_job(
    *,
    all_docs: bool = False,
    extract: bool = False,
    doc_ids: list | None = None,
    owner_id=None,
) -> bool:
    return enqueue_graph(
        all_docs=all_docs,
        extract=extract,
        doc_ids=doc_ids,
        owner_id=owner_id,
    )
