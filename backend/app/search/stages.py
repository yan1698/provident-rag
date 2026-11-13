"""Millisecond stage marks for one request. Values are durations, never document text."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from math import isfinite

_stages: ContextVar[dict[str, int] | None] = ContextVar("provident_stage_ms", default=None)
_meta: ContextVar[dict[str, str | int] | None] = ContextVar("provident_stage_meta", default=None)
_candidates: ContextVar[list[dict] | None] = ContextVar("provident_candidates", default=None)
_depth: ContextVar[int] = ContextVar("provident_search_depth", default=0)
_errors: ContextVar[int] = ContextVar("provident_diagnostic_errors", default=0)


def best_effort(function):
    """Diagnostic sinks must never change the result or mask a retrieval exception."""

    @wraps(function)
    def safe(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:  # noqa: BLE001 — no text/exception payload in diagnostic fallback
            _errors.set(_errors.get() + 1)
            return None

    return safe


def reset_stages(*, capture_candidates: bool = False) -> None:
    _stages.set({})
    _meta.set({"timing_schema_version": 2, "candidate_trace_enabled": int(capture_candidates)})
    _candidates.set([] if capture_candidates else None)
    _errors.set(0)


@best_effort
def add_stage(name: str, started: float) -> None:
    """Add elapsed milliseconds. A widened second pass sums into the same key."""
    bucket = _stages.get()
    if bucket is None:
        bucket = {}
        _stages.set(bucket)
    elapsed = int((time.perf_counter() - started) * 1000)
    bucket[name] = int(bucket.get(name) or 0) + elapsed


def snapshot_stages() -> dict[str, int]:
    try:
        return dict(_stages.get() or {})
    except Exception:  # noqa: BLE001
        return {}


@best_effort
def note_meta(name: str, value: str | int) -> None:
    """Structured timing context. Callers must pass counts, names, or devices, never document text."""
    if isinstance(value, bool):
        stored: str | int = int(value)
    elif isinstance(value, int):
        stored = value
    else:
        stored = str(value)[:80]
    bucket = _meta.get()
    if bucket is None:
        bucket = {}
        _meta.set(bucket)
    bucket[name] = stored


def snapshot_meta() -> dict[str, str | int]:
    try:
        return {**dict(_meta.get() or {}), "diagnostics_error_count": _errors.get()}
    except Exception:  # noqa: BLE001
        return {"diagnostics_error_count": _errors.get() + 1}


# Inclusive parents. Children are excluded only when that parent is present.
STAGE_PARENTS: dict[str, tuple[str, ...]] = {
    "retrieval_total_ms": (
        "vector_ms",
        "embed_ms",
        "embed_model_load_ms",
        "embed_inference_ms",
        "ann_ms",
        "bm25_ms",
        "sparse_query_tokenize_ms",
        "sparse_corpus_load_ms",
        "sparse_tokenize_ms",
        "sparse_index_build_ms",
        "sparse_search_ms",
        "rrf_ms",
        "rerank_fetch_ms",
        "sparse_version_check_ms",
        "sparse_cache_wait_ms",
        "sparse_snapshot_prepare_ms",
        "rerank_call_ms",
        "rerank_wait_ms",
        "rerank_model_load_ms",
        "rerank_ms",
    ),
    "vector_ms": ("embed_ms", "embed_model_load_ms", "embed_inference_ms", "ann_ms"),
    "embed_ms": ("embed_model_load_ms", "embed_inference_ms"),
    "bm25_ms": (
        "sparse_query_tokenize_ms",
        "sparse_corpus_load_ms",
        "sparse_tokenize_ms",
        "sparse_index_build_ms",
        "sparse_search_ms",
        "sparse_version_check_ms",
        "sparse_cache_wait_ms",
        "sparse_snapshot_prepare_ms",
    ),
    "rerank_call_ms": ("rerank_wait_ms", "rerank_model_load_ms", "rerank_ms"),
    "verify_ms": ("verify_wait_ms",),
}


def leaf_stages(stages: dict[str, int]) -> dict[str, int]:
    """Drop child timers so a total is not larger than the wall clock."""
    children = {name for parent, kids in STAGE_PARENTS.items() if parent in stages for name in kids}
    return {key: int(value) for key, value in stages.items() if key not in children}


def leaf_total_ms(stages: dict[str, int]) -> int:
    return sum(leaf_stages(stages).values())


@contextmanager
def stage_timer(name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        best_effort(add_stage)(name, started)


@best_effort
def record_candidates(stage: str, hits, *, scored: bool = False) -> None:
    """Allowlisted IDs/scores only; preserve order, including ties and repeated passes.

    ``scored`` accepts (hit, CE score) pairs before final TopK truncation.
    Dict input is the existing internal RRF accumulator, never a document payload.
    """
    bucket = _candidates.get()
    if bucket is None:
        return
    rows = []
    for rank, item in enumerate(hits, 1):
        hit, score = item if scored else (item, None)
        get = hit.get if isinstance(hit, dict) else lambda key, default=None: getattr(hit, key, default)
        value = float(score if scored else get("score", get("rrf")))
        row = {
            "chunk_id": str(get("chunk_id")),
            "doc_id": str(get("doc_id")),
            "score": value if isfinite(value) else None,
            "rank": rank,
        }
        rows.append(row)
    bucket.append({"stage": stage, "candidate_count": len(rows), "candidates": rows})


def snapshot_candidates() -> list[dict]:
    try:
        bucket = _candidates.get()
        return deepcopy(bucket) if isinstance(bucket, list) else []
    except Exception:  # noqa: BLE001
        return []


def observe_search(stage: str | None = None, timer: str | None = None):
    """Outermost search wall time/final TopK; nested searches retain their own stages."""

    def decorate(function):
        @wraps(function)
        def observed(*args, **kwargs):
            depth = _depth.get()
            token = _depth.set(depth + 1)
            started = time.perf_counter()
            try:
                hits = function(*args, **kwargs)
                if stage:
                    best_effort(record_candidates)(stage, hits)
                if depth == 0:
                    best_effort(record_candidates)("final", hits)
                    best_effort(note_meta)("returned_chunk_count", len(hits))
                return hits
            finally:
                if timer:
                    best_effort(add_stage)(timer, started)
                if depth == 0:
                    best_effort(add_stage)("retrieval_total_ms", started)
                _depth.reset(token)

        return observed

    return decorate
