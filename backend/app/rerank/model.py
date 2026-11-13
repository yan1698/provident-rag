from __future__ import annotations

import threading
import time
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from app.core.config import Settings, get_settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(
        settings.rerank_model_ref,
        device=settings.rerank_device,
        default_activation_function=torch.nn.Sigmoid(),
    )


def rerank_scores(
    query: str,
    passages: list[str],
    *,
    settings: Settings | None = None,
) -> list[float]:
    if not passages:
        return []
    return pair_scores([(query, passage) for passage in passages], settings=settings, label="rerank")


_inflight = 0
_inflight_lock = threading.Lock()


def pair_scores(
    pairs: list[tuple[str, str]],
    *,
    settings: Settings | None = None,
    label: str = "",
) -> list[float]:
    """Score arbitrary (query, passage) pairs with the Day 5 Cross-Encoder.

    Used by rerank (one query, many passages) and by the Day 7 Verifier
    (many claim–evidence pairs with different claims).
    """
    if not pairs:
        return []
    settings = settings or get_settings()
    from app.rerank.limit import rerank_slot
    from app.search.stages import add_stage, note_meta, stage_timer

    global _inflight
    wait_started = time.perf_counter()
    with rerank_slot(int(settings.rerank_max_inflight)):
        if label == "rerank":
            add_stage("rerank_wait_ms", wait_started)
        elif label == "verify":
            add_stage("verify_wait_ms", wait_started)
        with _inflight_lock:
            _inflight += 1
            running = _inflight
        try:
            if label == "rerank":
                note_meta("rerank_pairs", len(pairs))
                note_meta("rerank_inflight", running)
                note_meta("rerank_max_inflight", int(settings.rerank_max_inflight))
                note_meta("rerank_device", settings.rerank_device)
                note_meta("rerank_batch_size", int(settings.rerank_batch_size))
                note_meta("rerank_model", settings.rerank_model_name)
                note_meta("rerank_cuda", int(torch.cuda.is_available()))
            elif label == "verify":
                note_meta("verify_pairs", len(pairs))
                note_meta("verify_inflight", running)
            if label == "rerank":
                with stage_timer("rerank_model_load_ms"):
                    model = get_reranker()
                with stage_timer("rerank_ms"):
                    scores = model.predict(
                        pairs,
                        batch_size=settings.rerank_batch_size,
                        show_progress_bar=False,
                    )
            else:
                model = get_reranker()
                scores = model.predict(
                    pairs,
                    batch_size=settings.rerank_batch_size,
                    show_progress_bar=False,
                )
            return [float(s) for s in scores]
        finally:
            with _inflight_lock:
                _inflight -= 1
