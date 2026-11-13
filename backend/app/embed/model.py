from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings

EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(
        settings.embedding_model_ref,
        device=settings.embedding_device,
    )


def encode_texts(
    texts: list[str],
    *,
    settings: Settings | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    settings = settings or get_settings()
    from app.search.stages import stage_timer

    with stage_timer("embed_model_load_ms"):
        model = get_embedder()
    with stage_timer("embed_inference_ms"):
        vectors = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return [row.tolist() for row in vectors]


def encode_query(text: str, *, settings: Settings | None = None) -> list[float]:
    settings = settings or get_settings()
    prefix = settings.embedding_query_prefix or ""
    return encode_texts([f"{prefix}{text}" if prefix else text], settings=settings)[0]
