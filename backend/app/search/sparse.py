from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import version
from types import MappingProxyType
from uuid import UUID

from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.search.stages import note_meta, observe_search, stage_timer

SNIPPET_CHARS = 200
# created_at-id-asc-v1 makes batch-ingest timestamp ties deterministic.
# Cache keys include this tuple so older snapshots cannot be reused.
INDEX_ALGORITHM = ("ascii-cjk-bigram-v1", "BM25Okapi-defaults", "created_at-id-asc-v1", version("rank-bm25"))
CORPUS_ORDER_SQL = "ORDER BY created_at ASC, id ASC"

# BM25 requires pre-tokenized text. Postgres 的 `simple` FTS 配置不做中文分词，
# 且离线环境不便引入 jieba 词典，因此这里用轻量正则近似：
# ASCII 单词/数字当作一个 token，CJK 保留单字，并追加相邻两字，避免「合同」和「同合」同分。
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def tokenize(value: str) -> list[str]:
    raw = [token.lower() for token in _TOKEN_RE.findall(value)]
    tokens = list(raw)
    for left, right in zip(raw, raw[1:]):
        if _CJK_CHAR.fullmatch(left) and _CJK_CHAR.fullmatch(right):
            tokens.append(left + right)
    return tokens


@dataclass
class SparseHit:
    chunk_id: UUID
    doc_id: UUID
    page: int
    snippet: str
    score: float
    start: int
    end: int


@observe_search("sparse", "bm25_ms")
def sparse_search(
    db: Session,
    query: str,
    *,
    top_k: int | None = None,
    owner_id: UUID | None = None,
    scope=None,
    settings: Settings | None = None,
) -> list[SparseHit]:
    settings = settings or get_settings()
    k = top_k if top_k is not None else settings.sparse_top_k
    k = max(1, min(k, settings.sparse_max_top_k))

    q = query.strip()
    if not q:
        return []
    with stage_timer("sparse_query_tokenize_ms"):
        query_tokens = tokenize(q)
    if not query_tokens:
        return []

    if settings.sparse_index_cache_enabled:
        from app.search.sparse_cache import CACHE, CacheLimits
        from app.search.sparse_version import Uncacheable, read_stamp

        try:
            stamp = read_stamp(db)
            key = (
                stamp.identity,
                str(owner_id) if owner_id is not None else None,
                _effective_scope(scope),
                INDEX_ALGORITHM,
            )
            snapshot = CACHE.get(
                key,
                stamp,
                lambda: _build_snapshot(db, owner_id, scope, settings),
                lambda: read_stamp(db),
                CacheLimits(
                    settings.sparse_index_cache_max_entries,
                    settings.sparse_index_cache_max_bytes,
                    settings.sparse_index_cache_wait_seconds,
                ),
            )
        except Uncacheable as exc:
            CACHE.fallback(str(exc))
        except Exception:  # optional cache failure cannot replace the legacy result/exception
            CACHE.fallback("cache_error")
        else:
            note_meta("sparse_corpus_chunk_count", len(snapshot.rows))
            note_meta("sparse_snapshot_estimated_bytes", snapshot.estimated_bytes)
            _note_rss("query")
            if not snapshot.rows:
                return []
            with stage_timer("sparse_search_ms"):
                scores = snapshot.index.get_scores(query_tokens)
                ranked = sorted(zip(snapshot.rows, scores), key=lambda item: item[1], reverse=True)
            return [
                SparseHit(
                    chunk_id=row.id,
                    doc_id=row.doc_id,
                    page=row.page,
                    snippet=row.snippet,
                    score=round(float(score), 6),
                    start=row.start_offset,
                    end=row.end_offset,
                )
                for row, score in ranked[:k]
                if not score <= 0
            ]
    else:
        note_meta("sparse_index_cache_status", "disabled")

    return _uncached_search(db, query_tokens, k, owner_id=owner_id, scope=scope)


def _uncached_search(db, query_tokens, k, *, owner_id=None, scope=None):
    """Original full-read/build/score path, also the fail-closed fallback."""

    owner_sql, extra = _owner_clause(owner_id)
    from app.corpus.scope import scope_sql

    scope_filter, scope_extra = scope_sql(scope)
    extra.update(scope_extra)
    owner_sql = f"{owner_sql}\n            {scope_filter}"
    with stage_timer("sparse_corpus_load_ms"):
        rows = (
            db.execute(
                text(
                    f"""
                SELECT id, doc_id, page, start_offset, end_offset, text
                FROM chunks
                WHERE 1=1
                {owner_sql}
                {CORPUS_ORDER_SQL}
                """
                ),
                extra,
            )
            .mappings()
            .all()
        )
    note_meta("sparse_corpus_chunk_count", len(rows))
    note_meta("sparse_index_reused", 0)
    if not rows:
        return []

    with stage_timer("sparse_tokenize_ms"):
        corpus_tokens = [tokenize(row["text"] or "") for row in rows]
    with stage_timer("sparse_index_build_ms"):
        bm25 = BM25Okapi(corpus_tokens)
    with stage_timer("sparse_search_ms"):
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(zip(rows, scores), key=lambda item: item[1], reverse=True)

    hits: list[SparseHit] = []
    for row, score in ranked[:k]:
        if score <= 0:
            continue
        content = row["text"] or ""
        snippet = content[:SNIPPET_CHARS]
        if len(content) > SNIPPET_CHARS:
            snippet += "…"
        hits.append(
            SparseHit(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                page=int(row["page"]),
                snippet=snippet,
                score=round(float(score), 6),
                start=int(row["start_offset"]),
                end=int(row["end_offset"]),
            )
        )
    return hits


def _effective_scope(scope):
    """Match scope_sql exactly; labels/widened flags are not SQL predicates."""
    if scope is None or scope.mode == "all" or not scope.collection_ids:
        return ("all", ())
    return ("hard" if scope.mode == "hard" else "soft", tuple(sorted(set(scope.collection_ids))))


def _build_snapshot(db, owner_id, scope, settings):
    # A failed optional SELECT must not abort the caller's transaction and prevent
    # the legacy fallback. Use Connection savepoints, never Session implicit flush.
    conn = db.connection() if isinstance(db, Session) else db
    with conn.begin_nested():
        return _construct_snapshot(db, owner_id, scope, settings)


def _construct_snapshot(db, owner_id, scope, settings):
    from app.corpus.scope import scope_sql
    from app.search.sparse_cache import SparseSnapshot, SnapshotRow, deep_size
    from app.search.sparse_version import Uncacheable

    _note_rss("build_before")
    owner_sql, params = _owner_clause(owner_id)
    scope_filter, extra = scope_sql(scope)
    params.update(extra)
    cap = settings.sparse_index_cache_max_chunks
    if cap < 1 or settings.sparse_index_cache_max_source_chars < 1:
        raise Uncacheable("build_budget_disabled")
    params["cache_row_limit"] = cap + 1
    with stage_timer("sparse_corpus_load_ms"):
        result = db.execute(
            text(f"""
            SELECT id, doc_id, page, start_offset, end_offset, text, created_at
            FROM chunks WHERE 1=1 {owner_sql} {scope_filter}
            {CORPUS_ORDER_SQL} LIMIT :cache_row_limit
        """),
            params,
        )
        rows, chars = [], 0
        try:
            for row in result.mappings():
                chars += len(row["text"] or "")
                if len(rows) >= cap or chars > settings.sparse_index_cache_max_source_chars:
                    raise Uncacheable("source_over_budget")
                rows.append(row)
        finally:
            result.close()
    note_meta("sparse_corpus_chunk_count", len(rows))
    # Duplicate created_at is now cacheable because id is a deterministic
    # secondary key on both paths. NULL timestamps remain unsafe: NULLS FIRST
    # / LAST differs across engines, and embedding-only writes must not invent
    # a new order. Duplicate (created_at, id) is still refused.
    order_keys = [(row["created_at"], row["id"]) for row in rows]
    if any(created is None or ident is None for created, ident in order_keys) or len(set(order_keys)) != len(
        order_keys
    ):
        raise Uncacheable("ambiguous_created_at")
    with stage_timer("sparse_tokenize_ms"):
        tokens = [tokenize(row["text"] or "") for row in rows]
    with stage_timer("sparse_index_build_ms"):
        index = BM25Okapi(tokens) if rows else None
    with stage_timer("sparse_snapshot_prepare_ms"):
        # No full text or tokenized corpus is retained. Freeze BM25's containers;
        # rank-bm25 get_scores allocates its query arrays and only reads these.
        if index is not None:
            index.doc_freqs = tuple(MappingProxyType(freq) for freq in index.doc_freqs)
            index.idf = MappingProxyType(index.idf)
            index.doc_len = tuple(index.doc_len)
        metadata = tuple(
            SnapshotRow(
                row["id"],
                row["doc_id"],
                int(row["page"]),
                int(row["start_offset"]),
                int(row["end_offset"]),
                (row["text"] or "")[:SNIPPET_CHARS] + ("…" if len(row["text"] or "") > SNIPPET_CHARS else ""),
            )
            for row in rows
        )
        estimate = deep_size((metadata, index)) + 1024
        snapshot = SparseSnapshot(metadata, index, estimate)
    _note_rss("build_after")
    return snapshot


def _note_rss(stage):
    from app.search.sparse_cache import process_rss_bytes

    value = process_rss_bytes()
    if value is not None:
        note_meta("sparse_rss_" + stage + "_bytes", value)


def _owner_clause(owner_id: UUID | None) -> tuple[str, dict[str, str]]:
    if owner_id is None:
        return "", {}
    return (
        "AND doc_id IN (SELECT id FROM documents WHERE owner_id = CAST(:owner_id AS uuid))",
        {"owner_id": str(owner_id)},
    )
