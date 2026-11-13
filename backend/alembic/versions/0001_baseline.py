"""Baseline schema matching the live 0.32 database.

Revision ID: 0001baseline
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE users (
            id            UUID PRIMARY KEY,
            username      VARCHAR(64) NOT NULL UNIQUE,
            display_name  VARCHAR(64) NOT NULL DEFAULT '',
            password_hash VARCHAR(255) NOT NULL,
            is_active     BOOLEAN NOT NULL DEFAULT TRUE,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_login_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE TABLE documents (
            id                 UUID PRIMARY KEY,
            owner_id           UUID REFERENCES users(id) ON DELETE CASCADE,
            filename           TEXT NOT NULL,
            content_type       VARCHAR(128),
            byte_size          INTEGER NOT NULL,
            file_hash          VARCHAR(64) NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            graph_extracted_at TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE INDEX idx_documents_owner_id ON documents (owner_id)")
    op.execute(
        """
        CREATE UNIQUE INDEX uq_documents_owner_hash
        ON documents (owner_id, file_hash) WHERE owner_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_documents_anon_hash
        ON documents (file_hash) WHERE owner_id IS NULL
        """
    )
    op.execute(
        """
        CREATE TABLE chunks (
            id           UUID PRIMARY KEY,
            doc_id       UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page         INTEGER NOT NULL DEFAULT 1,
            start_offset INTEGER NOT NULL,
            end_offset   INTEGER NOT NULL,
            text         TEXT NOT NULL,
            token_count  INTEGER NOT NULL,
            chunk_hash   VARCHAR(64) NOT NULL,
            embedding    vector(384),
            embedded_at  TIMESTAMPTZ,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_chunks_doc_hash UNIQUE (doc_id, chunk_hash)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX idx_chunks_embedding_cosine
        ON chunks USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        """
        CREATE TABLE query_traces (
            id                   UUID PRIMARY KEY,
            user_id              UUID REFERENCES users(id) ON DELETE CASCADE,
            query                TEXT NOT NULL,
            answer               TEXT NOT NULL,
            provider             VARCHAR(32) NOT NULL,
            llm_model            VARCHAR(128) NOT NULL DEFAULT '',
            eval_id              VARCHAR(64),
            top_k                INTEGER,
            citations            JSONB NOT NULL DEFAULT '[]'::jsonb,
            evidence             JSONB NOT NULL DEFAULT '[]'::jsonb,
            verification         JSONB,
            usage                JSONB,
            coverage_rate        DOUBLE PRECISION,
            hallucination_rate   DOUBLE PRECISION,
            claim_count          INTEGER,
            supported_count      INTEGER,
            hallucination_count  INTEGER,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_query_traces_created_at ON query_traces (created_at)")
    op.execute("CREATE INDEX idx_query_traces_eval_id ON query_traces (eval_id)")
    op.execute("CREATE INDEX idx_query_traces_user_id ON query_traces (user_id)")
    op.execute(
        """
        CREATE TABLE agent_runs (
            id             UUID PRIMARY KEY,
            user_id        UUID REFERENCES users(id) ON DELETE CASCADE,
            goal           TEXT NOT NULL,
            status         VARCHAR(16) NOT NULL,
            answer         TEXT NOT NULL DEFAULT '',
            provider       VARCHAR(32) NOT NULL DEFAULT '',
            llm_model      VARCHAR(128) NOT NULL DEFAULT '',
            planner        VARCHAR(16) NOT NULL DEFAULT 'rules',
            step_count     INTEGER NOT NULL DEFAULT 0,
            steps          JSONB NOT NULL DEFAULT '[]'::jsonb,
            verification   JSONB,
            task_kind      VARCHAR(16),
            outline        JSONB,
            gaps           JSONB,
            catalog_digest JSONB,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            heartbeat_at   TIMESTAMPTZ,
            finished_at    TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE INDEX idx_agent_runs_created_at ON agent_runs (created_at)")
    op.execute("CREATE INDEX idx_agent_runs_user_id ON agent_runs (user_id)")
    op.execute(
        """
        CREATE TABLE query_tags (
            id         UUID PRIMARY KEY,
            user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
            name       VARCHAR(64) NOT NULL,
            color      VARCHAR(16) NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_query_tags_user_id ON query_tags (user_id)")
    op.execute(
        """
        CREATE UNIQUE INDEX uq_query_tags_user_name
        ON query_tags (user_id, name) WHERE user_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_query_tags_anon_name
        ON query_tags (name) WHERE user_id IS NULL
        """
    )
    op.execute(
        """
        CREATE TABLE query_trace_tags (
            trace_id UUID NOT NULL REFERENCES query_traces(id) ON DELETE CASCADE,
            tag_id   UUID NOT NULL REFERENCES query_tags(id) ON DELETE CASCADE,
            PRIMARY KEY (trace_id, tag_id)
        )
        """
    )
    op.execute("CREATE INDEX idx_query_trace_tags_tag_id ON query_trace_tags (tag_id)")
    op.execute(
        """
        CREATE TABLE query_feedback (
            id            UUID PRIMARY KEY,
            user_id       UUID REFERENCES users(id) ON DELETE CASCADE,
            trace_id      UUID,
            agent_run_id  UUID,
            rating        VARCHAR(8) NOT NULL,
            comment       TEXT NOT NULL DEFAULT '',
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_feedback_trace_id ON query_feedback (trace_id)")
    op.execute("CREATE INDEX idx_feedback_agent_run_id ON query_feedback (agent_run_id)")
    op.execute("CREATE INDEX idx_feedback_created_at ON query_feedback (created_at)")
    op.execute("CREATE INDEX idx_query_feedback_user_id ON query_feedback (user_id)")
    op.execute(
        """
        CREATE TABLE graph_entities (
            id            UUID PRIMARY KEY,
            owner_id      UUID REFERENCES users(id) ON DELETE CASCADE,
            name          TEXT NOT NULL,
            norm_name     VARCHAR(256) NOT NULL,
            kind          VARCHAR(32) NOT NULL,
            aliases       JSONB NOT NULL DEFAULT '[]'::jsonb,
            mention_count INTEGER NOT NULL DEFAULT 0,
            merged_into   UUID REFERENCES graph_entities(id) ON DELETE SET NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_graph_entities_owner ON graph_entities (owner_id)")
    op.execute(
        """
        CREATE UNIQUE INDEX uq_graph_entities_owner_norm
        ON graph_entities (owner_id, kind, norm_name) WHERE owner_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_graph_entities_anon_norm
        ON graph_entities (kind, norm_name) WHERE owner_id IS NULL
        """
    )
    op.execute(
        """
        CREATE TABLE graph_edges (
            id             UUID PRIMARY KEY,
            owner_id       UUID REFERENCES users(id) ON DELETE CASCADE,
            src_id         UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            dst_id         UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            relation       VARCHAR(32) NOT NULL,
            confidence     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            source         VARCHAR(16) NOT NULL DEFAULT 'rules',
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_graph_edges_owner ON graph_edges (owner_id)")
    op.execute(
        """
        CREATE UNIQUE INDEX uq_graph_edges_owner_spo
        ON graph_edges (owner_id, src_id, relation, dst_id) WHERE owner_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_graph_edges_anon_spo
        ON graph_edges (src_id, relation, dst_id) WHERE owner_id IS NULL
        """
    )
    op.execute(
        """
        CREATE TABLE graph_mentions (
            id          UUID PRIMARY KEY,
            edge_id     UUID REFERENCES graph_edges(id) ON DELETE CASCADE,
            entity_id   UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
            doc_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            chunk_hash  VARCHAR(64) NOT NULL,
            quote       TEXT NOT NULL,
            start_char  INTEGER NOT NULL DEFAULT 0,
            end_char    INTEGER NOT NULL DEFAULT 0,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_graph_mentions_chunk ON graph_mentions (chunk_id)")
    op.execute("CREATE INDEX idx_graph_mentions_hash ON graph_mentions (chunk_hash)")
    op.execute("CREATE INDEX idx_graph_mentions_doc ON graph_mentions (doc_id)")
    op.execute("CREATE INDEX idx_graph_mentions_edge ON graph_mentions (edge_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS graph_mentions")
    op.execute("DROP TABLE IF EXISTS graph_edges")
    op.execute("DROP TABLE IF EXISTS graph_entities")
    op.execute("DROP TABLE IF EXISTS query_feedback")
    op.execute("DROP TABLE IF EXISTS query_trace_tags")
    op.execute("DROP TABLE IF EXISTS query_tags")
    op.execute("DROP TABLE IF EXISTS agent_runs")
    op.execute("DROP TABLE IF EXISTS query_traces")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS users")
