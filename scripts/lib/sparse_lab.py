"""Disposable local PostgreSQL schema for BM25 correctness/performance, no public writes."""

from __future__ import annotations

import importlib.util
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]


class SparseLab:
    def __init__(self):
        url = make_url(Settings().sqlalchemy_database_url)
        if url.get_backend_name() != "postgresql" or url.host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("lab requires existing loopback PostgreSQL; no remote targets")
        self.schema = "bm25_r2_" + uuid.uuid4().hex[:12]
        self.admin = create_engine(url, poolclass=NullPool, connect_args={"connect_timeout": 3})
        self.engine = create_engine(
            url,
            pool_size=5,
            max_overflow=0,
            connect_args={
                "connect_timeout": 3,
                "options": f"-c search_path={self.schema} -c statement_timeout=10000 -c lock_timeout=5000",
            },
        )
        path = ROOT / "backend/alembic/versions/0011_sparse_generation.py"
        spec = importlib.util.spec_from_file_location("sparse_migration", path)
        self.migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.migration)

    def __enter__(self):
        with self.admin.begin() as db:
            db.execute(text(f'CREATE SCHEMA "{self.schema}"'))
        try:
            with self.engine.begin() as db:
                for ddl in (
                    "CREATE TABLE users (id uuid PRIMARY KEY)",
                    "CREATE TABLE collections (id uuid PRIMARY KEY)",
                    "CREATE TABLE documents (id uuid PRIMARY KEY, owner_id uuid REFERENCES users(id) ON DELETE CASCADE, collection_id uuid REFERENCES collections(id) ON DELETE SET NULL)",
                    "CREATE TABLE chunks (id uuid PRIMARY KEY, doc_id uuid REFERENCES documents(id) ON DELETE CASCADE, page int NOT NULL, start_offset int NOT NULL, end_offset int NOT NULL, text text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), embedding text, embedded_at timestamp)",
                ):
                    db.execute(text(ddl))
                self.migrate(db, "upgrade")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def migrate(self, db, direction):
        with patch.object(self.migration, "op", Operations(MigrationContext.configure(db))):
            getattr(self.migration, direction)()

    def __exit__(self, *args):
        self.engine.dispose()
        if not re.fullmatch(r"bm25_r2_[0-9a-f]{12}", self.schema):
            raise ValueError("refusing cleanup outside this isolated lab")
        with self.admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{self.schema}" CASCADE'))
        self.admin.dispose()

    def seed(self, count=64, repeats=8):
        with self.engine.begin() as db:
            db.execute(text("TRUNCATE chunks, documents, users, collections CASCADE"))
            db.execute(text("INSERT INTO users VALUES (:a),(:b)"), {"a": uuid.UUID(int=1), "b": uuid.UUID(int=2)})
            db.execute(
                text("INSERT INTO collections VALUES (:a),(:b)"), {"a": uuid.UUID(int=11), "b": uuid.UUID(int=12)}
            )
            docs, chunks = [], []
            for i in range(count):
                docs.append(
                    {
                        "id": uuid.UUID(int=1000 + i),
                        "owner": uuid.UUID(int=1 + i % 2),
                        "collection": uuid.UUID(int=11 + i % 2) if i % 3 else None,
                    }
                )
                chunks.append(
                    {
                        "id": uuid.UUID(int=10000 + i),
                        "doc": uuid.UUID(int=1000 + i),
                        "body": (("租赁合同 租金期限 押金 " if i % 5 == 0 else "天文观测 星空数据 科学 ") + str(i))
                        * repeats,
                        "created": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=i),
                    }
                )
            db.execute(text("INSERT INTO documents VALUES (:id,:owner,:collection)"), docs)
            db.execute(
                text(
                    "INSERT INTO chunks (id,doc_id,page,start_offset,end_offset,text,created_at) VALUES (:id,:doc,1,0,100,:body,:created)"
                ),
                chunks,
            )

    def execute(self, sql, params=None):
        with self.engine.begin() as db:
            return db.execute(text(sql), params or {})
