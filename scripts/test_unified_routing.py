"""Free route API integration, admission and document target contracts. No providers."""

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from fastapi import FastAPI
import asyncio
import json
from app.api.query import router
from app.auth.deps import get_current_user
from app.db.session import get_db
from app.agent.routing import decide, validate_scope, validate_target
from app.agent.store import SqlAgentStore
from app.corpus.scope import Scope
from app.corpus.catalog import CatalogBuild, CatalogDocument


def main():
    owner, other, target, library = (uuid.uuid4() for _ in range(4))
    with patch("app.llm.get_router_llm_client", side_effect=AssertionError("paid classifier")):
        assert decide(_TEST_MATERIALS["text_005"])["actual"] == "quick"
        assert decide(_TEST_MATERIALS["text_006"])["actual"] == "synthesis"
        assert decide(_TEST_MATERIALS["text_007"], "quick")["actual"] == "quick"
        assert decide(_TEST_MATERIALS["text_008"])["fallback_reason"] == "reading_target_missing"
        assert decide(_TEST_MATERIALS["text_009"], target_document_id=target)["actual"] == "digest"
        assert (
            decide(_TEST_MATERIALS["text_010"], answer_source="general_explanation")["actual"] == "general_explanation"
        )
    with patch("app.agent.routing.classify_task_kind", side_effect=TimeoutError):
        assert decide("unknown")["fallback_reason"] == "classifier_failed"
    with patch("app.corpus.library.list_collections", return_value=[SimpleNamespace(id=library)]):
        validate_scope(None, owner, "hard", [library])
        for ids in ([], [target], [library, target]):
            try:
                validate_scope(None, owner, "hard", ids)
                raise AssertionError("scope widened")
            except ValueError:
                pass

    class DB:
        document = SimpleNamespace(owner_id=owner, collection_id=library)

        def get(self, *args):
            return self.document

    db = DB()
    validate_target(db, owner, target, Scope(mode="hard", collection_ids=(str(library),)))
    for uid, scope in [(other, Scope()), (owner, Scope(mode="hard", collection_ids=(str(other),)))]:
        try:
            validate_target(db, uid, target, scope)
            raise AssertionError("foreign target accepted")
        except ValueError:
            pass
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=owner)
    app.dependency_overrides[get_db] = lambda: db

    async def request(payload):
        messages = []

        async def receive():
            return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

        async def send(message):
            messages.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/query/route",
                "raw_path": b"/query/route",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "server": ("test", 80),
                "client": ("test", 1),
            },
            receive,
            send,
        )
        body = b"".join(m.get("body", b"") for m in messages)
        return SimpleNamespace(status_code=messages[0]["status"], json=lambda: json.loads(body))

    with patch("app.corpus.router.resolve_scope", return_value=Scope()):

        def route(**kw):
            return asyncio.run(request({"query": _TEST_MATERIALS["text_013"], "scope_mode": "all", **kw}))

        assert route().json()["actual"] == "quick"
        assert route(query=_TEST_MATERIALS["text_014"]).json()["actual"] == "synthesis"
        assert route(analysis_mode="digest").status_code == 400
        assert route(analysis_mode="digest", target_document_id=str(target)).json()["actual"] == "digest"
        assert route(answer_source="mixed").status_code == 422
        assert route(answer_source="general_explanation").json()["actual"] == "general_explanation"
        db.document.owner_id = other
        assert route(analysis_mode="digest", target_document_id=str(target)).status_code == 400
    store = SqlAgentStore(None, owner_id=owner)
    store.target_document_id = str(target)
    assert store.load_doc_chunks(str(other)) == []  # Refused before any SQL.
    with patch(
        "app.corpus.catalog.load_corpus_catalog",
        return_value=CatalogBuild(
            documents=(CatalogDocument(str(target), "selected"), CatalogDocument(str(other), "outside"))
        ),
    ) as load:
        assert [d.doc_id for d in store.load_catalog(goal=_TEST_MATERIALS["text_015"], max_sections=80).documents] == [
            str(target)
        ]
        assert load.call_args.kwargs["target_document_id"] == str(target)
    assert store.expand_graph([], goal=_TEST_MATERIALS["text_011"], settings=None)[0] == []
    # Real SQLite admission records exercise the new route metadata and replay boundary.
    from test_general_explanation import AgentRun, AgentSession, AgentSessionState, AgentIdempotency
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.agent.service import create_agent_run
    from app.core.config import Settings

    engine = create_engine("sqlite://")
    for model in (AgentRun, AgentSession, AgentSessionState, AgentIdempotency):
        model.__table__.create(engine)
    settings = Settings(_env_file=None, answer_llm_provider="stub", allow_llm_stub=True)
    with (
        Session(engine) as session,
        patch("app.agent.lifecycle.lock_owner"),
        patch("app.agent.session_runtime.acquire_session"),
        patch("app.agent.service._resolve_scope", return_value=Scope().to_dict()),
    ):

        def create(**kw):
            return create_agent_run(
                session,
                _TEST_MATERIALS["text_003"],
                owner_id=owner,
                settings=settings,
                analysis_mode="auto",
                idempotency_key="route-key",
                **kw,
            )

        row = create()
        assert row.task_kind == "synthesis" and row.outline["routing"]["actual"] == "synthesis"
        assert create().id == row.id
        try:
            create(target_document_id=target)
            raise AssertionError(_TEST_MATERIALS["text_004"])
        except ValueError as exc:
            assert _TEST_MATERIALS["text_012"] in str(exc)
        row.status = "finished"
        session.commit()
        follow = create_agent_run(
            session,
            _TEST_MATERIALS["text_002"],
            owner_id=owner,
            settings=settings,
            analysis_mode="quick",
            source_run_id=row.id,
            session_id=row.session_id,
        )
        assert follow.task_kind == "factoid"
        assert follow.outline["routing"]["actual"] == "agent_factoid"
        assert follow.scope == row.scope
    print(_TEST_MATERIALS["text_001"])


if __name__ == "__main__":
    main()
