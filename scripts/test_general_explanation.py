"""Free contracts: real SQLite run/session records; mocked model and worker lease."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from sqlalchemy import create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session
from fastapi import HTTPException
from pydantic import ValidationError
from app.models.agent import AgentRun, AgentSession, AgentSessionState, AgentIdempotency
from app.agent.service import create_agent_run, _recent_session_runs
from app.agent.general import execute_explanation
from app.agent.session_runtime import session_snapshot
from app.api.agent import AgentRunRequest, _to_out
from app.api.query import QueryRequest, query
from app.core.config import Settings
from app.generate.general import explain, source_mode
from app.llm import LLMError, LLMResponse, LLMUsage


@compiles(JSONB, "sqlite")
def jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


def main():
    settings = Settings(_env_file=None, answer_llm_provider="stub", task_llm_provider="stub", allow_llm_stub=True)
    assert AgentRunRequest(goal=_TEST_MATERIALS["text_012"]).answer_source == "document_only"
    assert QueryRequest(query=_TEST_MATERIALS["text_013"]).answer_source == "document_only"
    for cls, args in [
        (AgentRunRequest, {"goal": _TEST_MATERIALS["text_007"]}),
        (QueryRequest, {"query": _TEST_MATERIALS["text_008"]}),
    ]:
        try:
            cls(**args, answer_source="mixed")
            raise AssertionError("mixed accepted")
        except ValidationError:
            pass
    assert source_mode(type("Old", (), {"outline": None})()) == "document_only"

    class Model:
        def generate(self, **kw):
            assert "secret-document" not in kw["prompt"]
            assert kw["purpose"] == "general_explanation"
            return LLMResponse(text=_TEST_MATERIALS["text_014"], usage=LLMUsage(provider="stub"))

    with patch("app.generate.general.get_llm_client", return_value=Model()):
        result = explain("why", topic="backup", settings=settings)
        assert not result.citations and not result.evidence
    for bad in ("fake [S1]", "fake S2", ""):
        with patch("app.generate.general.get_llm_client") as model:
            model.return_value.generate.return_value = LLMResponse(text=bad, usage=LLMUsage())
            try:
                explain("why", settings=settings)
                raise AssertionError("source marker accepted")
            except LLMError:
                pass
    engine = create_engine("sqlite://")
    for model in (AgentSession, AgentRun, AgentSessionState, AgentIdempotency):
        model.__table__.create(engine)
    owner, other = uuid.uuid4(), uuid.uuid4()
    with Session(engine, expire_on_commit=False) as db, ExitStack() as stack:
        stack.enter_context(patch("app.agent.lifecycle.lock_owner"))
        stack.enter_context(patch("app.agent.session_runtime._lock_sessions"))
        stack.enter_context(patch("app.core.config.get_settings", return_value=settings))
        stack.enter_context(patch("app.queue.client.writer_owns", return_value=True))
        stack.enter_context(
            patch("app.agent.service._committed_status", side_effect=lambda db, key: db.get(AgentRun, key).status)
        )
        cancelled = stack.enter_context(patch("app.queue.client.is_cancelled", return_value=False))
        stack.enter_context(patch("app.llm.ledger.record_llm_call"))
        # Any accidental retrieval, scope routing or memory promotion fails the test.
        for target in (
            "app.agent.service._resolve_scope",
            "app.memory.store.persist_turn_memory",
            "app.generate.service.answer_query",
            "app.verify.service.verify_answer",
        ):
            stack.enter_context(patch(target, side_effect=AssertionError(target)))
        session = AgentSession(user_id=owner, title=_TEST_MATERIALS["text_009"])
        db.add(session)
        db.flush()
        source = AgentRun(
            user_id=owner,
            session_id=session.id,
            goal=_TEST_MATERIALS["text_010"],
            answer=_TEST_MATERIALS["text_011"],
            status="finished",
            provider="stub",
            llm_model="stub",
            planner="rules",
            task_kind="factoid",
            steps=[],
            outline={"operation_anchor": {"run_id": "original"}},
            scope={"mode": "hard", "collection_ids": [str(uuid.uuid4())]},
            created_at=datetime.now(timezone.utc),
        )
        db.add(source)
        db.commit()

        def create(**extra):
            return create_agent_run(
                db,
                "why keep connection",
                owner_id=owner,
                session_id=session.id,
                settings=settings,
                answer_source="general_explanation",
                source_run_id=source.id,
                **extra,
            )

        for changes in ({"owner_id": other}, {"session_id": uuid.uuid4()}):
            args = dict(
                owner_id=owner,
                session_id=session.id,
                settings=settings,
                answer_source="general_explanation",
                source_run_id=source.id,
            )
            args.update(changes)
            try:
                create_agent_run(db, "why", **args)
                raise AssertionError(_TEST_MATERIALS["text_015"])
            except ValueError:
                db.rollback()
        try:
            create(selected_context="secret-document")
            raise AssertionError("unselected context accepted")
        except ValueError:
            db.rollback()
        row = create(idempotency_key="general-key", selected_context="Keep connection open")
        assert row.scope == source.scope
        assert "memory" not in row.outline
        assert create(idempotency_key="general-key", selected_context="Keep connection open").id == row.id
        try:
            create_agent_run(
                db,
                row.goal,
                owner_id=owner,
                session_id=session.id,
                settings=settings,
                idempotency_key="general-key",
                answer_source="document_only",
                source_run_id=source.id,
            )
            raise AssertionError("mode replay accepted")
        except ValueError:
            db.rollback()
        before = session_snapshot(db, session.id)["checkpoint_json"]
        row.status = "running"
        db.commit()
        result = execute_explanation(db, row, settings=settings)
        assert result.status == "finished" and result.verification is None, (result.status, result.answer)
        assert result.outline["source_run_id"] == str(source.id)
        snap = session_snapshot(db, session.id)
        assert snap == before and snap["checkpoint_json"] is None
        assert [r.id for r in _recent_session_runs(db, session.id, owner_id=owner)] == [source.id]
        from app.agent.memory import compose_session_memory, select_answer_anchor

        assert compose_session_memory([source, result]) == compose_session_memory([source])
        assert select_answer_anchor([source, result]) == select_answer_anchor([source])
        out = _to_out(result)
        assert out.answer_source == "general_explanation" and not out.outline.evidence
        # Reading the persisted result after session expiration is the refresh contract.
        db.expire_all()
        assert _to_out(db.get(AgentRun, result.id)).answer_source == "general_explanation"
        for kind in ("timeout", "unavailable", "cancel"):
            row = create()
            row.status = "running"
            db.commit()
            prior = session_snapshot(db, session.id)["checkpoint_json"]
            if kind == "cancel":
                cancelled.return_value = True
            with patch(_TEST_MATERIALS["text_016"], side_effect=LLMError("failure", kind=kind)):
                result = execute_explanation(db, row, settings=settings)
            assert result.status == ("cancelled" if kind == "cancel" else "error")
            assert result.verification is None and source_mode(result) == "general_explanation"
            assert session_snapshot(db, session.id) == prior
            cancelled.return_value = False
        row = create()
        row.status = "running"
        db.commit()
        pending = session_snapshot(db, session.id)
        with patch("app.queue.client.writer_owns", return_value=False):
            result = execute_explanation(db, row, settings=settings)
        assert result.status == "running" and not result.answer
        assert session_snapshot(db, session.id) == pending
        # Simulate the existing recovery owner closing that abandoned turn.
        from app.agent.session_runtime import complete_session_turn

        complete_session_turn(db, session.id, cancelled=True)
        row.status = "cancelled"
        db.commit()
        # Ordinary document follow-up sees only the original document memory and scope.
        row = create_agent_run(
            db,
            "explain step 1",
            owner_id=owner,
            session_id=session.id,
            settings=settings,
            planner="rules",
            task_kind="factoid",
            source_run_id=source.id,
        )
        assert source_mode(row) == "document_only" and row.scope == source.scope
        assert "stub-provider" not in row.outline.get("memory", "")
        assert db.query(AgentRun).count() == 7
    # /query general branch is authorized before any model or scope read.
    with patch("app.trace.service.get_trace", return_value=None), patch(_TEST_MATERIALS["text_002"]) as model:
        try:
            query(
                QueryRequest(
                    query=_TEST_MATERIALS["text_021"], answer_source="general_explanation", source_trace_id=uuid.uuid4()
                ),
                db=None,
                settings=settings,
                user=None,
                _=None,
            )
            raise AssertionError("missing source accepted")
        except HTTPException as exc:
            assert exc.status_code == 404
        model.assert_not_called()
    from types import SimpleNamespace
    from app.corpus.scope import Scope
    from app.generate.service import AnswerResult

    trace = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=owner,
        query=_TEST_MATERIALS["text_003"],
        answer=_TEST_MATERIALS["text_004"],
        diagnostics={},
    )
    user = SimpleNamespace(id=owner)
    with (
        patch("app.trace.service.get_trace", return_value=trace),
        patch("app.api.query.save_trace") as save,
        patch(
            _TEST_MATERIALS["text_005"],
            return_value=AnswerResult(
                query=_TEST_MATERIALS["text_022"],
                answer=_TEST_MATERIALS["text_023"],
                provider="stub",
                citations=[],
                evidence=[],
            ),
        ) as model,
        patch("app.corpus.router.resolve_scope", side_effect=AssertionError(_TEST_MATERIALS["text_017"])),
        patch("app.api.query.verify_answer", side_effect=AssertionError(_TEST_MATERIALS["text_018"])),
        patch("app.llm.ledger.attach_call_refs"),
    ):
        save.return_value = SimpleNamespace(id=uuid.uuid4())
        result = query(
            QueryRequest(
                query=_TEST_MATERIALS["text_019"],
                answer_source="general_explanation",
                source_trace_id=trace.id,
                verify=True,
                persist=True,
            ),
            db=None,
            settings=settings,
            user=user,
            _=None,
        )
        assert result.answer_source == "general_explanation" and result.verification is None
        assert result.citations == result.evidence == []
        assert model.call_args.kwargs["topic"] == trace.query
        assert save.call_args.kwargs["diagnostics"]["source_trace_id"] == str(trace.id)
        assert save.call_args.kwargs["verification"] is None
    with (
        patch("app.corpus.router.resolve_scope", return_value=Scope()),
        patch(
            "app.api.query.answer_query",
            return_value=AnswerResult(
                query=_TEST_MATERIALS["text_024"],
                answer=_TEST_MATERIALS["text_025"],
                provider="stub",
                citations=[],
                evidence=[],
            ),
        ),
        patch(_TEST_MATERIALS["text_006"], side_effect=AssertionError("automatic model fallback")),
        patch("app.verify.structure.load_citation_rows", return_value={}),
    ):
        result = query(
            QueryRequest(query=_TEST_MATERIALS["text_020"], verify=False, persist=False),
            db=None,
            settings=settings,
            user=user,
            _=None,
        )
        assert result.answer_source == "document_only" and result.answer == "No document evidence"
    print(_TEST_MATERIALS["text_001"])


if __name__ == "__main__":
    main()
