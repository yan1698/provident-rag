"""Persist MeteredClient calls when a bind_call_context() is active."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from app.core.log import get_logger
from app.core.request_id import get_request_id, get_run_id, set_run_id, set_step_index
from app.llm.base import LLMUsage

_log = get_logger("llm.ledger")


@dataclass
class CallContext:
    user_id: UUID | None = None
    run_id: UUID | None = None
    trace_id: UUID | None = None


_CTX: ContextVar[CallContext | None] = ContextVar("provident_llm_call_ctx", default=None)
_HOOK: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("provident_llm_call_hook", default=None)


@contextmanager
def bind_call_context(
    *,
    user_id: UUID | None = None,
    run_id: UUID | None = None,
    trace_id: UUID | None = None,
) -> Iterator[CallContext]:
    current = _CTX.get() or CallContext()
    ctx = CallContext(
        user_id=user_id if user_id is not None else current.user_id,
        run_id=run_id if run_id is not None else current.run_id,
        trace_id=trace_id if trace_id is not None else current.trace_id,
    )
    token = _CTX.set(ctx)
    prev_run = get_run_id()
    if ctx.run_id is not None:
        set_run_id(str(ctx.run_id))
    try:
        yield ctx
    finally:
        set_run_id(prev_run)
        set_step_index(None)
        _CTX.reset(token)


@contextmanager
def capture_calls(bucket: list[dict[str, Any]]) -> Iterator[None]:
    def _hook(payload: dict[str, Any]) -> None:
        bucket.append(payload)

    token = _HOOK.set(_hook)
    try:
        yield
    finally:
        _HOOK.reset(token)


def record_llm_call(usage: LLMUsage, *, status: str = "ok") -> None:
    payload = {
        "provider": usage.provider or "",
        "model": usage.model or "",
        "purpose": usage.purpose or "",
        "prompt_tokens": int(usage.prompt_tokens or 0),
        "completion_tokens": int(usage.completion_tokens or 0),
        "latency_ms": int(usage.latency_ms or 0),
        "status": status or "ok",
        "request_id": get_request_id(),
    }
    hook = _HOOK.get()
    if hook is not None:
        hook(payload)
        return
    ctx = _CTX.get()
    if ctx is None:
        return
    try:
        _insert_row(payload, ctx)
    except Exception:  # noqa: BLE001 — ledger must never break generation
        _log.exception("llm_calls insert failed")


def attach_call_refs(*, run_id: UUID | None = None, trace_id: UUID | None = None) -> None:
    rid = get_request_id()
    if not rid or (run_id is None and trace_id is None):
        return
    try:
        from sqlalchemy import update

        from app.db.session import SessionLocal
        from app.models.llm_call import LlmCall

        with SessionLocal() as db:
            if run_id is not None:
                db.execute(
                    update(LlmCall).where(LlmCall.request_id == rid, LlmCall.run_id.is_(None)).values(run_id=run_id)
                )
            if trace_id is not None:
                db.execute(
                    update(LlmCall)
                    .where(LlmCall.request_id == rid, LlmCall.trace_id.is_(None))
                    .values(trace_id=trace_id)
                )
            db.commit()
    except Exception:  # noqa: BLE001
        _log.exception("llm_calls attach failed")


def _insert_row(payload: dict[str, Any], ctx: CallContext) -> None:
    from app.db.session import SessionLocal
    from app.models.llm_call import LlmCall

    with SessionLocal() as db:
        db.add(
            LlmCall(
                id=uuid4(),
                user_id=ctx.user_id,
                run_id=ctx.run_id,
                trace_id=ctx.trace_id,
                provider=payload["provider"],
                model=payload["model"],
                purpose=payload["purpose"],
                prompt_tokens=payload["prompt_tokens"],
                completion_tokens=payload["completion_tokens"],
                latency_ms=payload["latency_ms"],
                status=payload["status"],
                request_id=payload["request_id"],
            )
        )
        db.commit()


def selfcheck() -> None:
    """Insert one stub row and read it back. Run inside the API container."""
    from sqlalchemy import text

    from app.core.request_id import set_request_id
    from app.db.session import engine

    rid = f"ledger-selfcheck-{uuid4().hex[:12]}"
    set_request_id(rid)
    with bind_call_context():
        record_llm_call(
            LLMUsage(
                provider="stub",
                model="stub",
                purpose="answer",
                prompt_tokens=3,
                completion_tokens=2,
                latency_ms=1,
            ),
            status="ok",
        )
    with engine.connect() as conn:
        found = conn.execute(
            text("SELECT count(*) FROM llm_calls WHERE request_id = :rid"),
            {"rid": rid},
        ).scalar()
    if int(found or 0) != 1:
        raise SystemExit(f"expected 1 llm_calls row for selfcheck, got {found}")
    print("llm_calls persist ok")


if __name__ == "__main__":
    selfcheck()
