"""Request / agent ids in contextvars so logs and llm_calls share them."""

from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

_REQUEST_ID: ContextVar[str] = ContextVar("provident_request_id", default="")
_RUN_ID: ContextVar[str] = ContextVar("provident_run_id", default="")
_STEP_INDEX: ContextVar[int | None] = ContextVar("provident_step_index", default=None)


def get_request_id() -> str:
    return _REQUEST_ID.get() or ""


def set_request_id(value: str) -> None:
    _REQUEST_ID.set((value or "").strip())


def get_run_id() -> str:
    return _RUN_ID.get() or ""


def set_run_id(value: str) -> None:
    _RUN_ID.set((value or "").strip())


def get_step_index() -> int | None:
    return _STEP_INDEX.get()


def set_step_index(value: int | None) -> None:
    _STEP_INDEX.set(value)


def new_request_id() -> str:
    return uuid4().hex


class RequestIdMiddleware:
    """Pure ASGI middleware — no Starlette import, so offline scripts can load this module."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = ""
        for key, value in scope.get("headers") or []:
            if key.lower() == b"x-request-id":
                incoming = value.decode("latin-1").strip()
                break
        rid = incoming or new_request_id()
        token = _REQUEST_ID.set(rid)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", rid.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _REQUEST_ID.reset(token)
