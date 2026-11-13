"""JSON logs on stderr. No structlog — keeps the image from rebuilding."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.core.request_id import get_request_id, get_run_id, get_step_index


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid
        run_id = get_run_id() or getattr(record, "run_id", "")
        if run_id:
            payload["run_id"] = str(run_id)
        step = get_step_index()
        if step is None:
            step = getattr(record, "step_index", None)
        if step is not None:
            payload["step_index"] = step
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("provident")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False


def get_logger(name: str = "provident") -> logging.Logger:
    if not name.startswith("provident"):
        name = f"provident.{name}"
    return logging.getLogger(name)
