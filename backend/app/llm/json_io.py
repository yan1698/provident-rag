"""Tolerant JSON parsing for model output.

Every lane asks the model for one JSON object, and a single stray newline or
unescaped quote used to throw the whole draft away and silently degrade to an
extractive collage. Repair what is safely repairable, then give up loudly.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\uff02": '"',
}
_CONTROL_MAP = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
_CLOSERS = set(",:}]")

RETRY_HINT = (
    "上一次输出不是合法 JSON。请只输出一个 JSON 对象："
    "字符串内不要换行，内部引号要转义，不要 Markdown 围栏，不要注释和多余文字。"
)


def _slice_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _escape_inside_strings(text: str) -> str:
    """Escape raw control chars and inner quotes that would break json.loads."""
    out: list[str] = []
    in_string = False
    escaped = False
    for index, ch in enumerate(text):
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = in_string
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
                continue
            nxt = next((c for c in text[index + 1 :] if not c.isspace()), "")
            if nxt in _CLOSERS or nxt == "":
                in_string = False
                out.append(ch)
            else:
                out.append('\\"')
            continue
        if in_string and ch in _CONTROL_MAP:
            out.append(_CONTROL_MAP[ch])
            continue
        out.append(ch)
    return "".join(out)


def _close_truncated(text: str) -> str:
    """Best-effort close of a response cut off by max_output_tokens."""
    in_string = False
    escaped = False
    stack: list[str] = []
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = in_string
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    tail = '"' if in_string else ""
    for opener in reversed(stack):
        tail += "}" if opener == "{" else "]"
    return text + tail


def repair_json_text(raw: str) -> str:
    text = _slice_object(_FENCE_RE.sub("", (raw or "").strip()))
    for bad, good in _SMART_QUOTES.items():
        text = text.replace(bad, good)
    text = _escape_inside_strings(text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return _close_truncated(text)


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Strict parse first, then a repaired parse. ``None`` when both fail."""
    text = _FENCE_RE.sub("", (raw or "").strip())
    for candidate in (text, _slice_object(text), repair_json_text(raw)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_complete_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Lossless local recovery only; never complete a truncated model draft.

    Accept an object, a JSON-encoded object string, or one extra escaped string
    layer with the outer quotes omitted (the observed provider response).
    A successful decode must still pass the caller's schema/source contracts.
    """
    text = _FENCE_RE.sub("", (raw or "").strip())
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data, "strict"
        if isinstance(data, str):
            inner = json.loads(data)
            if isinstance(inner, dict):
                return inner, "json_string"
    except (json.JSONDecodeError, ValueError):
        pass
    if text.startswith(r"{\"") and text.endswith("}"):
        try:
            inner = json.loads('"' + text + '"')
            data = json.loads(inner)
            if isinstance(data, dict):
                return data, "escaped_object"
        except (json.JSONDecodeError, ValueError):
            pass
    return None, "unrecoverable"


def completion_text(choice) -> str:
    """Prefer content; reasoning models often spend the budget on reasoning_content."""
    message = getattr(choice, "message", None)
    text = (getattr(message, "content", None) or "").strip()
    if text:
        return text
    return (getattr(message, "reasoning_content", None) or "").strip()
