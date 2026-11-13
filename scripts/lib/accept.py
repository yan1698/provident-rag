"""Tiny helpers shared by Day acceptance scripts."""

from __future__ import annotations


def assert_ok(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def version_tuple(raw: str) -> tuple[int, ...]:
    parts = []
    for piece in str(raw).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)
