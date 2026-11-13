"""Filename usability. A junk name is not a topic signal."""

from __future__ import annotations

import re
from pathlib import Path

_JUNK = re.compile(
    r"^(未命名|新建文档|新建 microsoft word 文档|document|doc|scan|img|image|wx|screenshot|屏幕截图)$",
    re.IGNORECASE,
)
_COPY = re.compile(r"(副本|\(\d+\))\s*$")
_HEX = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)
_DIGITS = re.compile(r"^[\d_\-\s.]+$")


def filename_stem(filename: str) -> str:
    return Path(filename or "").stem.strip()


def filename_usable(filename: str) -> bool:
    stem = filename_stem(filename)
    if len(stem) < 2:
        return False
    if _HEX.fullmatch(stem) or _DIGITS.fullmatch(stem) or _JUNK.fullmatch(stem):
        return False
    stripped = _COPY.sub("", stem).strip()
    if stripped != stem and len(stripped) < 2:
        return False
    return True
