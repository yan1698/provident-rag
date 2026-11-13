"""Normalize extracted text before chunking.

PDF lines are hard-wrapped; Markdown newlines are structure and stay intact.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u2060]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_PAGE_NO = re.compile(
    r"^(?:"
    r"第\s*\d{1,4}\s*页(?:\s*/\s*共\s*\d{1,4}\s*页)?"
    r"|[-—–]?\s*\d{1,4}\s*[-—–]"
    r"|page\s+\d{1,4}(?:\s*/\s*\d{1,4})?"
    r"|\d{1,4}\s*/\s*\d{1,4}"
    r")$",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^(?:[-*•·]|\d{1,3}[.、)])\s+\S")
_SENTENCE_END = re.compile(r"[。！？!?；;]$|(?<!\d)[.]$")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def decode_bytes(data: bytes) -> str:
    """UTF-8 first, then GB18030. UTF-16 BOM is honored."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def normalize_markup(text: str) -> str:
    """Unicode and whitespace cleanup that keeps Markdown structure."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def is_page_number(line: str) -> bool:
    return bool(_PAGE_NO.match(line.strip()))


def merge_hard_breaks(text: str) -> str:
    """Join PDF visual line wraps. Headings and list items stay their own blocks."""
    paragraphs: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            paragraphs.append(buf.strip())
        buf = ""

    for raw in text.split("\n"):
        line = re.sub(r"[ \t]{2,}", " ", raw.strip())
        if not line or is_page_number(line):
            flush()
            continue
        if line.startswith("#") or _LIST_ITEM.match(line):
            flush()
            paragraphs.append(line)
            continue
        if not buf:
            buf = line
            continue
        if _SENTENCE_END.search(buf):
            flush()
            buf = line
            continue
        buf = _join_wrapped(buf, line)
    flush()
    return "\n\n".join(paragraphs)


def strip_repeated_edges(pages: list[list[str]]) -> list[list[str]]:
    """Drop a running header/footer that shows up on most pages."""
    if len(pages) < 3:
        return pages
    edges = [_edge_lines(lines) for lines in pages]
    threshold = max(2, (len(pages) + 1) // 2)
    drop_first = {
        line
        for line, count in Counter(first for first, _ in edges if first).items()
        if count >= threshold and len(line) <= 60
    }
    drop_last = {
        line
        for line, count in Counter(last for _, last in edges if last).items()
        if count >= threshold and len(line) <= 60
    }
    cleaned: list[list[str]] = []
    for lines, (first, last) in zip(pages, edges):
        kept: list[str] = []
        skipped_first = False
        for line in lines:
            stripped = line.strip()
            if not skipped_first and stripped == first and first in drop_first:
                skipped_first = True
                continue
            if stripped:
                skipped_first = True
            kept.append(line)
        if last in drop_last and last != first:
            for index in range(len(kept) - 1, -1, -1):
                if kept[index].strip() == last:
                    del kept[index]
                    break
        cleaned.append(kept)
    return cleaned


def _edge_lines(lines: list[str]) -> tuple[str, str]:
    nonempty = [line.strip() for line in lines if line.strip() and not is_page_number(line)]
    if not nonempty:
        return "", ""
    return nonempty[0], nonempty[-1]


def _join_wrapped(left: str, right: str) -> str:
    if left.endswith("-") and right[:1].isascii() and right[:1].isalpha():
        return left[:-1] + right
    if left and right and _CJK.match(left[-1]) and _CJK.match(right[0]):
        return left + right
    return f"{left} {right}"
