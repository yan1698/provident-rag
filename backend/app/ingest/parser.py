from __future__ import annotations

import re
from dataclasses import dataclass

import fitz

from app.ingest.clean import (
    decode_bytes,
    is_page_number,
    merge_hard_breaks,
    normalize_markup,
    strip_repeated_edges,
)

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
CONTENT_TYPE_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedPage:
    page: int
    text: str
    start_offset: int
    end_offset: int


def detect_suffix(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    lower = name.lower()
    for suffix in (".markdown", ".pdf", ".txt", ".md"):
        if lower.endswith(suffix):
            return suffix
    return ""


def parse_document(filename: str, data: bytes) -> list[ParsedPage]:
    suffix = detect_suffix(filename)
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedFileTypeError(f"unsupported file type: {filename!r}; allowed: PDF, TXT, MD")

    if suffix == ".pdf":
        pages = _parse_pdf(data)
    else:
        pages = _parse_text(data)

    nonempty = [p for p in pages if p.text.strip()]
    if not nonempty:
        raise EmptyDocumentError("no extractable text in document")
    return nonempty


def _parse_pdf(data: bytes) -> list[ParsedPage]:
    line_pages: list[tuple[int, list[str]]] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for index, page in enumerate(doc, start=1):
            line_pages.append((index, _pdf_page_lines(page)))
    stripped = strip_repeated_edges([lines for _, lines in line_pages])
    # Font inspection is page-local, but the numbered heading stack is not.
    # Admonitions at the next page must remain children of the preceding section.
    numbered_level = None
    for lines in stripped:
        for index, line in enumerate(lines):
            heading = re.match(r"^#{1,6}\s+(.+)$", line)
            if not heading:
                continue
            title = heading.group(1)
            number = re.match(r"^(\d+(?:\.\d+)*)\.\s", title)
            if re.match(r"^Chapter\s+\d+\.", title, re.I):
                numbered_level = 1
                level = 1
            elif number:
                numbered_level = len(number.group(1).split("."))
                level = min(6, numbered_level)
            elif numbered_level is not None:
                level = min(6, numbered_level + 1)
            else:
                continue
            lines[index] = f"{'#' * level} {title}"
    texts = [merge_hard_breaks("\n".join(lines)) for lines in stripped]
    return _pack_pages([(index, text) for (index, _), text in zip(line_pages, texts)])


def _parse_text(data: bytes) -> list[ParsedPage]:
    text = normalize_markup(decode_bytes(data))
    return _pack_pages([(1, text)])


def _pack_pages(pairs: list[tuple[int, str]]) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    cursor = 0
    for index, raw in pairs:
        text = normalize_markup(raw)
        if not text.strip():
            continue
        start = cursor
        end = start + len(text)
        pages.append(ParsedPage(page=index, text=text, start_offset=start, end_offset=end))
        cursor = end
    return pages


def _pdf_page_lines(page) -> list[str]:
    try:
        payload = page.get_text("dict", sort=True) or {}
    except Exception:  # noqa: BLE001 — fall back to plain text
        payload = {}
    lines = _lines_from_dict(payload)
    if lines:
        return lines
    text = page.get_text("text", sort=True) or ""
    return normalize_markup(text).split("\n")


def _lines_from_dict(payload: dict) -> list[str]:
    measured: list[tuple[float, bool, str, int]] = []
    for block_index, block in enumerate(payload.get("blocks") or []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(str(span.get("text") or "") for span in spans).strip()
            if not text or is_page_number(text):
                continue
            size = max((float(span.get("size") or 0) for span in spans), default=0.0)
            bold = bool(spans) and all(int(span.get("flags") or 0) & 16 for span in spans)
            measured.append((size, bold, text, block_index))
    if not measured:
        return []
    sizes = sorted(size for size, _, _, _ in measured if size > 0)
    body = sizes[len(sizes) // 2] if sizes else 0.0
    lines: list[str] = []
    cursor = 0
    numbered_level = 1
    while cursor < len(measured):
        size, bold, text, block_index = measured[cursor]
        level = _heading_level(size, bold, text, body)
        if level and re.match(r"^(?:Chapter\s+\d+|\d+(?:\.\d+)*\.)\s", text, re.I):
            numbered_level = level
        elif level and text in {"Note", "Important", "Tip", "Warning", "Caution"}:
            level = min(6, numbered_level + 1)
        # Wrapped headings share a PDF text block/font. Never merge following body
        # lines or a second numbered heading, even when they are on the same page.
        cursor += 1
        while level and cursor < len(measured):
            next_size, next_bold, following, next_block = measured[cursor]
            if (
                next_block != block_index
                or abs(next_size - size) > 0.3
                or next_bold != bold
                or re.match(r"^(?:Chapter\s+\d+|\d+(?:\.\d+)*\.)\s", following, re.I)
            ):
                break
            if len(text) + len(following) > 240:
                break
            text += " " + following
            cursor += 1
        lines.append(f"{'#' * level} {text}" if level else text)
    return lines


def _heading_level(size: float, bold: bool, text: str, body: float) -> int:
    numbered = re.match(r"^(\d+(?:\.\d+)*)\.\s+", text)
    if re.match(r"^Chapter\s+\d+\.", text, re.I) and (bold or size >= body * 1.2):
        return 1
    if numbered and len(text) <= 240 and (bold or (body > 0 and size >= body * 1.2)):
        return min(6, len(numbered.group(1).split(".")))
    if len(text) > 100 or re.search(r"[。！？!?；;]$", text):
        return 0
    if body <= 0:
        return 2 if bold and len(text) <= 30 else 0
    if size >= body * 1.45:
        return 1
    if size >= body * 1.2:
        return 2
    if bold and size >= body * 0.95 and len(text) <= 30:
        return 2
    return 0


_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_paragraphs(page: ParsedPage) -> list[ParsedPage]:
    """Split a page into paragraph spans, preserving character offsets."""
    text = page.text
    spans: list[ParsedPage] = []
    last = 0
    for match in _PARA_SPLIT.finditer(text):
        span = _span_from_slice(page, last, match.start())
        if span:
            spans.append(span)
        last = match.end()
    tail = _span_from_slice(page, last, len(text))
    if tail:
        spans.append(tail)
    return spans or ([page] if page.text.strip() else [])


def _span_from_slice(page: ParsedPage, start: int, end: int) -> ParsedPage | None:
    piece = page.text[start:end]
    stripped = piece.strip()
    if not stripped:
        return None
    lead = len(piece) - len(piece.lstrip())
    local_start = start + lead
    local_end = local_start + len(stripped)
    return ParsedPage(
        page=page.page,
        text=stripped,
        start_offset=page.start_offset + local_start,
        end_offset=page.start_offset + local_end,
    )
