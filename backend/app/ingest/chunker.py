from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.corpus.catalog import HEADING_LEVEL_RE
from app.ingest.parser import ParsedPage

_SENTENCE_SPLIT = re.compile(
    r"(?<=[。！？；])(?=\S)|(?<=[。！？；])$|(?<=[!?])\s+|(?<=[;])\s+|(?<!\d)(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])|\n+"
)
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_FENCE = re.compile(r"^\s{0,3}```")
_TABLE = re.compile(r"^\s*\|.+\|")


@dataclass(frozen=True)
class ChunkDraft:
    page: int
    start_offset: int
    end_offset: int
    text: str
    token_count: int
    chunk_hash: str
    section_path: str = ""


@dataclass(frozen=True)
class _Unit:
    span: ParsedPage
    starts_section: bool = False
    level: int = 0
    title: str = ""
    atomic: bool = False


def count_tokens(text: str) -> int:
    """Offline token estimate aligned with cl100k-style budgets.

    CJK character ≈ 1 token; Latin word ≈ 1 token; remaining symbols ≈ 1.
    Avoids tiktoken's first-run download of encoding files.
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    residual = _CJK.sub(" ", text)
    residual = _LATIN_WORD.sub(" ", residual)
    symbols = sum(1 for ch in residual if not ch.isspace())
    return cjk + latin + symbols


def chunk_pages(
    pages: list[ParsedPage],
    min_tokens: int = 140,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
    filename: str = "",
) -> list[ChunkDraft]:
    units = _units_from_pages(pages, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    if not units:
        return []

    drafts: list[ChunkDraft] = []
    buf: list[_Unit] = []
    pending: ParsedPage | None = None
    stack: list[tuple[int, str]] = []

    def path() -> str:
        from app.corpus.names import filename_usable

        parts = [filename.strip()] if filename.strip() and filename_usable(filename) else []
        parts.extend(title for _, title in stack)
        return " / ".join(parts)[:240]

    def flush(*, overlap: bool) -> None:
        nonlocal buf, pending
        if not buf:
            pending = None
            return
        drafts.append(_draft_from_parts(pending, buf, section_path=path()))
        pending = _overlap_page(buf, overlap_tokens) if overlap else None
        buf = []

    for unit in units:
        if unit.starts_section:
            if any(not item.starts_section for item in buf):
                flush(overlap=False)
            else:
                pending = None
            _push_heading(stack, unit.level, unit.title)
            buf.append(unit)
            continue
        trial = _pieces(pending, buf + [unit])
        if buf and count_tokens(_join(trial)) > max_tokens:
            flush(overlap=True)
            if pending and count_tokens(_join(_pieces(pending, [unit]))) > max_tokens:
                pending = None
            buf = [unit]
        else:
            if not buf and pending and count_tokens(_join(_pieces(pending, [unit]))) > max_tokens:
                pending = None
            buf = buf + [unit]
        if count_tokens(_join(_pieces(pending, buf))) >= min_tokens:
            flush(overlap=True)

    flush(overlap=False)
    return drafts


def _units_from_pages(
    pages: list[ParsedPage],
    max_tokens: int,
    overlap_tokens: int,
) -> list[_Unit]:
    units: list[_Unit] = []
    budget = max(max_tokens - max(overlap_tokens, 0), max(1, max_tokens // 2))
    for page in pages:
        for unit in _segment_page(page):
            if unit.starts_section or count_tokens(unit.span.text) <= max_tokens:
                units.append(unit)
                continue
            if unit.atomic and count_tokens(unit.span.text) <= max_tokens:
                units.append(unit)
                continue
            for piece in _split_oversized(unit.span, budget):
                units.append(_Unit(span=piece, atomic=unit.atomic))
    return units


def _segment_page(page: ParsedPage) -> list[_Unit]:
    lines = list(_iter_lines(page.text))
    units: list[_Unit] = []
    index = 0
    while index < len(lines):
        start, end, line = lines[index]
        if not line.strip():
            index += 1
            continue
        if _FENCE.match(line):
            close = index + 1
            while close < len(lines) and not _FENCE.match(lines[close][2]):
                close += 1
            if close < len(lines):
                close += 1
            units.append(_slice_unit(page, start, lines[close - 1][1], atomic=True))
            index = close
            continue
        heading = HEADING_LEVEL_RE.match(line)
        if heading and heading.group(2).strip():
            units.append(
                _slice_unit(
                    page,
                    start,
                    end,
                    starts_section=True,
                    level=len(heading.group(1)),
                    title=heading.group(2).strip(),
                )
            )
            index += 1
            continue
        if _TABLE.match(line):
            close = index + 1
            while close < len(lines) and _TABLE.match(lines[close][2]):
                close += 1
            units.append(_slice_unit(page, start, lines[close - 1][1], atomic=True))
            index = close
            continue
        close = index + 1
        while close < len(lines):
            nxt = lines[close][2]
            if not nxt.strip() or _FENCE.match(nxt) or HEADING_LEVEL_RE.match(nxt) or _TABLE.match(nxt):
                break
            close += 1
        units.append(_slice_unit(page, start, lines[close - 1][1]))
        index = close
    return [unit for unit in units if unit.span.text.strip()]


def _iter_lines(text: str):
    pos = 0
    for line in text.split("\n"):
        yield pos, pos + len(line), line
        pos += len(line) + 1


def _slice_unit(
    page: ParsedPage,
    start: int,
    end: int,
    *,
    starts_section: bool = False,
    level: int = 0,
    title: str = "",
    atomic: bool = False,
) -> _Unit:
    span = _span_from_slice(page, start, end)
    return _Unit(
        span=span or ParsedPage(page.page, "", page.start_offset, page.start_offset),
        starts_section=starts_section,
        level=level,
        title=title,
        atomic=atomic,
    )


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


def _split_oversized(span: ParsedPage, budget: int) -> list[ParsedPage]:
    pieces = _split_sentences(span)
    if len(pieces) == 1 and count_tokens(pieces[0].text) > budget:
        pieces = _split_text_by_tokens(span.text, max_tokens=budget, overlap_tokens=0)
        spans: list[ParsedPage] = []
        cursor = 0
        for text in pieces:
            local = span.text.find(text, cursor)
            if local < 0:
                local = cursor
            start = span.start_offset + local
            spans.append(
                ParsedPage(
                    page=span.page,
                    text=text,
                    start_offset=start,
                    end_offset=start + len(text),
                )
            )
            cursor = local + max(len(text), 1)
        return spans
    packed: list[ParsedPage] = []
    buf: list[ParsedPage] = []
    for piece in pieces:
        if count_tokens(piece.text) > budget:
            if buf:
                packed.append(_concat_spans(buf))
                buf = []
            packed.extend(_split_oversized(piece, budget))
            continue
        candidate = buf + [piece]
        if buf and count_tokens(_join(candidate)) > budget:
            packed.append(_concat_spans(buf))
            buf = [piece]
        else:
            buf = candidate
    if buf:
        packed.append(_concat_spans(buf))
    return packed or [span]


def _split_sentences(para: ParsedPage) -> list[ParsedPage]:
    text = para.text
    spans: list[ParsedPage] = []
    last = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        piece = text[last : match.start()].strip()
        if piece:
            local = text.find(piece, last)
            start = para.start_offset + local
            spans.append(
                ParsedPage(
                    page=para.page,
                    text=piece,
                    start_offset=start,
                    end_offset=start + len(piece),
                )
            )
        last = match.end()
    tail = text[last:].strip()
    if tail:
        local = text.find(tail, last)
        start = para.start_offset + max(local, 0)
        spans.append(
            ParsedPage(
                page=para.page,
                text=tail,
                start_offset=start,
                end_offset=start + len(tail),
            )
        )
    return spans or [para]


def _concat_spans(parts: list[ParsedPage]) -> ParsedPage:
    return ParsedPage(
        page=parts[0].page,
        text=_join(parts),
        start_offset=parts[0].start_offset,
        end_offset=parts[-1].end_offset,
    )


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _overlap_page(units: list[_Unit], overlap_tokens: int) -> ParsedPage | None:
    if overlap_tokens <= 0:
        return None
    chosen: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if unit.starts_section:
            break
        chosen.append(unit)
        total += count_tokens(unit.span.text)
        if total >= overlap_tokens:
            break
    chosen.reverse()
    if not chosen:
        return None
    text = _join([unit.span for unit in chosen])
    if count_tokens(text) > overlap_tokens:
        keep = _tail_for_tokens(text, overlap_tokens)
        text = text[-keep:].lstrip()
    if not text.strip():
        return None
    host = chosen[-1].span
    local = host.text.rfind(text)
    if local >= 0:
        start = host.start_offset + local
        return ParsedPage(page=host.page, text=text, start_offset=start, end_offset=start + len(text))
    return ParsedPage(
        page=chosen[0].span.page,
        text=text,
        start_offset=chosen[0].span.start_offset,
        end_offset=chosen[-1].span.end_offset,
    )


def _pieces(prefix: ParsedPage | None, units: list[_Unit]) -> list[ParsedPage]:
    parts: list[ParsedPage] = []
    if prefix is not None and prefix.text.strip():
        parts.append(prefix)
    parts.extend(unit.span for unit in units if unit.span.text.strip())
    return parts


def _draft_from_parts(prefix: ParsedPage | None, units: list[_Unit], *, section_path: str) -> ChunkDraft:
    parts = _pieces(prefix, units)
    return _make_draft(
        page=parts[0].page,
        start_offset=parts[0].start_offset,
        end_offset=parts[-1].end_offset,
        text=_join(parts),
        section_path=section_path,
    )


def _split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        lo, hi = start + 1, length
        best = start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            n = count_tokens(text[start:mid])
            if n <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        piece = text[start:best].strip() or text[start:best]
        parts.append(piece)
        if best >= length:
            break
        overlap = _tail_for_tokens(text[start:best], overlap_tokens)
        start = max(best - overlap, start + 1)
    return [p for p in parts if p.strip()]


def _tail_for_tokens(text: str, overlap_tokens: int) -> int:
    if overlap_tokens <= 0 or not text:
        return 0
    lo, hi = 0, len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_tokens(text[len(text) - mid :])
        if n <= overlap_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _join(parts: list[ParsedPage]) -> str:
    return "\n\n".join(part.text for part in parts if part.text)


def _make_draft(
    *,
    page: int,
    start_offset: int,
    end_offset: int,
    text: str,
    section_path: str = "",
) -> ChunkDraft:
    digest = hashlib.sha256(f"{page}:{start_offset}:{end_offset}:{section_path}:{text}".encode("utf-8")).hexdigest()
    return ChunkDraft(
        page=page,
        start_offset=start_offset,
        end_offset=end_offset,
        text=text,
        token_count=count_tokens(text),
        chunk_hash=digest,
        section_path=section_path,
    )
