"""Closed relation set, name normalization, quote alignment."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

RELATIONS_G0 = frozenset({"part_of"})
KINDS_G0 = frozenset({"document", "doc_section"})
KINDS_EXTRACT = frozenset({"concept", "method", "metric", "tool", "role"})
KINDS_ALL = KINDS_G0 | KINDS_EXTRACT
GRAPH_SUFFIXES = frozenset({".md", ".markdown"})


def is_markdown_filename(filename: str | None) -> bool:
    """True when rules can build an ATX heading tree. Extract does not use this gate."""
    return Path(filename or "").suffix.lower() in GRAPH_SUFFIXES


def graph_eligible(filename: str | None) -> bool:
    """Back-compat alias: ATX skeleton likely. Concept extract ignores this."""
    return is_markdown_filename(filename)


RELATIONS_ALL = frozenset(
    {
        "part_of",
        "defines",
        "requires",
        "causes",
        "contrasts_with",
        "measured_by",
        "instance_of",
        "related_to",
    }
)
RELATION_LABELS = {
    "part_of": "从属",
    "defines": "定义",
    "requires": "前置依赖",
    "causes": "因果",
    "contrasts_with": "对照",
    "measured_by": "度量",
    "instance_of": "实例",
    "related_to": "相关",
}
_SPACE = re.compile(r"\s+")
_FIRST_SENTENCE = re.compile(r"(.+?[。！？!?]|.+$)")


@dataclass
class DraftEntity:
    name: str
    kind: str
    quote: str
    chunk_id: str
    chunk_hash: str
    start_char: int = 0
    end_char: int = 0
    aliases: list[str] = field(default_factory=list)


@dataclass
class DraftEdge:
    src_name: str
    src_kind: str
    dst_name: str
    dst_kind: str
    relation: str
    quote: str
    chunk_id: str
    chunk_hash: str
    source: str = "rules"
    start_char: int = 0
    end_char: int = 0


@dataclass
class HeadingHit:
    name: str
    level: int
    quote: str
    chunk_id: str
    chunk_hash: str
    start_char: int = 0
    end_char: int = 0


@dataclass
class GraphDrafts:
    entities: list[DraftEntity] = field(default_factory=list)
    edges: list[DraftEdge] = field(default_factory=list)
    headings: list[HeadingHit] = field(default_factory=list)
    skipped: int = 0
    note: str = ""
    source: str = "rules"
    calls: int = 0
    failed: int = 0
    empty: int = 0
    dropped: int = 0

    def extend(self, other: "GraphDrafts") -> None:
        had = bool(self.entities or self.edges)
        self.entities.extend(other.entities)
        self.edges.extend(other.edges)
        self.headings.extend(other.headings)
        self.skipped += int(other.skipped or 0)
        self.calls += int(other.calls or 0)
        self.failed += int(other.failed or 0)
        self.empty += int(other.empty or 0)
        self.dropped += int(other.dropped or 0)
        notes = [piece for piece in (self.note, other.note) if piece]
        self.note = "；".join(dict.fromkeys(notes))
        if other.source == "llm" and (other.entities or other.edges):
            if had and self.source in {"rules", "mixed"}:
                self.source = "mixed"
            else:
                self.source = "llm"


def prefix_compatible(left: str, right: str) -> bool:
    """True when two names may be the same entity. Proper prefixes are not aliases."""
    a = norm_name(left)
    b = norm_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer.startswith(shorter) and len(longer) - len(shorter) >= 2:
        return False
    return True


def norm_name(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw or "")
    text = _SPACE.sub("", text).casefold()
    return text[:256]


def first_sentence(text: str, *, limit: int = 80) -> str:
    blob = (text or "").strip()
    if not blob:
        return ""
    match = _FIRST_SENTENCE.match(blob)
    piece = (match.group(1) if match else blob).strip()
    if len(piece) <= limit:
        return piece
    return piece[:limit]


def quote_in_chunk(quote: str, text: str) -> tuple[int, int] | None:
    """Return [start, end) in chunk.text, or None if quote cannot be aligned."""
    needle = quote or ""
    hay = text or ""
    if not needle.strip() or not hay:
        return None
    idx = hay.find(needle)
    if idx >= 0:
        return idx, idx + len(needle)
    collapsed_q = _SPACE.sub("", needle)
    if not collapsed_q:
        return None
    mapping: list[int] = []
    collapsed_chars: list[str] = []
    for i, ch in enumerate(hay):
        if ch.isspace():
            continue
        mapping.append(i)
        collapsed_chars.append(ch)
    collapsed_h = "".join(collapsed_chars)
    pos = collapsed_h.find(collapsed_q)
    if pos < 0:
        return None
    start = mapping[pos]
    last = mapping[pos + len(collapsed_q) - 1]
    return start, last + 1
