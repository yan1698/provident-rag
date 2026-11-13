from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_ONLY_CITATIONS_RE = re.compile(r"^(?:\s*\[S\d+\])+\s*$")
_WHITESPACE_RE = re.compile(r"\s+")
# Markdown scaffolding asserts nothing, so it must not count as an uncited claim.
_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]*-[\s:|-]*\|[\s:|-]*$")
_TABLE_ROW_RE = re.compile(r"^\|.*\|$")
_HEADING_RE = re.compile(r"^(?:#{1,6}\s+\S.*|\*\*[^*]{1,60}\*\*)[:：]?$")
_RULE_RE = re.compile(r"^([-*_=])\1{2,}$")


@dataclass(frozen=True)
class Claim:
    """One sentence-level statement extracted from an LLM answer."""

    text: str
    raw: str
    citation_indices: list[int]


def split_claims(answer: str) -> list[Claim]:
    """Split an answer into claims, keeping trailing ``[S<n>]`` markers attached.

    Citation-only fragments (a ``[S1]`` that landed on its own line after a
    sentence split) are merged back onto the previous claim. Fragments shorter
    than 4 characters after stripping citations are dropped.

    Markdown structure is not a claim: headings and table separators are dropped,
    and table rows inherit the citation of the sentence that introduced the table,
    since models never repeat ``[S<n>]`` inside a table cell.
    """
    if not answer or not answer.strip():
        return []

    fragments = [part.strip() for part in _SPLIT_RE.split(answer) if part and part.strip()]
    merged: list[str] = []
    for frag in fragments:
        if merged and _ONLY_CITATIONS_RE.match(frag):
            merged[-1] = f"{merged[-1]} {frag}"
        else:
            merged.append(frag)

    plain = [_WHITESPACE_RE.sub(" ", _CITATION_RE.sub("", raw)).strip() for raw in merged]
    claims: list[Claim] = []
    inherited: list[int] = []
    for position, raw in enumerate(merged):
        indices = _unique_indices(int(match.group(1)) for match in _CITATION_RE.finditer(raw))
        text = plain[position]
        header = _TABLE_ROW_RE.match(text) and position + 1 < len(plain) and _TABLE_SEP_RE.match(plain[position + 1])
        if header or _is_scaffolding(text):
            continue
        text = text.strip(" \t-–—:：;；,，")
        if len(text) < 4:
            continue
        if indices:
            inherited = indices
        elif _TABLE_ROW_RE.match(text):
            indices = list(inherited)
        claims.append(Claim(text=text, raw=raw, citation_indices=indices))
    return claims


def _is_scaffolding(text: str) -> bool:
    return bool(_TABLE_SEP_RE.match(text) or _HEADING_RE.match(text) or _RULE_RE.match(text))


def _unique_indices(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
