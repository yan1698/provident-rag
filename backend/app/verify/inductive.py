"""Inductive claims: support_set from citations; sub-assertions; no silent elevation."""

from __future__ import annotations

import re
from typing import Any

from app.verify.claims import split_claims

STATUS_SUPPORTED_INDUCTIVE = "supported_inductive"
STATUS_UNSUPPORTED_INDUCTIVE = "unsupported_inductive"

_SUB_SPLIT = re.compile(r"[，,；;]")
_ELEVATION = re.compile(r"因此|总之|本质上|这意味着|所以必须|核心是")

INDUCTIVE_CONCAT_CAP = 800


def split_subclaims(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [piece.strip(" 。．.、") for piece in _SUB_SPLIT.split(raw) if piece.strip()]
    return [item for item in parts if len(item) >= 2] or [raw]


def is_elevation(text: str) -> bool:
    return bool(_ELEVATION.search(text or ""))


def judge_inductive(
    sentence: str,
    *,
    citation_indices: list[int],
    evidence_by_index: dict[int, Any],
    subset_entailed: dict[tuple[str, tuple[int, ...]], bool],
    concat_cap: int = INDUCTIVE_CONCAT_CAP,
) -> str:
    support = [idx for idx in citation_indices if idx in evidence_by_index]
    if not support:
        return STATUS_UNSUPPORTED_INDUCTIVE
    support_t = tuple(support)
    blob = _concat(evidence_by_index, support, cap=concat_cap)
    if len(blob) > concat_cap:
        return STATUS_UNSUPPORTED_INDUCTIVE
    subs = split_subclaims(sentence)
    if is_elevation(sentence):
        if not subset_entailed.get((sentence, support_t), False):
            return STATUS_UNSUPPORTED_INDUCTIVE
        return STATUS_SUPPORTED_INDUCTIVE
    for sub in subs:
        if not subset_entailed.get((sub, support_t), False):
            return STATUS_UNSUPPORTED_INDUCTIVE
    return STATUS_SUPPORTED_INDUCTIVE


def _concat(evidence_by_index: dict[int, Any], indexes: list[int], *, cap: int) -> str:
    parts: list[str] = []
    used = 0
    for idx in indexes:
        item = evidence_by_index[idx]
        text = str(getattr(item, "text", None) or item.get("text") if isinstance(item, dict) else item or "")
        if used + len(text) > cap:
            text = text[: max(0, cap - used)]
        parts.append(text)
        used += len(text)
        if used >= cap:
            break
    return "".join(parts)


def claims_for_inductive(answer: str) -> list:
    return split_claims(answer)
