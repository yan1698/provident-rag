"""Synthesis verifier helpers. No reranker import — gaps/order_note are meta."""

from __future__ import annotations

import re
from dataclasses import dataclass

STATUS_META = "meta"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


@dataclass
class MetaClaim:
    text: str
    status: str = STATUS_META
    hallucination: bool = False
    citation_indices: list[int] | None = None


def meta_claim_checks(order_note: str, gaps: list[dict[str, str]] | None = None) -> list[MetaClaim]:
    checks: list[MetaClaim] = []
    note = (order_note or "").strip()
    if note:
        checks.append(MetaClaim(text=note))
    for gap in gaps or []:
        topic = str(gap.get("topic") or "").strip()
        reason = str(gap.get("reason") or "").strip()
        text = f"{topic}：{reason}".strip("：") if topic else reason
        if len(text) < 4:
            continue
        checks.append(MetaClaim(text=text))
    return checks


def _module_texts(module: dict) -> list[str]:
    texts = [str(module.get("heading") or module.get("title") or "")]
    for item in module.get("paragraphs") or module.get("bullets") or []:
        if isinstance(item, dict):
            texts.append(str(item.get("text") or ""))
        else:
            texts.append(str(item))
    return [piece for piece in texts if piece]


def module_coverage_rate(modules: list[dict], checks: list) -> float:
    """Share of modules that have at least one supported sentence."""
    countable = [module for module in modules if _module_texts(module)]
    if not countable:
        return 1.0
    supported_texts = [item.text for item in checks if getattr(item, "status", "") == "supported"]
    covered = 0
    for module in countable:
        blobs = _module_texts(module)
        hit = False
        for claim_text in supported_texts:
            for blob in blobs:
                if blob and (blob in claim_text or claim_text in blob):
                    hit = True
                    break
            if hit:
                break
        if hit:
            covered += 1
    return round(covered / len(countable), 6)


def unit_is_covered(question: str, modules: list[dict]) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    q_tokens = {token.casefold() for token in _TOKEN_RE.findall(q)}
    for module in modules:
        blob = " ".join(_module_texts(module))
        if q in blob:
            return True
        m_tokens = {token.casefold() for token in _TOKEN_RE.findall(blob)}
        if q_tokens and len(q_tokens & m_tokens) / len(q_tokens) >= 0.35:
            return True
    return False


def task_coverage_rate(questions: list[str], modules: list[dict]) -> float:
    countable = [item for item in questions if (item or "").strip()]
    if not countable:
        return 1.0
    covered = sum(1 for question in countable if unit_is_covered(question, modules))
    return round(covered / len(countable), 6)
