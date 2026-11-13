"""Lexical overlap for catalog ranking and off-topic skips.

No domain lexicon: score only uses the user goal vs filename / headings / preview.
Intent morphology (指南、规划、…) is stripped so「学习指南」does not match「设计指南」.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.agent.classify import INTENT_WORDS

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
_FILLER = re.compile(
    r"根据知识库|知识库中|知识库能|知识库|请你|请根据|关于|哪些部分|怎么学|"
    r"的要点|一下|相关内容|帮我|请"
)
_CHAT = re.compile(
    r"我是一名|我是一个|我是|初学者|为我|帮我|一份|一篇|这个|那个|"
    r"可以吗|能否|怎么样"
)
_EXTRA_STRIP = ("说明", "解释", "大纲")
_SPACE = re.compile(r"\s+")
_SEGMENT = re.compile(r"[^\w\u4e00-\u9fff]+")
OFF_TOPIC_FLOOR = 0.08


def tokenize(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text or "")}


def filename_stem(filename: str) -> str:
    return Path(filename or "").stem


def strip_chat(text: str) -> str:
    """Drop conversational filler but keep topic and task words, for retrieval queries."""
    cleaned = _FILLER.sub(" ", text or "")
    cleaned = _CHAT.sub(" ", cleaned)
    return _SPACE.sub(" ", cleaned).strip(" 。？?！!，,、的")


def strip_intent(text: str) -> str:
    cleaned = _FILLER.sub(" ", text or "")
    cleaned = _CHAT.sub(" ", cleaned)
    words = sorted((*INTENT_WORDS, *_EXTRA_STRIP), key=len, reverse=True)
    for word in words:
        cleaned = cleaned.replace(word, " ")
    return cleaned


def lexical_keys(text: str) -> set[str]:
    """Latin words plus CJK bigrams. Unigrams like 学 in 教学 must not match 学习."""
    latin = {token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")}
    chars = [ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff"]
    bigrams = {a + b for a, b in zip(chars, chars[1:])}
    return latin | bigrams


def topic_tokens(goal: str) -> set[str]:
    """Content tokens from a goal, without task-shape words like 指南 / 规划."""
    return tokenize(strip_intent(goal))


def topic_keys(goal: str) -> set[str]:
    return lexical_keys(strip_intent(goal)) or lexical_keys(goal or "")


def coverage(part: set[str], whole: set[str]) -> float:
    if not part:
        return 0.0
    return len(part & whole) / len(part)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def uncovered_terms(goal: str, blob: str, *, limit: int = 3) -> list[str]:
    """Goal segments that never appear in ``blob``.

    Used to build a follow-up retrieve query from the user's own words instead of
    from claims the verifier already rejected.
    """
    haystack = (blob or "").casefold()
    out: list[str] = []
    for piece in _SEGMENT.split(strip_intent(goal)):
        term = piece.strip().casefold()
        if len(term) < 2 or term in out:
            continue
        if term in haystack:
            continue
        out.append(term)
        if len(out) >= limit:
            break
    return out


def score_document(
    goal: str,
    filename: str,
    sections: list[str] | None = None,
    preview: str = "",
) -> float:
    """How well a document matches the user goal. Filename overlap is weighted highest."""
    topic = topic_keys(goal)
    if not topic:
        return 0.0
    stem = filename_stem(filename)
    file_keys = lexical_keys(f"{stem} {filename}")
    sec_keys = lexical_keys(" ".join((sections or [])[:24]))
    preview_keys = lexical_keys((preview or "")[:1200])
    blob = f"{filename} {' '.join(sections or [])} {preview or ''}".casefold()

    score = max(
        coverage(topic, sec_keys),
        jaccard(topic, sec_keys | file_keys),
        0.5 * coverage(topic, preview_keys),
    )
    if file_keys & topic:
        score = max(score, 0.62 + 0.38 * coverage(topic, file_keys))
    stem_cf = stem.casefold()
    goal_cf = (goal or "").casefold()
    if stem_cf and (stem_cf in goal_cf or goal_cf in stem_cf):
        score = max(score, 0.88)
    for token in topic:
        if len(token) >= 3 and token.casefold() in blob:
            in_name = token.casefold() in filename.casefold() or token.casefold() in stem_cf
            score = max(score, 0.72 if in_name else 0.35)
    return min(1.0, score)
