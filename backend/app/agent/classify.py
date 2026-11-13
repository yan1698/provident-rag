"""Classify agent goals into factoid vs synthesis.

Rules first; optional DeepSeek when the goal has no intent words.
"""

from __future__ import annotations

import re
from typing import Any

TASK_KIND_AUTO = "auto"
TASK_KIND_FACTOID = "factoid"
TASK_KIND_SYNTHESIS = "synthesis"
TASK_KIND_DIGEST = "digest"
TASK_KINDS = (TASK_KIND_AUTO, TASK_KIND_FACTOID, TASK_KIND_SYNTHESIS, TASK_KIND_DIGEST)
LONGFORM_KINDS = (TASK_KIND_SYNTHESIS, TASK_KIND_DIGEST)

# Cross-object or deliverable shapes → synthesis. Do not add subject words
# (前端 / 设计 / 学习 / React).
SYNTHESIS_WORDS: tuple[str, ...] = (
    "对比",
    "比较",
    "异同",
    "区别",
    "差异",
    "对照",
    "相比",
    "清单",
    "盘点",
    "覆盖",
    "汇总",
    "大纲",
    "路径",
    "步骤",
    "方案",
    "制定",
    "指南",
    "教程",
    "梳理",
    "整理",
)

# Appear in both short how/why questions and long tasks. Strip for ranking,
# but do not send "不同类型…如何规划…" to synthesis by themselves.
WEAK_TASK_WORDS: tuple[str, ...] = ("规划", "要点", "不同")

# Task-shape intents. Used by strip_intent; classify uses SYNTHESIS_WORDS first.
INTENT_WORDS: tuple[str, ...] = tuple(dict.fromkeys((*SYNTHESIS_WORDS, *WEAK_TASK_WORDS)))

HOW_WHY_RE = re.compile(r"如何|为什么|为啥|为何|是什么|怎么|怎样")
COMPARE_DIFFERENT_RE = re.compile(r"有什么不同|有何不同|有哪些不同|不同之处|有什么不一样|两者不同|二者不同|有何差异")

_SYNTHESIS_RE = re.compile("|".join(re.escape(word) for word in SYNTHESIS_WORDS))

CLASSIFY_SYSTEM = (
    "你判断用户这句话在知识库 Agent 里应走哪条链。"
    "只输出一个 JSON 对象，不要 Markdown。"
    '形状：{"task_kind":"factoid"|"synthesis"}。'
    "factoid：单句事实、定义、某个术语/字段是什么，以及单点机制的如何/为什么"
    "（即使句子里有「规划」「不同类型」）。"
    "synthesis：必须是要产出物或跨对象——学习指南、步骤方案、清单、对比、盘点、跨章节整理。"
    "不要把「如何规划 X，为什么」当成制定方案。"
    "不要根据学科主题（前端/Agent/财务）选择，只看任务形态。"
)


def has_strong_synthesis(goal: str) -> bool:
    text = goal or ""
    return bool(_SYNTHESIS_RE.search(text) or COMPARE_DIFFERENT_RE.search(text))


def is_point_how_why(goal: str) -> bool:
    return bool(HOW_WHY_RE.search(goal or ""))


_SHORT_MORE_RE = re.compile(r"比.{1,40}多了什么")
_LONG_ANSWER_MARKS = ("对比", "比较", "区别", "差异", "维度", "展开", "详细", "异同", "对照", "相比")


def holds_short_answer(goal: str) -> bool:
    """Keep a short 「比…多了什么」 as factoid. Real comparisons still go to synthesis."""
    text = (goal or "").strip()
    if any(word in text for word in _LONG_ANSWER_MARKS) or COMPARE_DIFFERENT_RE.search(text):
        return False
    return bool(_SHORT_MORE_RE.search(text))


def classify_task_kind(
    goal: str,
    requested: str | None = TASK_KIND_AUTO,
    *,
    settings: Any | None = None,
) -> str:
    """Resolve task_kind. Explicit factoid/synthesis/digest wins; auto uses intent words then DeepSeek."""
    kind = (requested or TASK_KIND_AUTO).strip().lower()
    if kind not in TASK_KINDS:
        raise ValueError(f"task_kind must be one of {', '.join(TASK_KINDS)}")
    if kind in {TASK_KIND_FACTOID, TASK_KIND_SYNTHESIS, TASK_KIND_DIGEST}:
        return kind
    text = goal or ""
    if has_strong_synthesis(text):
        return TASK_KIND_SYNTHESIS
    if ("规划" in text or "要点" in text) and not is_point_how_why(text):
        return TASK_KIND_SYNTHESIS
    if is_point_how_why(text):
        return TASK_KIND_FACTOID
    if holds_short_answer(text):
        return TASK_KIND_FACTOID
    if settings is None:
        return TASK_KIND_FACTOID
    from app.llm import LLMError, router_llm_available

    if router_llm_available(settings):
        try:
            judged = _llm_classify(text, settings)
        except (LLMError, ValueError, TypeError):
            judged = None
        if judged in {TASK_KIND_FACTOID, TASK_KIND_SYNTHESIS}:
            return judged
    return TASK_KIND_FACTOID


def _llm_classify(goal: str, settings: Any) -> str | None:
    from app.llm import get_router_llm_client

    client = get_router_llm_client(settings)
    raw = client.generate(
        system=CLASSIFY_SYSTEM,
        prompt=f"用户目标：{goal}\n请输出 JSON。",
        max_output_tokens=80,
        temperature=0.0,
        purpose="classify",
    ).text
    data = _parse_json(raw)
    if data is None:
        return None
    kind = str(data.get("task_kind") or "").strip().lower()
    if kind in {TASK_KIND_FACTOID, TASK_KIND_SYNTHESIS}:
        return kind
    return None


def _parse_json(raw: str) -> dict[str, Any] | None:
    from app.llm import parse_json_object

    return parse_json_object(raw)
