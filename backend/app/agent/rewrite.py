"""Rewrite a user goal into a shorter retrieval query. Rules first; LLM optional."""

from __future__ import annotations

import re
from typing import Any

_FILLER = re.compile(
    r"根据知识库|知识库中|知识库能|知识库|请你|请根据|关于|哪些部分|怎么学|"
    r"的要点|一下|相关内容|帮我|请|能否|可以|一下"
)
_SPACE = re.compile(r"\s+")

REWRITE_SYSTEM = (
    "你把用户问题改写成适合在本地知识库检索的一句短 query。"
    "只输出这一句，不要解释、不要引号、不要编号。"
    "保留用户点名的文档或主题词。禁止编造知识库未提及的主题。"
    "去掉「我是一名…」「为我…」这类对话措辞，只留检索得上的主题词。"
)


def llm_available(settings: Any | None) -> bool:
    from app.llm import router_llm_available

    return router_llm_available(settings)


def rule_rewrite_query(goal: str, previous: str = "") -> str:
    from app.corpus.relevance import strip_chat

    text = _SPACE.sub(" ", _FILLER.sub(" ", strip_chat(goal))).strip(" 。？?！!，,")
    if len(text) >= 2:
        return text[:80]
    fallback = (previous or goal or "").strip()
    return fallback[:80]


def rewrite_search_query(
    goal: str,
    previous: str = "",
    *,
    settings: Any | None = None,
    filenames: list[str] | None = None,
) -> str:
    ruled = rule_rewrite_query(goal, previous)
    if not llm_available(settings):
        return ruled
    try:
        from app.llm import LLMError

        rewritten = _llm_rewrite(goal, previous, settings, filenames or [])
    except (LLMError, ValueError, TypeError):
        return ruled
    cleaned = _SPACE.sub(" ", rewritten).strip(" 。？?！!\"'“”")
    if len(cleaned) < 2:
        return ruled
    return cleaned[:80]


def _llm_rewrite(
    goal: str,
    previous: str,
    settings: Any,
    filenames: list[str],
) -> str:
    from app.llm import get_router_llm_client

    names = "、".join(filenames[:12]) if filenames else "（无）"
    prompt = f"用户问题：{goal}\n上次检索句：{previous or '（无）'}\n知识库文件名：{names}\n输出一句检索 query。"
    client = get_router_llm_client(settings)
    return client.generate(
        system=REWRITE_SYSTEM,
        prompt=prompt,
        max_output_tokens=min(80, int(getattr(settings, "llm_max_output_tokens", 1024) or 1024)),
        temperature=0.0,
        purpose="rewrite",
    ).text
