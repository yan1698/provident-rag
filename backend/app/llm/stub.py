from __future__ import annotations

import re

from app.llm.base import LLMClient, LLMResponse, LLMUsage

_FENCED_RE = re.compile(r'<evidence\s+id="S1"[^>]*>\s*(.*?)\s*</evidence>', re.DOTALL | re.IGNORECASE)
_LEGACY_RE = re.compile(r"\[S1\][^\n]*\n(.+?)(?:\n\[S2\]|\Z)", re.DOTALL)


class StubClient(LLMClient):
    """Deterministic, API-free provider for offline tests / CI.

    Does not call any external service. It quotes the first evidence block
    verbatim and cites it as ``[S1]``, so the /query response shape (answer +
    citations + evidence) can be exercised end-to-end without real API keys.

    Enable only with both ``LLM_PROVIDER=stub`` and ``ALLOW_LLM_STUB=true``.
    Rejected when ``APP_ENV=production``.
    """

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
        purpose: str = "",
        **kwargs,
    ) -> LLMResponse:
        if purpose == "general_explanation":
            return LLMResponse(
                text="[stub-provider] 通用知识解释离线占位，未经文档核验。",
                usage=LLMUsage(provider="stub", model="stub", purpose=purpose),
            )
        match = _FENCED_RE.search(prompt) or _LEGACY_RE.search(prompt)
        quote = match.group(1).strip()[:280] if match else "（无可用证据）"
        text = (
            "[stub-provider] 未配置真实 LLM API Key，以下为离线占位回答，"
            "仅用于验证 /query 链路的字段结构与引用解析，不代表真实生成质量：\n"
            f"{quote} [S1]"
        )
        return LLMResponse(text=text, usage=LLMUsage(provider="stub", model="stub", purpose=purpose))
