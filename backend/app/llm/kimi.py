from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient, LLMError, LLMResponse, fail_provider_call, llm_timeout
from app.llm.meter import usage_from_openai

# 国内平台默认端点；海外可用 https://api.moonshot.ai/v1（通过 KIMI_BASE_URL 覆盖）
DEFAULT_KIMI_BASE_URL = "https://api.moonshot.cn/v1"


class KimiClient(LLMClient):
    """Moonshot / Kimi OpenAI-compatible Chat Completions client."""

    def __init__(self, settings: Settings):
        api_key = settings.moonshot_api_key or settings.kimi_api_key
        if not api_key:
            raise LLMError("MOONSHOT_API_KEY（或 KIMI_API_KEY）未配置")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.kimi_base_url or DEFAULT_KIMI_BASE_URL,
            timeout=llm_timeout(settings),
            max_retries=0,
        )
        self._model = settings.llm_kimi_model
        self._temperature = settings.llm_temperature

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
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_output_tokens,
        }

        # kimi-k2.x 系列对 temperature / thinking 有硬约束：
        # - thinking=disabled 时只允许 temperature=0.6
        # - thinking=enabled（默认）时只允许 temperature=1
        # RAG 生成优先关掉 thinking，降低延迟与输出噪声。
        if self._model.startswith("kimi-k2"):
            kwargs["temperature"] = 0.6
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            fail_provider_call("kimi", exc)

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else None
        if not text:
            raise LLMError("Kimi returned an empty response")
        return LLMResponse(
            text=text,
            usage=usage_from_openai(response, provider="kimi", model=self._model, purpose=purpose),
        )
