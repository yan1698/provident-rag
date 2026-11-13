from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient, LLMError, LLMResponse, fail_provider_call, llm_timeout
from app.llm.json_io import completion_text
from app.llm.meter import usage_from_openai

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(LLMClient):
    """DeepSeek OpenAI-compatible Chat Completions client (classify / rewrite)."""

    def __init__(self, settings: Settings):
        api_key = (settings.deepseek_api_key or "").strip()
        if not api_key:
            raise LLMError("DEEPSEEK_API_KEY 未配置")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=(settings.deepseek_base_url or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/"),
            timeout=llm_timeout(settings),
            max_retries=0,
        )
        self._model = settings.llm_deepseek_model or "deepseek-v4-flash"

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
        json_object: bool = False,
        model: str | None = None,
        purpose: str = "",
        **kwargs,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": (model or self._model).strip() or self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            fail_provider_call("deepseek", exc)

        choice = response.choices[0] if response.choices else None
        text = completion_text(choice)
        if text:
            return LLMResponse(
                text=text,
                usage=usage_from_openai(response, provider="deepseek", model=str(kwargs["model"]), purpose=purpose),
            )
        reason = getattr(choice, "finish_reason", "") or "unknown"
        raise LLMError(f"DeepSeek returned an empty response (finish_reason={reason})")
