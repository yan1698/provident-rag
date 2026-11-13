"""Alibaba Cloud Model Studio Qwen client, OpenAI-compatible mode."""

from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient, LLMError, LLMResponse, fail_provider_call, llm_timeout
from app.llm.json_io import completion_text
from app.llm.meter import usage_from_openai

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def normalize_qwen_base_url(url: str | None) -> str:
    """Accept a compatible-mode host. Do not append /v1 again or keep /chat/completions."""
    raw = (url or "").strip() or DEFAULT_QWEN_BASE_URL
    raw = raw.rstrip("/")
    suffix = "/chat/completions"
    if raw.endswith(suffix):
        raw = raw[: -len(suffix)].rstrip("/")
    return raw or DEFAULT_QWEN_BASE_URL


class QwenClient(LLMClient):
    def __init__(self, settings: Settings, *, model: str):
        api_key = (settings.dashscope_api_key or "").strip()
        if not api_key:
            raise LLMError("DASHSCOPE_API_KEY 未配置", kind="auth")
        chosen = (model or "").strip()
        if not chosen:
            raise LLMError("Qwen model is empty", kind="unavailable")
        from openai import OpenAI

        self._model = chosen
        self._client = OpenAI(
            api_key=api_key,
            base_url=normalize_qwen_base_url(settings.qwen_base_url),
            timeout=llm_timeout(settings),
            max_retries=0,
        )

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
        purpose: str = "",
        json_object: bool = False,
        model: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        total_output_cap = kwargs.pop("total_output_cap", None)
        request_timeout = kwargs.pop("request_timeout", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        chosen = (model or self._model).strip() or self._model
        body: dict = {
            "model": chosen,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if json_object:
            body["response_format"] = {"type": "json_object"}
        if total_output_cap is not None:
            body.pop("max_tokens", None)
            body["max_completion_tokens"] = int(total_output_cap)
        if request_timeout is not None:
            body["timeout"] = float(request_timeout)
        if thinking_budget is not None:
            body["extra_body"] = {"thinking_budget": int(thinking_budget)}
        try:
            response = self._client.chat.completions.create(**body)
        except Exception as exc:  # noqa: BLE001 — classified, then the provider body is dropped
            fail_provider_call("qwen", exc)
        choice = response.choices[0] if response.choices else None
        text = completion_text(choice)
        if not text:
            raise LLMError("qwen empty response", kind="unavailable")
        return LLMResponse(
            text=text,
            usage=usage_from_openai(response, provider="qwen", model=chosen, purpose=purpose),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
        )
