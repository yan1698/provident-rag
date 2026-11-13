from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient, LLMError, LLMResponse, llm_timeout
from app.llm.meter import usage_from_openai


class OpenAIClient(LLMClient):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=llm_timeout(settings),
            max_retries=0,
        )
        self._model = settings.llm_openai_model

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
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a uniform LLMError
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message else None
        if not text:
            raise LLMError("OpenAI returned an empty response")
        return LLMResponse(
            text=text,
            usage=usage_from_openai(response, provider="openai", model=self._model, purpose=purpose),
        )
