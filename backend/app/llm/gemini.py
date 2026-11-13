from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient, LLMError, LLMResponse, LLMUsage


class GeminiClient(LLMClient):
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not configured")
        from google import genai
        from google.genai import types

        self._genai = genai
        read_ms = int(float(settings.llm_read_timeout_seconds) * 1000)
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=read_ms),
        )
        self._model = settings.llm_gemini_model

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
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surface as a uniform LLMError
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("Gemini returned an empty response")
        meta = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                provider="gemini",
                model=self._model,
                prompt_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
                completion_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
                purpose=purpose,
            ),
        )
