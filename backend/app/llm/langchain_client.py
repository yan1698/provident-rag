"""Optional adapter. Uses LangChain's provider layer, not its agent loop."""

from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient, LLMError, LLMResponse, LLMUsage


class LangChainClient(LLMClient):
    def __init__(self, settings: Settings):
        model_id = (getattr(settings, "langchain_model", "") or "").strip()
        if not model_id:
            raise LLMError("LLM_PROVIDER=langchain 需要设置 LANGCHAIN_MODEL，例如 anthropic:claude-sonnet-4-5")
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:
            raise LLMError("LLM_PROVIDER=langchain 需要安装 langchain。这是可选依赖，业务代码不依赖它。") from exc
        self._model = init_chat_model(model_id)
        self._model_id = model_id

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
            message = self._model.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LangChain request failed: {exc}") from exc
        text = getattr(message, "text", None)
        if callable(text):
            text = text()
        if not text:
            text = getattr(message, "content", None)
        if not text:
            raise LLMError("LangChain returned an empty response")
        meta = getattr(message, "usage_metadata", None) or {}
        return LLMResponse(
            text=str(text),
            usage=LLMUsage(
                provider="langchain",
                model=self._model_id,
                prompt_tokens=int(meta.get("input_tokens") or 0),
                completion_tokens=int(meta.get("output_tokens") or 0),
                purpose=purpose,
            ),
        )
