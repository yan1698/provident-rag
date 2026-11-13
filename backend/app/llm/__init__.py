from __future__ import annotations

from typing import TYPE_CHECKING

from app.llm.base import LLMClient, LLMError, LLMResponse, LLMUsage, public_llm_failure

if TYPE_CHECKING:
    from app.core.config import Settings
from app.llm.json_io import RETRY_HINT, parse_json_object
from app.llm.meter import MeteredClient, collect_usage

__all__ = [
    "LLMClient",
    "LLMError",
    "public_llm_failure",
    "LLMResponse",
    "LLMUsage",
    "RETRY_HINT",
    "collect_usage",
    "answer_model_name",
    "answer_provider_name",
    "get_llm_client",
    "get_router_llm_client",
    "parse_json_object",
    "role_model_name",
    "role_provider_name",
    "router_llm_available",
    "task_model_name",
    "task_provider_name",
]


def answer_provider_name(settings: Settings) -> str:
    return role_provider_name(settings, "answer")


def task_provider_name(settings: Settings) -> str:
    return role_provider_name(settings, "task")


def answer_model_name(settings: Settings) -> str:
    return role_model_name(settings, "answer")


def task_model_name(settings: Settings) -> str:
    return role_model_name(settings, "task")


def role_provider_name(settings: Settings, role: str) -> str:
    field = "answer_llm_provider" if role == "answer" else "task_llm_provider"
    raw = getattr(settings, field, None)
    return str(raw).strip().lower() if raw else "qwen"


def role_model_name(settings: Settings, role: str) -> str:
    provider = role_provider_name(settings, role)
    if provider == "qwen":
        raw = settings.answer_llm_model if role == "answer" else settings.task_llm_model
        return (raw or "").strip()
    if provider == "kimi":
        return settings.llm_kimi_model
    if provider == "deepseek":
        return settings.llm_deepseek_model
    if provider == "gemini":
        return settings.llm_gemini_model
    if provider == "openai":
        return settings.llm_openai_model
    if provider == "langchain":
        return settings.langchain_model
    if provider == "stub":
        return "stub"
    return provider


def get_llm_client(settings: Settings) -> LLMClient:
    provider = answer_provider_name(settings)
    inner = _build_client(settings, provider, role="answer")
    return MeteredClient(inner, provider=provider, model=answer_model_name(settings))


def _build_client(settings: Settings, provider: str, *, role: str) -> LLMClient:
    if provider == "qwen":
        from app.llm.qwen import QwenClient

        return QwenClient(settings, model=role_model_name(settings, role))
    if provider == "kimi":
        from app.llm.kimi import KimiClient

        return KimiClient(settings)
    if provider == "gemini":
        from app.llm.gemini import GeminiClient

        return GeminiClient(settings)
    if provider == "openai":
        from app.llm.openai_client import OpenAIClient

        return OpenAIClient(settings)
    if provider == "deepseek":
        from app.llm.deepseek import DeepSeekClient

        return DeepSeekClient(settings)
    if provider == "langchain":
        from app.llm.langchain_client import LangChainClient

        return LangChainClient(settings)
    if provider == "stub":
        env = settings.app_env.strip().lower()
        if env == "production":
            raise LLMError(
                "stub 禁止在 APP_ENV=production 下使用。请改用 qwen / kimi / gemini / openai / deepseek / langchain。"
            )
        if not settings.allow_llm_stub:
            raise LLMError(
                "stub 需要同时设置 ALLOW_LLM_STUB=true（仅离线/CI）。"
                "正式问答请配置 DASHSCOPE_API_KEY，或显式改回其他供应商。"
            )
        from app.llm.stub import StubClient

        return StubClient()
    raise LLMError(
        f"Unknown model provider '{provider}'. Expected one of: qwen, kimi, gemini, openai, deepseek, langchain, stub."
    )


def _text(settings: Settings, name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


def router_llm_available(settings: Settings | None) -> bool:
    """True only when the configured task provider itself has credentials.

    A leftover Moonshot or DeepSeek key must not make another provider look ready.
    Missing fields count as empty, so an offline fixture cannot look ready by accident.
    """
    if settings is None:
        return False
    provider = task_provider_name(settings)
    if provider == "qwen":
        return bool(_text(settings, "dashscope_api_key"))
    if provider == "deepseek":
        return bool(_text(settings, "deepseek_api_key"))
    if provider == "kimi":
        return bool(_text(settings, "moonshot_api_key") or _text(settings, "kimi_api_key"))
    if provider == "openai":
        return bool(_text(settings, "openai_api_key"))
    if provider == "gemini":
        return bool(_text(settings, "gemini_api_key"))
    if provider == "langchain":
        return bool(_text(settings, "langchain_model"))
    if provider == "stub":
        return _text(settings, "app_env").lower() != "production" and bool(getattr(settings, "allow_llm_stub", False))
    return False


def get_router_llm_client(settings: Settings) -> LLMClient:
    """Task understanding and extraction. Uses TASK_LLM_PROVIDER, not a leftover key."""
    provider = task_provider_name(settings)
    inner = _build_client(settings, provider, role="task")
    return MeteredClient(inner, provider=provider, model=task_model_name(settings))
