from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import NoReturn


class LLMError(RuntimeError):
    """Raised when the configured LLM provider is not usable (missing key, request failure, ...)."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "unavailable",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read a numeric Retry-After header. HTTP-date values are ignored."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def fail_provider_call(provider: str, exc: Exception) -> NoReturn:
    """Replace a provider exception with a short error. The original body is not kept."""
    status = getattr(exc, "status_code", None)
    blob = f"{type(exc).__name__} {status or ''}".lower()
    if status == 429 or "ratelimit" in blob:
        kind = "rate_limit"
    elif status in {401, 403}:
        kind = "auth"
    elif "timeout" in blob:
        kind = "timeout"
    else:
        kind = "unavailable"
        text = str(exc).lower()
        if "429" in text or "rate limit" in text or "too many requests" in text:
            kind = "rate_limit"
        elif "401" in text or "403" in text or "authentication" in text or "invalid_api_key" in text:
            kind = "auth"
        elif "timeout" in text or "timed out" in text:
            kind = "timeout"
    raise LLMError(
        f"{provider} {kind.replace('_', ' ')}",
        kind=kind,
        retry_after=retry_after_seconds(exc),
    ) from None


def public_llm_failure(exc: BaseException) -> str:
    """Short client-facing reason. Provider bodies can contain account ids and key fragments."""
    kind = str(getattr(exc, "kind", "") or "")
    labels = {
        "rate_limit": "模型服务限流",
        "auth": "模型服务鉴权失败",
        "timeout": "模型服务超时",
        "unavailable": "模型服务暂时不可用",
    }
    if kind in labels:
        return labels[kind]
    text = str(exc).lower()
    if "429" in text or "rate limit" in text:
        return labels["rate_limit"]
    if "401" in text or "403" in text or "auth failed" in text:
        return labels["auth"]
    if "timeout" in text or "timed out" in text:
        return labels["timeout"]
    return labels["unavailable"]


@dataclass(frozen=True)
class LLMUsage:
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    purpose: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: str = ""


def llm_timeout(settings):
    """Connect, read, and write budgets for one model call. Pool wait matches connect."""
    import httpx

    connect = float(settings.llm_connect_timeout_seconds)
    return httpx.Timeout(
        connect=connect,
        read=float(settings.llm_read_timeout_seconds),
        write=float(settings.llm_write_timeout_seconds),
        pool=connect,
    )


class LLMClient(ABC):
    @abstractmethod
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
        """Return the model's answer text plus token usage for a single-turn call."""
