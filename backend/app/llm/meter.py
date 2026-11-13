"""One exit for every LLM call. Bind a ledger to record usage; no ledger → no-op wrap."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

from app.llm.base import LLMClient, LLMError, LLMResponse, LLMUsage

_LEDGER: ContextVar[list[LLMUsage] | None] = ContextVar("provident_llm_ledger", default=None)
_STOP: ContextVar[Callable[[], bool] | None] = ContextVar("provident_llm_stop", default=None)
_SLEEP = time.sleep
_RETRYABLE = frozenset({"rate_limit", "timeout"})


@contextmanager
def collect_usage(ledger: list[LLMUsage]):
    token = _LEDGER.set(ledger)
    try:
        yield
    finally:
        _LEDGER.reset(token)


@contextmanager
def bind_llm_stop(stop: Callable[[], bool] | None):
    """Checked before each attempt, including the gap between retries."""
    token = _STOP.set(stop)
    try:
        yield
    finally:
        _STOP.reset(token)


def _retry_policy() -> tuple[int, float, float]:
    from app.core.config import get_settings

    settings = get_settings()
    retries = max(0, int(settings.llm_max_retries))
    base = max(0.0, float(settings.llm_retry_base_seconds))
    cap = max(base, float(settings.llm_retry_max_seconds))
    return retries, base, cap


def _retry_delay(attempt: int, base: float, cap: float, retry_after: float | None) -> float | None:
    """Seconds to wait, or None when Retry-After is longer than the cap."""
    if retry_after is not None and retry_after > cap:
        return None
    expo = min(cap, base * (2 ** max(attempt - 1, 0)))
    delay = expo + random.uniform(0, expo * 0.25)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, cap)


def _as_llm_error(exc: Exception, *, provider: str) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    return LLMError(f"{provider} unavailable", kind="unavailable")


def usage_from_openai(response, *, provider: str, model: str, purpose: str = "") -> LLMUsage:
    usage = getattr(response, "usage", None)
    return LLMUsage(
        provider=provider,
        model=model,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        purpose=purpose,
    )


class MeteredClient(LLMClient):
    """Wrap any provider. Appends to the bound ledger when one is active."""

    def __init__(self, inner: LLMClient, *, provider: str = "", model: str = ""):
        self._inner = inner
        self._provider = provider
        self._model = model

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
        kwargs.pop("stop", None)
        retries, base, cap = _retry_policy()
        attempt = 0
        while True:
            stopper = _STOP.get()
            if stopper is not None and stopper():
                raise LLMError("cancelled", kind="cancelled")
            attempt += 1
            from app.agent.run_budget import reserve, settle

            reservation = reserve(system, prompt, max_output_tokens, purpose, self._provider)
            attempt_kwargs = dict(kwargs)
            attempt_output = max_output_tokens
            if reservation is not None:
                attempt_output = reservation["wire_output_cap"]
                attempt_kwargs["total_output_cap"] = attempt_output
                attempt_kwargs["request_timeout"] = reservation["timeout_seconds"]
                attempt_kwargs["thinking_budget"] = reservation["thinking_tokens"]
            started = time.perf_counter()
            try:
                response = self._inner.generate(
                    system=system,
                    prompt=prompt,
                    max_output_tokens=attempt_output,
                    temperature=temperature,
                    purpose=purpose,
                    **attempt_kwargs,
                )
            except Exception as exc:
                settle(reservation, status="failed")
                elapsed = int((time.perf_counter() - started) * 1000)
                llm_exc = _as_llm_error(exc, provider=self._provider or "llm")
                self._remember(
                    LLMUsage(
                        provider=self._provider,
                        model=self._model,
                        purpose=purpose,
                        latency_ms=elapsed,
                    ),
                    status="error",
                )
                if llm_exc.kind not in _RETRYABLE or attempt > retries:
                    raise llm_exc
                delay = _retry_delay(attempt, base, cap, llm_exc.retry_after)
                if delay is None:
                    raise llm_exc
                stopper = _STOP.get()
                if stopper is not None and stopper():
                    raise LLMError("cancelled", kind="cancelled")
                _SLEEP(delay)
                continue
            elapsed = int((time.perf_counter() - started) * 1000)
            usage = replace(
                response.usage,
                provider=response.usage.provider or self._provider,
                model=response.usage.model or self._model,
                purpose=purpose or response.usage.purpose,
                latency_ms=elapsed or response.usage.latency_ms,
            )
            self._remember(usage, status="ok")
            settle(reservation, usage, status="success")
            return LLMResponse(
                text=response.text, usage=usage, finish_reason=getattr(response, "finish_reason", "") or ""
            )

    def _remember(self, usage: LLMUsage, *, status: str) -> None:
        ledger = _LEDGER.get()
        if ledger is not None:
            ledger.append(usage)
        from app.llm.ledger import record_llm_call

        record_llm_call(usage, status=status)


def selfcheck() -> None:
    from app.llm.ledger import capture_calls
    from app.llm.stub import StubClient

    client = MeteredClient(StubClient(), provider="stub", model="stub")
    ledger: list[LLMUsage] = []
    with collect_usage(ledger):
        first = client.generate(
            system="sys",
            prompt="[S1] (doc_id=x, page=1)\n视觉层级用字号建立扫描路径。\n",
            max_output_tokens=64,
            temperature=0.0,
            purpose="answer",
        )
    if not first.text or len(ledger) != 1 or ledger[0].purpose != "answer":
        raise SystemExit(f"usage ledger failed: {ledger}")

    recorded: list[dict] = []
    with capture_calls(recorded):
        client.generate(
            system="sys",
            prompt="ok",
            max_output_tokens=8,
            temperature=0.0,
            purpose="digest",
        )
    if not recorded or recorded[0].get("status") != "ok" or recorded[0].get("purpose") != "digest":
        raise SystemExit(f"capture ok failed: {recorded}")

    class Boom(LLMClient):
        def generate(self, **kwargs):
            raise RuntimeError("provider down")

    failed: list[dict] = []
    with capture_calls(failed):
        try:
            MeteredClient(Boom(), provider="stub", model="stub").generate(
                system="sys",
                prompt="x",
                max_output_tokens=8,
                temperature=0.0,
                purpose="compose",
            )
        except LLMError as exc:
            if exc.kind != "unavailable":
                raise SystemExit(f"unclassified failure: {exc.kind}")
        else:
            raise SystemExit("provider failure was swallowed")
    if not failed or failed[0].get("status") != "error" or len(failed) != 1:
        raise SystemExit(f"error capture failed: {failed}")

    global _SLEEP, _retry_policy
    saved_sleep, saved_policy = _SLEEP, _retry_policy
    delays: list[float] = []

    def _fast(seconds: float) -> None:
        delays.append(seconds)

    def _two_retries() -> tuple[int, float, float]:
        return 2, 0.0, 8.0

    class Flaky(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise LLMError("limited", kind="rate_limit", retry_after=0)
            return LLMResponse(text="ok", usage=LLMUsage(provider="stub", model="stub", purpose="answer"))

    class AuthBoom(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            raise LLMError("denied", kind="auth")

    class LongWait(LLMClient):
        def generate(self, **kwargs):
            raise LLMError("limited", kind="rate_limit", retry_after=30)

    _SLEEP = _fast
    _retry_policy = _two_retries
    try:
        flaky = Flaky()
        flaky_calls: list[dict] = []
        with capture_calls(flaky_calls):
            recovered = MeteredClient(flaky, provider="stub", model="stub").generate(
                system="sys",
                prompt="x",
                max_output_tokens=8,
                temperature=0.0,
                purpose="answer",
            )
        if recovered.text != "ok" or flaky.calls != 3 or len(delays) != 2:
            raise SystemExit(f"retry failed: calls={flaky.calls} delays={delays} rows={flaky_calls}")
        if sum(1 for row in flaky_calls if row.get("status") == "error") != 2:
            raise SystemExit(f"retry ledger failed: {flaky_calls}")

        auth = AuthBoom()
        before = len(delays)
        try:
            MeteredClient(auth, provider="stub", model="stub").generate(
                system="sys",
                prompt="x",
                max_output_tokens=8,
                temperature=0.0,
                purpose="answer",
            )
        except LLMError as exc:
            if exc.kind != "auth" or auth.calls != 1 or len(delays) != before:
                raise SystemExit(f"auth was retried: {exc.kind} calls={auth.calls}")
        else:
            raise SystemExit("auth error was swallowed")

        try:
            MeteredClient(LongWait(), provider="stub", model="stub").generate(
                system="sys",
                prompt="x",
                max_output_tokens=8,
                temperature=0.0,
                purpose="answer",
            )
        except LLMError as exc:
            if exc.kind != "rate_limit" or len(delays) != before:
                raise SystemExit(f"long Retry-After was slept: {delays}")
        else:
            raise SystemExit("long Retry-After was retried")

        seen = {"n": 0}

        class Once(LLMClient):
            def generate(self, **kwargs):
                seen["n"] += 1
                raise LLMError("limited", kind="rate_limit", retry_after=0)

        try:
            with bind_llm_stop(lambda: seen["n"] >= 1):
                MeteredClient(Once(), provider="stub", model="stub").generate(
                    system="sys",
                    prompt="x",
                    max_output_tokens=8,
                    temperature=0.0,
                    purpose="answer",
                )
        except LLMError as exc:
            if exc.kind != "cancelled" or seen["n"] != 1:
                raise SystemExit(f"cancel during retry failed: {exc.kind} n={seen['n']}")
        else:
            raise SystemExit("cancel during retry continued")
    finally:
        _SLEEP = saved_sleep
        _retry_policy = saved_policy
    print("llm usage unit ok")


if __name__ == "__main__":
    selfcheck()
