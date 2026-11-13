"""Per-run, persisted Token reservations, separate from any currency authorization."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from app.llm.base import LLMError

_CURRENT = ContextVar("longform_run_budget", default=None)
_CHECKPOINT = ContextVar("longform_budget_checkpoint", default=None)


def initialize(state, settings):
    if state.task_kind not in {"synthesis", "digest"}:
        return
    if not state.run_budget:
        state.run_budget = {
            "max_calls": max(1, int(getattr(settings, "agent_longform_max_model_calls", 8))),
            "max_input_tokens": max(1, int(getattr(settings, "agent_longform_input_tokens", 64000))),
            "max_output_tokens": max(1, int(getattr(settings, "agent_longform_output_tokens", 16000))),
            "final_output_reserve": max(1, int(getattr(settings, "agent_longform_final_output_reserve", 6000))),
            "analysis_output_cap": max(1, int(getattr(settings, "agent_longform_analysis_output_tokens", 1800))),
            "model_timeout_seconds": max(
                1.0, min(300.0, float(getattr(settings, "agent_longform_model_timeout_seconds", 240)))
            ),
            "thinking_tokens": max(0, int(getattr(settings, "agent_longform_thinking_tokens", 1024))),
            "analysis_thinking_tokens": max(0, int(getattr(settings, "agent_longform_analysis_thinking_tokens", 384))),
            "calls": [],
            "currency_status": "unpriced; Token balance is not currency authorization",
            "recovery": "failed reservations remain charged; no automatic model replay",
        }


@contextmanager
def bind(state, settings, *, checkpoint=None):
    initialize(state, settings)
    token = _CURRENT.set(state.run_budget if state.run_budget else None)
    saved = _CHECKPOINT.set(checkpoint)
    try:
        yield
    finally:
        _CURRENT.reset(token)
        _CHECKPOINT.reset(saved)


def reserve(system, prompt, output, purpose, provider):
    b = _CURRENT.get()
    if b is None or provider == "stub":
        return None
    if provider != "qwen":
        raise LLMError("此供应商尚未验证思考与正文的合计上限，受控长回答未发出请求。", kind="budget")
    calls = b["calls"]
    used_in = sum(c["input_reserved"] for c in calls)
    used_out = sum(c["output_reserved"] for c in calls)
    incoming = len((system + prompt).encode("utf8")) + 2048
    protected = b["final_output_reserve"] if purpose not in {"compose", "answer"} else 0
    # In a bounded run the writer gets the explicitly reserved full-output cap;
    # do not exhaust reasoning at the legacy answer-only limit of 1200.
    requested = b["final_output_reserve"] if purpose == "compose" else output
    if purpose == "analyze":
        requested = b["analysis_output_cap"]
    outgoing = min(requested, b["max_output_tokens"] - used_out - protected - 12)
    if len(calls) >= b["max_calls"] or used_in + incoming > b["max_input_tokens"] or outgoing < 1:
        b["stop_reason"] = "model Token/request budget exhausted; currency balance not inferred"
        raise LLMError("运行Token或请求预算不足，保留已读证据并部分交付。", kind="budget")
    record = {
        "purpose": purpose,
        "provider": provider,
        "input_reserved": incoming,
        "output_reserved": outgoing + 12,
        "wire_output_cap": outgoing,
        "timeout_seconds": b["model_timeout_seconds"],
        "thinking_tokens": min(
            b["analysis_thinking_tokens"] if purpose == "analyze" else b["thinking_tokens"], max(0, outgoing // 3)
        ),
        "status": "reserved",
    }
    calls.append(record)
    persist = _CHECKPOINT.get()
    if persist is not None and not persist():
        raise LLMError("运行执行权已失效，未发送模型请求。", kind="cancelled")
    return record


def settle(record, usage=None, *, status="failed"):
    if record is None:
        return
    record["status"] = status
    if usage and usage.prompt_tokens > 0 and usage.completion_tokens > 0:
        if usage.prompt_tokens > record["input_reserved"] or usage.completion_tokens > record["output_reserved"]:
            b = _CURRENT.get()
            b["stop_reason"] = "provider usage exceeded reservation"
            b["max_calls"] = len(b["calls"])  # fail closed on another attempt
        record["actual_input_tokens"] = usage.prompt_tokens
        record["actual_output_tokens"] = usage.completion_tokens
        record["input_reserved"] = max(usage.prompt_tokens, 0)
        record["output_reserved"] = max(usage.completion_tokens, 0)
    persist = _CHECKPOINT.get()
    if persist is not None:
        persist()
