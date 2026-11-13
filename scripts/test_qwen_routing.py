"""Provider routing without a network call or a real API key."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from types import SimpleNamespace

from app.agent.analyze import _llm_analyze
from app.agent.classify import _llm_classify
from app.agent.digest import _llm_digest
from app.agent.planner import llm_plan
from app.agent.rewrite import _llm_rewrite
from app.agent.state import AgentState, TaskSpec
from app.core.config import Settings
from app.generate.prompt import EvidenceItem
from app.generate.service import answer_from_evidence
from app.generate.synthesis import _llm_compose
from app.graph.extract import ExtractWindow, _extract_model, _extract_window
from app.llm import (
    LLMError,
    answer_model_name,
    answer_provider_name,
    get_llm_client,
    get_router_llm_client,
    router_llm_available,
    task_model_name,
)
from app.llm.base import fail_provider_call, public_llm_failure
from app.llm.qwen import normalize_qwen_base_url

CALLS: list[dict] = []


class _Message:
    def __init__(self, content: str):
        self.content = content
        self.reasoning_content = ""


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]
        self.usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)


class _Completions:
    def create(self, **kwargs):
        CALLS.append(kwargs)
        content = kwargs.pop("_content", None) or _Completions.content
        return _Response(content)


_Completions.content = "{}"


class _Chat:
    completions = _Completions()


class _OpenAI:
    def __init__(self, **kwargs):
        CALLS.append({"init": {key: kwargs.get(key) for key in ("base_url",)}})
        self.chat = _Chat()


def _settings(**overrides) -> Settings:
    data = dict(
        app_env="development",
        allow_llm_stub=False,
        llm_provider="kimi",
        moonshot_api_key="sk-leftover-moonshot",
        deepseek_api_key="sk-leftover-deepseek",
        dashscope_api_key="",
        answer_llm_provider="qwen",
        answer_llm_model="qwen3.8-max",
        task_llm_provider="qwen",
        task_llm_model="qwen3.8-flash",
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_kimi_model="kimi-k2.6",
        llm_deepseek_model="deepseek-v4-flash",
        graph_extract_deepseek_model="deepseek-chat",
    )
    data.update(overrides)
    return Settings(**data)


def _check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    print(f"PASS {name}")


def test_defaults() -> None:
    fields = Settings.model_fields
    _check("default answer model", fields["answer_llm_model"].default == "qwen3.8-max")
    _check("default task model", fields["task_llm_model"].default == "qwen3.8-flash")
    _check("default answer provider", fields["answer_llm_provider"].default == "qwen")
    _check("default task provider", fields["task_llm_provider"].default == "qwen")
    _check(
        "default base",
        fields["qwen_base_url"].default == "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def test_overrides_ignore_legacy_keys() -> None:
    settings = _settings(
        answer_llm_model="qwen3.8-max-custom",
        task_llm_model="qwen3.8-flash-custom",
        qwen_base_url="https://workspace.example/compatible-mode/v1/chat/completions",
        dashscope_api_key="test-key",
    )
    _check("answer override", answer_model_name(settings) == "qwen3.8-max-custom")
    _check("task override", task_model_name(settings) == "qwen3.8-flash-custom")
    _check(
        "base override",
        normalize_qwen_base_url(settings.qwen_base_url) == "https://workspace.example/compatible-mode/v1",
    )
    _check(_TEST_MATERIALS["text_008"], answer_provider_name(settings) == "qwen")
    _check(_TEST_MATERIALS["text_009"], router_llm_available(_settings()) is False)
    try:
        get_llm_client(_settings())
    except LLMError as exc:
        _check("missing qwen key", exc.kind == "auth" and "leftover" not in str(exc))
    else:
        raise SystemExit(_TEST_MATERIALS["text_026"])
    try:
        get_router_llm_client(_settings(deepseek_api_key="sk-leftover-deepseek"))
    except LLMError as exc:
        _check(_TEST_MATERIALS["text_036"], exc.kind == "auth" and "leftover" not in str(exc))
    else:
        raise SystemExit(_TEST_MATERIALS["text_027"])


def test_explicit_rollback() -> None:
    import openai

    openai.OpenAI = _OpenAI
    CALLS.clear()
    kimi = get_llm_client(_settings(answer_llm_provider="kimi"))
    _check("rollback answer provider", getattr(kimi, "_provider", "") == "kimi")
    _check("rollback answer base", CALLS[-1]["init"]["base_url"] == "https://api.moonshot.cn/v1")
    CALLS.clear()
    deepseek = get_router_llm_client(_settings(task_llm_provider="deepseek"))
    _check("rollback task provider", getattr(deepseek, "_provider", "") == "deepseek")
    _check("rollback task base", CALLS[-1]["init"]["base_url"].startswith("https://api.deepseek.com"))
    _check(
        _TEST_MATERIALS["text_010"],
        _extract_model(_settings(task_llm_provider="deepseek")) == "deepseek-chat",
    )


def test_call_routes() -> None:
    import openai

    openai.OpenAI = _OpenAI
    settings = _settings(dashscope_api_key="test-key")
    evidence = [EvidenceItem(index=1, chunk_id="c", doc_id="d", page=1, text=_TEST_MATERIALS["text_037"])]
    _Completions.content = _TEST_MATERIALS["text_001"]
    CALLS.clear()
    answer_from_evidence(_TEST_MATERIALS["text_011"], evidence, settings=settings)
    _check(_TEST_MATERIALS["text_012"], CALLS[-1].get("model") == "qwen3.8-max")

    state = AgentState(
        goal=_TEST_MATERIALS["text_028"], task_spec=TaskSpec(intent="explain", deliverable=_TEST_MATERIALS["text_039"])
    )
    _Completions.content = _TEST_MATERIALS["text_002"]
    CALLS.clear()
    _llm_compose(state, settings)
    _check(_TEST_MATERIALS["text_013"], CALLS[-1].get("model") == "qwen3.8-max")

    _Completions.content = _TEST_MATERIALS["text_003"]
    CALLS.clear()
    plan = llm_plan(_TEST_MATERIALS["text_014"], [], settings)
    _check(_TEST_MATERIALS["text_015"], CALLS[-1].get("model") == "qwen3.8-max" and plan.source == "llm")

    _Completions.content = '{"task_kind":"factoid"}'
    CALLS.clear()
    kind = _llm_classify(_TEST_MATERIALS["text_016"], settings)
    _check(_TEST_MATERIALS["text_017"], CALLS[-1].get("model") == "qwen3.8-flash" and kind == "factoid")

    _Completions.content = _TEST_MATERIALS["text_004"]
    CALLS.clear()
    spec = _llm_analyze(_TEST_MATERIALS["text_018"], settings)
    _check(_TEST_MATERIALS["text_019"], CALLS[-1].get("model") == "qwen3.8-flash" and spec is not None)

    _Completions.content = _TEST_MATERIALS["text_005"]
    CALLS.clear()
    _llm_rewrite(_TEST_MATERIALS["text_020"], "", settings, [])
    _check(_TEST_MATERIALS["text_021"], CALLS[-1].get("model") == "qwen3.8-flash")

    _Completions.content = _TEST_MATERIALS["text_006"]
    CALLS.clear()
    _llm_digest(_TEST_MATERIALS["text_022"], evidence, settings, "a.txt")
    _check(_TEST_MATERIALS["text_023"], CALLS[-1].get("model") == "qwen3.8-flash")

    _Completions.content = '{"entities":[],"edges":[],"headings":[]}'
    CALLS.clear()
    chunk = SimpleNamespace(text=_TEST_MATERIALS["text_029"])
    _extract_window(
        ExtractWindow(section=_TEST_MATERIALS["text_038"], chunks=[chunk]),
        settings,
        filename=_TEST_MATERIALS["text_030"],
    )
    _check(
        _TEST_MATERIALS["text_024"],
        CALLS[-1].get("model") == "qwen3.8-flash" and "response_format" in CALLS[-1],
    )


def test_errors_are_redacted() -> None:
    secret = _TEST_MATERIALS["text_007"]

    class Boom(Exception):
        def __init__(self, status: int | None, text: str):
            super().__init__(text)
            self.status_code = status

    cases = (
        (429, "rate limit account " + secret, "rate_limit", _TEST_MATERIALS["text_031"]),
        (401, "invalid_api_key " + secret, "auth", _TEST_MATERIALS["text_032"]),
        (403, "forbidden " + secret, "auth", _TEST_MATERIALS["text_033"]),
        (502, "bad gateway " + secret, "unavailable", _TEST_MATERIALS["text_034"]),
        (None, "timed out " + secret, "timeout", _TEST_MATERIALS["text_035"]),
    )
    for status, text, kind, label in cases:
        try:
            fail_provider_call("qwen", Boom(status, text))
        except LLMError as exc:
            _check(
                f"error {status}",
                exc.kind == kind and secret not in str(exc) and public_llm_failure(exc) == label,
                str(exc),
            )
        else:
            raise SystemExit(f"FAIL error {status}")


def main() -> int:
    test_defaults()
    test_overrides_ignore_legacy_keys()
    test_explicit_rollback()
    test_call_routes()
    test_errors_are_redacted()
    print(_TEST_MATERIALS["text_025"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
