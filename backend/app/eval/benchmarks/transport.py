"""Benchmark-only total-output limit; production prompts and provider implementation are unchanged."""

from __future__ import annotations

from app.llm import get_llm_client
from app.llm.meter import usage_from_openai


def bounded_client(settings, *, total_output_tokens: int, calls: list[dict]):
    client = get_llm_client(settings)
    resource = client._inner._client.chat.completions
    original_create = resource.create

    def create(**body):
        # Qwen max_tokens only caps final text. The total must include billed reasoning tokens.
        visible_limit = body.pop("max_tokens", None)
        body["max_completion_tokens"] = total_output_tokens
        response = original_create(**body)
        usage = usage_from_openai(response, provider="qwen", model=body["model"]).to_dict()
        usage.update(
            status="response_received",
            original_max_tokens=visible_limit,
            requested_max_completion_tokens=total_output_tokens,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
        )
        calls.append(usage)
        return response

    resource.create = create
    return client
