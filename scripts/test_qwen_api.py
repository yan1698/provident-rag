"""Optional live Qwen smoke test. Does nothing unless --live is passed."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import time

from app.core.config import get_settings
from app.llm import LLMError, get_llm_client
from app.llm.base import public_llm_failure


def main() -> int:
    if "--live" not in sys.argv:
        print(_TEST_MATERIALS["text_003"])
        return 0
    settings = get_settings()
    if not (settings.dashscope_api_key or "").strip():
        print(_TEST_MATERIALS["text_004"])
        return 0
    started = time.perf_counter()
    try:
        response = get_llm_client(settings).generate(
            system=_TEST_MATERIALS["text_005"],
            prompt=_TEST_MATERIALS["text_006"],
            max_output_tokens=8,
            temperature=0.0,
            purpose="smoke",
        )
    except LLMError as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        print(f"{_TEST_MATERIALS['text_002']}{exc.kind} public={public_llm_failure(exc)} elapsed_ms={elapsed}")
        return 1
    elapsed = int((time.perf_counter() - started) * 1000)
    print(
        f"{_TEST_MATERIALS['text_001']}{response.usage.provider} model={response.usage.model} elapsed_ms={elapsed} chars={len(response.text or '')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
