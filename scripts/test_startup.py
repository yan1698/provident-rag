"""Production startup gates (no Docker / no Settings import)."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.startup import assert_production_ready, collect_production_problems


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    dev = SimpleNamespace(
        app_env="development",
        auth_enabled=False,
        jwt_secret=_TEST_MATERIALS["text_001"],
        allow_llm_stub=True,
        postgres_password="rag",
        cors_origins="*",
        cors_origin_list=["*"],
    )
    _assert(collect_production_problems(dev) == [], collect_production_problems(dev))
    assert_production_ready(dev)

    bad = SimpleNamespace(
        app_env="production",
        auth_enabled=False,
        jwt_secret=_TEST_MATERIALS["text_002"],
        allow_llm_stub=True,
        postgres_password="rag",
        cors_origins="*",
        cors_origin_list=["*"],
    )
    problems = collect_production_problems(bad)
    _assert(_TEST_MATERIALS["text_003"] in problems, problems)
    _assert(_TEST_MATERIALS["text_004"] in problems, problems)
    _assert(_TEST_MATERIALS["text_005"] in problems, problems)
    _assert(_TEST_MATERIALS["text_006"] in problems, problems)
    _assert(_TEST_MATERIALS["text_007"] in problems, problems)
    try:
        assert_production_ready(bad)
    except RuntimeError as exc:
        _assert(_TEST_MATERIALS["text_010"] in str(exc), str(exc))
    else:
        raise SystemExit(_TEST_MATERIALS["text_008"])

    good = SimpleNamespace(
        app_env="production",
        auth_enabled=True,
        jwt_secret=_TEST_MATERIALS["text_009"],
        allow_llm_stub=False,
        postgres_password="rotated-once",
        cors_origins="http://localhost:5173",
        cors_origin_list=["http://localhost:5173"],
    )
    _assert(collect_production_problems(good) == [], collect_production_problems(good))
    assert_production_ready(good)
    print("startup gates ok")


if __name__ == "__main__":
    main()
