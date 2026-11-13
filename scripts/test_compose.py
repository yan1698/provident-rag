"""P2-1: production compose has no source mounts; override restores them."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    base = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    override = (ROOT / _TEST_MATERIALS["text_013"]).read_text(encoding="utf-8")
    ignore = ROOT / "backend" / ".dockerignore"
    _assert(ignore.is_file(), "backend/.dockerignore missing")

    _assert("./backend/app" not in base, _TEST_MATERIALS["text_001"])
    _assert("./web:" not in base, _TEST_MATERIALS["text_002"])
    _assert("./backend/alembic" not in base, _TEST_MATERIALS["text_003"])
    _assert("./scripts:" not in base, _TEST_MATERIALS["text_004"])
    _assert("./data/runtime/models" in base, _TEST_MATERIALS["text_005"])
    _assert("./data/eval" in base, _TEST_MATERIALS["text_006"])
    _assert("${POSTGRES_PORT:-5432}:5432" in base, _TEST_MATERIALS["text_007"])
    _assert("${REDIS_PORT:-6379}:6379" in base, _TEST_MATERIALS["text_008"])

    _assert("./backend/app:/app/app:ro" in override, _TEST_MATERIALS["text_009"])
    _assert("./web:/app/web:ro" in override, _TEST_MATERIALS["text_010"])
    _assert("./backend/alembic:/app/migrations:ro" in override, _TEST_MATERIALS["text_011"])
    _assert("./scripts:/app/scripts:ro" in override, _TEST_MATERIALS["text_012"])
    print("compose split ok")


if __name__ == "__main__":
    main()
