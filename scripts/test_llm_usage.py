"""Metered LLM client records usage without a live provider."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    try:
        from app.llm.meter import selfcheck

        selfcheck()
        return
    except ModuleNotFoundError:
        if os.environ.get("GITHUB_ACTIONS") or not (ROOT / ".env").is_file():
            raise
    ran = subprocess.run(
        ["docker", "compose", "exec", "-T", "api", "python", "-m", "app.llm.meter"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = (ran.stdout or "") + (ran.stderr or "")
    if ran.returncode != 0 or _TEST_MATERIALS["text_002"] not in out:
        raise SystemExit(f"FAIL: {out.strip()}")
    print(_TEST_MATERIALS["text_001"])


if __name__ == "__main__":
    main()
