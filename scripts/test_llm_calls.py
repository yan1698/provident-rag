"""P1-5: llm_calls table accepts a metered write (runs inside the API container)."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ran = subprocess.run(
        ["docker", "compose", "exec", "-T", "api", "python", "-m", "app.llm.ledger"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = (ran.stdout or "") + (ran.stderr or "")
    if ran.returncode != 0 or "llm_calls persist ok" not in out:
        raise SystemExit(f"FAIL: {out.strip()}")
    print("llm_calls persist ok")


if __name__ == "__main__":
    main()
