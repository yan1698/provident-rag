"""Run a host script inside rag-api when Anaconda lacks app deps."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def can_use_local_api() -> bool:
    """Docker fallback is for a local stack only. CI has no .env and no api container."""
    if os.environ.get("GITHUB_ACTIONS"):
        return False
    return (ROOT / ".env").is_file()


def run_script_in_api(script: Path, ok_token: str, extras: list[Path] | None = None) -> None:
    if not can_use_local_api():
        raise SystemExit(f"FAIL {script.name}: no local api container on this machine")
    dest = f"/tmp/{script.name}"
    copied = subprocess.run(
        ["docker", "compose", "cp", str(script), f"api:{dest}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if copied.returncode != 0:
        raise SystemExit(f"FAIL docker cp {script.name}: {(copied.stderr or copied.stdout).strip()}")
    for extra in extras or []:
        extra_dest = f"/tmp/{extra.name}"
        ran = subprocess.run(
            ["docker", "compose", "cp", str(extra), f"api:{extra_dest}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if ran.returncode != 0:
            raise SystemExit(f"FAIL docker cp {extra.name}: {(ran.stderr or ran.stdout).strip()}")
    ran = subprocess.run(
        ["docker", "compose", "exec", "-T", "api", "python", dest],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = (ran.stdout or "") + (ran.stderr or "")
    if ran.returncode != 0 or ok_token not in out:
        raise SystemExit(f"FAIL: {out.strip()}")
    sys.stdout.write(out if out.endswith("\n") else out + "\n")
