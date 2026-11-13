"""Write EVAL_GATE_USER_ID for an existing username. Does not print the UUID."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def _read_env(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").splitlines()


def _env_value(lines: list[str], key: str) -> str | None:
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        name, value = raw.split("=", 1)
        if name.strip() == key:
            return value
    return None


def _upsert(lines: list[str], updates: dict[str, str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            name = line.split("=", 1)[0].strip()
            if name in updates:
                out.append(f"{name}={updates[name]}")
                seen.add(name)
                continue
        out.append(line)
    for name, value in updates.items():
        if name not in seen:
            out.append(f"{name}={value}")
    return out


def _grant_admin(username: str) -> None:
    safe = username.replace("'", "''")
    ran = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "rag",
            "-d",
            "rag",
            "-c",
            f"UPDATE users SET role = 'admin' WHERE lower(username) = lower('{safe}');",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if ran.returncode != 0:
        raise SystemExit(f"grant admin failed: {(ran.stderr or ran.stdout).strip()}")


def _lookup_user_id(username: str) -> str:
    # username is constrained at register time; still escape quotes for the SQL literal.
    safe = username.replace("'", "''")
    ran = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "rag",
            "-d",
            "rag",
            "-t",
            "-A",
            "-c",
            f"SELECT id FROM users WHERE lower(username) = lower('{safe}') LIMIT 1;",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if ran.returncode != 0:
        raise SystemExit(f"psql failed: {(ran.stderr or ran.stdout).strip()}")
    return (ran.stdout or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grant admin role and pin EVAL_GATE_USER_ID")
    parser.add_argument("--username", default="", help="Existing account; may also be set through EVAL_GATE_USERNAME")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing EVAL_GATE_USER_ID")
    args = parser.parse_args()

    if not ENV_PATH.is_file():
        raise SystemExit(f"missing {ENV_PATH}")
    lines = _read_env(ENV_PATH)
    current = (_env_value(lines, "EVAL_GATE_USER_ID") or "").strip()
    username = (args.username or _env_value(lines, "EVAL_GATE_USERNAME") or "").strip()
    if not username:
        parser.error("provide --username or configure EVAL_GATE_USERNAME for the intended existing account")
    if current and not args.force:
        if username:
            _grant_admin(username)
        print("EVAL_GATE_USER_ID already set")
        return
    user_id = _lookup_user_id(username)
    if not user_id:
        print(f"no user named {username!r}; left username fallback")
        return
    _grant_admin(username)
    ENV_PATH.write_bytes(("\n".join(_upsert(lines, {"EVAL_GATE_USER_ID": user_id})) + "\n").encode("utf-8"))
    print(f"granted admin to existing account {username!r}")
    print("recreate the API container: docker compose up -d --force-recreate --no-build api")


if __name__ == "__main__":
    main()
