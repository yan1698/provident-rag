"""Replace default JWT / Postgres secrets and tighten CORS. Does not print secrets."""

from __future__ import annotations

import argparse
import secrets
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

WEAK_JWT = {"", "dev-only-change-me", "change-me-in-production", "secret"}
WEAK_PG = {"", "rag", "postgres", "password", "changeme"}
LOCAL_CORS = "http://localhost:5173,http://127.0.0.1:5173"


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


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


def _replace_url_password(url: str, password: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    auth, host = parts.netloc.rsplit("@", 1)
    user = auth.split(":", 1)[0]
    netloc = f"{user}:{quote(password, safe='')}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _write_env(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(path)


def _alter_postgres_password(password: str) -> None:
    sql_path = ROOT / "data" / "runtime" / ".rotate-pg.sql"
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    escaped = password.replace("'", "''")
    sql_path.write_text(f"ALTER USER rag WITH PASSWORD '{escaped}';\n", encoding="utf-8")
    try:
        copied = subprocess.run(
            ["docker", "compose", "cp", str(sql_path), "postgres:/tmp/rotate-pg.sql"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise SystemExit(
                "无法把 SQL 拷进 postgres 容器（docker compose cp 失败）。"
                f" {copied.stderr.strip() or copied.stdout.strip()}"
            )
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
                "-v",
                "ON_ERROR_STOP=1",
                "-f",
                "/tmp/rotate-pg.sql",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if ran.returncode != 0:
            raise SystemExit(f"ALTER USER 失败，未改 .env。 {ran.stderr.strip() or ran.stdout.strip()}")
    finally:
        if sql_path.is_file():
            sql_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate default JWT/Postgres secrets and tighten CORS")
    parser.add_argument("--force", action="store_true", help="Rotate even if current values are not defaults")
    args = parser.parse_args()

    _assert(ENV_PATH.is_file(), f"missing {ENV_PATH}")
    lines = _read_env(ENV_PATH)
    jwt = _env_value(lines, "JWT_SECRET")
    pg = _env_value(lines, "POSTGRES_PASSWORD")
    cors = _env_value(lines, "CORS_ORIGINS")
    db_url = _env_value(lines, "DATABASE_URL")

    updates: dict[str, str] = {}
    changed: list[str] = []

    if args.force or jwt is None or jwt.strip() in WEAK_JWT:
        updates["JWT_SECRET"] = secrets.token_urlsafe(32)
        changed.append("JWT_SECRET")

    rotate_pg = args.force or pg is None or pg.strip() in WEAK_PG
    new_pg = None
    if rotate_pg:
        new_pg = secrets.token_urlsafe(24)
        _alter_postgres_password(new_pg)
        updates["POSTGRES_PASSWORD"] = new_pg
        if db_url:
            updates["DATABASE_URL"] = _replace_url_password(db_url, new_pg)
        changed.append("POSTGRES_PASSWORD")
        if db_url:
            changed.append("DATABASE_URL")

    if cors is None or cors.strip() in {"", "*"}:
        updates["CORS_ORIGINS"] = LOCAL_CORS
        changed.append("CORS_ORIGINS")

    if not updates:
        print("secrets already rotated; CORS already tightened")
        return

    _write_env(ENV_PATH, _upsert(lines, updates))
    print("updated: " + ", ".join(changed))
    print("existing browser sessions must log in again")
    print("recreate the API container: docker compose up -d --force-recreate --no-build api")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
