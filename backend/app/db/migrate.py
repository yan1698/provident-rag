"""Run Alembic against the configured database."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import engine


def _ensure_alembic() -> None:
    try:
        from alembic import command  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "alembic==1.14.1"],
            stdout=sys.stderr,
        )


def _alembic_api():
    _ensure_alembic()
    from alembic import command
    from alembic.config import Config

    return command, Config


def alembic_config(*, url: str | None = None):
    _command, Config = _alembic_api()
    ini = _ini_path()
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(_scripts_path(ini)))
    cfg.set_main_option("sqlalchemy.url", url or get_settings().sqlalchemy_database_url)
    return cfg


def _ini_path() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] / "alembic.ini",
        Path("/app/alembic.ini"),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("alembic.ini not found")


def _scripts_path(ini: Path) -> Path:
    candidates = (
        Path("/app/migrations"),
        ini.parent / "alembic",
    )
    for path in candidates:
        if (path / "env.py").is_file():
            return path
    raise FileNotFoundError("Alembic env.py not found")


def _table_exists(name: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = :name"),
            {"name": name},
        ).first()
    return row is not None


def upgrade_head(*, url: str | None = None) -> None:
    command, _Config = _alembic_api()
    command.upgrade(alembic_config(url=url), "head")


def stamp_head(*, url: str | None = None) -> None:
    command, _Config = _alembic_api()
    command.stamp(alembic_config(url=url), "head")


def ensure_schema() -> None:
    """Existing volumes get stamped once; empty databases run the full upgrade."""
    command, _Config = _alembic_api()
    if _table_exists("users") and not _table_exists("alembic_version"):
        command.stamp(alembic_config(), "0001baseline")
    upgrade_head()


def current() -> None:
    command, _Config = _alembic_api()
    command.current(alembic_config())


def _url_with_db(name: str) -> str:
    base = get_settings().sqlalchemy_database_url.rstrip("/")
    return base.rsplit("/", 1)[0] + "/" + name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alembic helpers")
    parser.add_argument("command", choices=("upgrade", "stamp", "current", "ensure"))
    parser.add_argument("--database", default="", help="Override database name in DATABASE_URL")
    args = parser.parse_args(argv)
    url = _url_with_db(args.database) if args.database else None
    if args.command == "upgrade":
        upgrade_head(url=url)
    elif args.command == "stamp":
        stamp_head(url=url)
    elif args.command == "current":
        current()
    else:
        ensure_schema()
    return 0


if __name__ == "__main__":
    sys.exit(main())
