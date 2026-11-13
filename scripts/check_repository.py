"""Check upload candidates without printing credential values. Standard library only.

Default: tracked working-tree files plus non-ignored new files.
--staged: exact Git index contents, including files hidden by .gitignore.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

from lib.release_policy import local_test_material

MAX_BYTES = 5 * 1024 * 1024
PATTERNS = (
    ("provider API key", rb"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}\b"),
    ("GitHub token", rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    ("AWS access key", rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ("Google API key", rb"\bAIza[A-Za-z0-9_-]{35}\b"),
    ("Slack token", rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    ("private key", rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ("JWT token", rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
)
SECRET_NAME = re.compile(r"(?:API_KEY|SECRET|PASSWORD|ACCESS_TOKEN|AUTH_TOKEN|PRIVATE_KEY)$")
LOCAL_DIRS = {"node_modules", ".venv", "venv", "__pycache__", ".deps", ".cache", "uploads", "logs"}
PRIVATE_NAMES = {
    ".env",
    ".envrc",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
}
PRIVATE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".dump",
    ".bak",
    ".backup",
    ".log",
    ".pid",
    ".rdb",
}
LOCAL_NOTES = {"RAG项目完成手册.html", "PHASE1_GENERAL_EXPLANATION.md", "UNIFIED_QA_IMPLEMENTATION.md"}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def env_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.*)$", line)
        if match:
            values[match[1]] = match[2].strip().strip("\"'")
    return values


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    if name in LOCAL_NOTES:
        return True
    if path.name in PRIVATE_NAMES or path.suffix.lower() in PRIVATE_SUFFIXES:
        return True
    if path.name.startswith(".env.") and path.name != ".env.example":
        return True
    if path.name.startswith("service-account") and path.suffix == ".json":
        return True
    if LOCAL_DIRS.intersection(path.parts):
        return True
    if path.parts[0] in {"docs", "tmp", "dist", "build", "public-release"} or path.parts[0].startswith("tmp-"):
        return True
    if name.startswith(("benchmarks/reports/", "data/benchmarks/", "web/dist/")):
        return True
    if name.startswith("data/runtime/") and name != "data/runtime/.gitkeep":
        return True
    return name.startswith("data/raw/") and name not in {
        "data/raw/.gitkeep",
        "data/raw/sample.md",
        "data/raw/sample.txt",
    }


def check(root: Path, staged: bool = False, public: bool = False) -> tuple[int, list[str]]:
    args = (
        ("ls-files", "-z", "--cached") if staged else ("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    )
    names = sorted({os.fsdecode(name) for name in git(root, *args).split(b"\0") if name})
    defaults = {"dev-only-change-me", "change-me-in-production"}
    known = {
        value.encode(): key
        for key, value in env_values(root / ".env").items()
        if SECRET_NAME.search(key) and len(value) >= 10 and value not in defaults
    }
    issues = []
    count = 0
    for name in names:
        path = root / name
        if not staged and not path.is_file():
            continue
        count += 1
        if public and local_test_material(name):
            issues.append(f"{name}: concrete test materials must remain local")
            continue
        if private_path(name):
            issues.append(f"{name}: local/private file must not be committed")
            continue
        if staged:
            body = git(root, "show", f":{name}")
        elif path.is_symlink():
            issues.append(f"{name}: review symlink before publishing")
            continue
        else:
            if path.stat().st_size > MAX_BYTES:
                issues.append(f"{name}: file exceeds 5 MiB; keep generated artifacts locally")
                continue
            body = path.read_bytes()
        if len(body) > MAX_BYTES:
            issues.append(f"{name}: file exceeds 5 MiB; keep generated artifacts locally")
            continue
        for label, pattern in PATTERNS:
            match = re.search(pattern, body)
            if match:
                line = body[: match.start()].count(b"\n") + 1
                issues.append(f"{name}:{line}: possible {label} (value withheld)")
        for value, key in known.items():
            if value in body:
                issues.append(f"{name}: contains local .env credential {key} (value withheld)")
        if PurePosixPath(name).name == ".env.example":
            for line in body.decode("utf-8-sig").splitlines():
                match = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
                if match and match[1].endswith(("API_KEY", "ACCESS_TOKEN", "PRIVATE_KEY")) and match[2].strip():
                    issues.append(f"{name}: template credential {match[1]} must be empty")
    return count, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="inspect the exact Git index")
    parser.add_argument(
        "--public", action="store_true", help="also reject local question sets, answers, and text fixtures"
    )
    args = parser.parse_args()
    root = Path(os.fsdecode(git(Path.cwd(), "rev-parse", "--show-toplevel")).strip())
    count, issues = check(root, args.staged, args.public)
    if issues:
        print("Repository check failed:\n" + "\n".join(issues))
        return 1
    print(f"Repository check passed ({count} files; known credential patterns and private paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
