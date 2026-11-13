"""Copy reviewed working-tree files into a standalone directory, without Git history.

This command reads Git's file list. It never initializes, stages, commits, or pushes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath

from check_repository import check, git
from lib.release_policy import local_test_material

ROOT_FILES = {
    ".editorconfig",
    ".env.example",
    ".gitattributes",
    ".gitignore",
    ".prettierrc.json",
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    "docker-compose.yml",
    "docker-compose.override.yml",
    "ruff.toml",
}
SOURCE_DIRS = {".github", "backend", "web", "scripts", "benchmarks", "docker", "data", "media"}


def prepare(root: Path, destination: Path) -> dict:
    root, destination = root.resolve(), destination.resolve()
    release_root = (root / "public-release").resolve()
    if release_root not in destination.parents:
        raise ValueError("destination must be a new directory inside public-release/")
    if destination.exists():
        raise FileExistsError("destination already exists; choose a new directory to preserve its contents")
    _, issues = check(root)
    if issues:
        raise ValueError("repository check failed:\n" + "\n".join(issues))
    names = sorted(
        {
            os.fsdecode(n)
            for n in git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
            if n
        }
    )
    selected = []
    for name in names:
        path = PurePosixPath(name)
        if name not in ROOT_FILES and path.parts[0] not in SOURCE_DIRS:
            continue
        if local_test_material(name):
            continue
        source = root / name
        if not source.is_file():
            continue
        if source.is_symlink() or root not in source.resolve().parents:
            raise ValueError(f"refusing symlink or external source: {name}")
        selected.append(name)
    required = {"README.md", "LICENSE", ".gitignore", ".env.example"}
    if not required.issubset(selected):
        raise ValueError("release must include README.md, LICENSE, .gitignore, and .env.example")
    destination.mkdir(parents=True)
    files = []
    for name in selected:
        source, target = root / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        body = target.read_bytes()
        if body != source.read_bytes():
            raise RuntimeError(f"source changed during export; review export before publishing: {name}")
        files.append({"path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    manifest = {
        "format": 1,
        "contains_git_history": False,
        "test_materials_included": False,
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    destination.with_name(destination.name + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("public-release/provident-rag"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.destination if args.destination.is_absolute() else root / args.destination
    manifest = prepare(root, destination)
    print(f"Prepared {manifest['file_count']} files ({manifest['total_bytes']} bytes) at {destination.resolve()}")
    print("No Git metadata copied. Repository creation, commit, and push remain manual.")


if __name__ == "__main__":
    main()
