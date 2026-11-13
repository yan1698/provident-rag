"""Identify local test materials that must stay out of source releases."""

from pathlib import PurePosixPath


def local_test_material(name: str) -> bool:
    path = PurePosixPath(name)
    if path.name == ".gitkeep":
        return False
    return name.startswith(("data/test-materials/", "data/eval/", "data/raw/", "benchmarks/fixtures/"))
