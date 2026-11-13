"""Helpers smoke test for user-defined multi-tags (no DB required)."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def test_normalize():
    from app.tag.service import normalize_tag_color, normalize_tag_name

    assert normalize_tag_name(_TEST_MATERIALS["text_002"]) == _TEST_MATERIALS["text_001"]
    assert normalize_tag_color("") == ""
    assert normalize_tag_color("#a1b2c3") == "#A1B2C3"
    try:
        normalize_tag_color("red")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    test_normalize()
    print("PASS")
