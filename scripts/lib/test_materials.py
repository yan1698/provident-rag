"""Load locally supplied text cases without bundling them with test logic."""

from __future__ import annotations

import json
import os
from pathlib import Path

MISSING_MATERIALS = 77


def load_test_materials(script: str) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    directory = Path(os.environ.get("PROVIDENT_TEST_MATERIALS_DIR", root / "data/test-materials"))
    name = Path(script).stem
    path = directory / f"{name}.json"
    if not path.is_file():
        print(f"SKIP {name}: local text materials are unavailable; configure PROVIDENT_TEST_MATERIALS_DIR.")
        raise SystemExit(MISSING_MATERIALS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        texts = payload["texts"]
        if payload.get("schema_version") != 1 or not isinstance(texts, dict):
            raise ValueError
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in texts.items()):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise SystemExit(
            f"Invalid local materials for {name}; expected schema_version=1 and a texts mapping."
        ) from None
    return texts
