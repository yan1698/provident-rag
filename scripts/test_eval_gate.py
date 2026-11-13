"""Eval gate: admin role wins; UUID pin / username stay as fallback."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.eval_gate import role_is_admin, user_matches_eval_gate


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    _assert(role_is_admin("admin"), _TEST_MATERIALS["text_001"])
    _assert(role_is_admin("Admin"), _TEST_MATERIALS["text_002"])
    _assert(not role_is_admin("user"), _TEST_MATERIALS["text_003"])
    _assert(not role_is_admin(""), _TEST_MATERIALS["text_004"])
    _assert(not role_is_admin(None), _TEST_MATERIALS["text_005"])

    maintainer = uuid4()
    other = uuid4()
    _assert(
        user_matches_eval_gate(
            user_id=maintainer,
            username="someone-else",
            auth_enabled=True,
            gate_user_id=str(maintainer),
            gate_username="traveller",
        ),
        _TEST_MATERIALS["text_006"],
    )
    _assert(
        not user_matches_eval_gate(
            user_id=other,
            username="traveller",
            auth_enabled=True,
            gate_user_id=str(maintainer),
            gate_username="traveller",
        ),
        _TEST_MATERIALS["text_007"],
    )
    _assert(
        not user_matches_eval_gate(
            user_id=maintainer,
            username="traveller",
            auth_enabled=True,
            gate_user_id="not-a-uuid",
            gate_username="traveller",
        ),
        _TEST_MATERIALS["text_008"],
    )
    _assert(
        user_matches_eval_gate(
            user_id=other,
            username="Traveller",
            auth_enabled=True,
            gate_user_id="",
            gate_username="traveller",
        ),
        _TEST_MATERIALS["text_009"],
    )
    _assert(
        not user_matches_eval_gate(
            user_id=other,
            username="traveller",
            auth_enabled=False,
            gate_user_id=str(maintainer),
            gate_username="traveller",
        ),
        _TEST_MATERIALS["text_010"],
    )
    print(_TEST_MATERIALS["text_011"])


if __name__ == "__main__":
    main()
