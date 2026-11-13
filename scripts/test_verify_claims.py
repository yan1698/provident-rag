"""Claim segmentation checks: Markdown scaffolding must not count as uncited claims."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.verify.claims import split_claims  # noqa: E402


def _assert(cond: bool, detail: object) -> None:
    if not cond:
        print(f"FAIL: {detail}")
        raise SystemExit(1)


ANSWER = _TEST_MATERIALS["text_001"]


def main() -> None:
    claims = split_claims(ANSWER)
    texts = [claim.text for claim in claims]

    _assert(not any(text.startswith("|:") for text in texts), texts)
    _assert(_TEST_MATERIALS["text_005"] not in texts, texts)
    _assert(_TEST_MATERIALS["text_006"] not in texts, texts)

    rows = [claim for claim in claims if claim.text.startswith(_TEST_MATERIALS["text_007"])]
    _assert(len(rows) == 1, texts)
    _assert(rows[0].citation_indices == [2], rows[0])

    uncited = [claim.text for claim in claims if not claim.citation_indices]
    _assert(not uncited, uncited)

    # A plain sentence without a citation is still an uncited claim.
    loose = split_claims(_TEST_MATERIALS["text_002"])
    _assert(len(loose) == 1 and not loose[0].citation_indices, loose)

    # Citation-only lines still merge back, and short fragments stay dropped.
    merged = split_claims(_TEST_MATERIALS["text_003"])
    _assert(len(merged) == 1 and merged[0].citation_indices == [1], merged)

    # Insertion / multi-clause splits used by inductive verification.
    from app.verify.inductive import split_subclaims

    parts = split_subclaims(_TEST_MATERIALS["text_004"])
    _assert(len(parts) >= 2, parts)

    print("verify claims ok")


if __name__ == "__main__":
    main()
