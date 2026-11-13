"""Structural citation checks stay off the Verifier status enum."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import uuid
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.generate.prompt import EvidenceItem  # noqa: E402
from app.verify.structure import assess_citations, assess_snapshot  # noqa: E402


def _assert(cond: bool, detail: object) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {detail}")


def _item(**kwargs) -> EvidenceItem:
    base = {
        "index": 1,
        "chunk_id": "c1",
        "doc_id": "d1",
        "page": 1,
        "text": _TEST_MATERIALS["text_001"],
        "chunk_hash": "abc",
    }
    base.update(kwargs)
    return EvidenceItem(**base)


def main() -> None:
    owner = uuid.uuid4()
    other = uuid.uuid4()
    row = {
        "doc_id": "d1",
        "page": 2,
        "start_offset": 10,
        "end_offset": 20,
        "text": _TEST_MATERIALS["text_002"],
        "chunk_hash": "abc",
        "section_path": _TEST_MATERIALS["text_003"],
        "owner_id": owner,
        "ingest_version": 3,
    }
    ok = assess_citations([_item()], {"c1": row}, owner_id=owner)
    _assert(ok[0]["status"] == "ok", ok)
    _assert(ok[0]["locator"] == "p2:10-20", ok)
    _assert(ok[0]["section_path"] == _TEST_MATERIALS["text_005"], ok)
    _assert(ok[0]["ingest_version"] == 3, ok)
    _assert("text" not in ok[0], ok)

    missing = assess_citations([_item()], {}, owner_id=owner)
    _assert(missing[0]["status"] == "missing", missing)

    hidden = assess_citations([_item()], {"c1": {**row, "owner_id": other}}, owner_id=owner)
    _assert(hidden[0]["status"] == "missing", hidden)
    _assert("section_path" not in hidden[0], hidden)

    stale = assess_citations([_item()], {"c1": {**row, "chunk_hash": "zzz"}}, owner_id=owner)
    _assert(stale[0]["status"] == "hash_mismatch", stale)

    drifted = assess_citations(
        [_item(chunk_hash="")],
        {"c1": {**row, "chunk_hash": "abc", "text": _TEST_MATERIALS["text_009"]}},
        owner_id=owner,
    )
    _assert(drifted[0]["status"] == "text_mismatch", drifted)

    snapshot = [vars(_item()), {"index": 2, "text": _TEST_MATERIALS["text_006"]}]
    with patch("app.verify.structure.load_citation_rows", return_value={"c1": row}) as load:
        checked = assess_snapshot(object(), snapshot, owner_id=owner)
        _assert([item["status"] for item in checked] == ["ok", "unverifiable"], checked)
        _assert(load.call_count == 1, _TEST_MATERIALS["text_007"])
        _assert(load.call_args.kwargs["owner_id"] == owner, _TEST_MATERIALS["text_008"])
    _assert("status" not in snapshot[0], "snapshot was mutated")
    print(_TEST_MATERIALS["text_004"])


if __name__ == "__main__":
    main()
