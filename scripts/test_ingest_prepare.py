"""Offline checks for cleaning, structure-aware chunks, and BM25 bigrams."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "backend"))


def assert_ok(cond: bool, message: object) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


from app.ingest.chunker import chunk_pages, count_tokens
from app.ingest.clean import decode_bytes, merge_hard_breaks, normalize_markup
from app.ingest.parser import ParsedPage, parse_document
from app.search.sparse import tokenize


def _page(text: str) -> list[ParsedPage]:
    return [ParsedPage(page=1, text=text, start_offset=0, end_offset=len(text))]


def _check_clean() -> None:
    joined = merge_hard_breaks(_TEST_MATERIALS["text_006"])
    assert_ok(_TEST_MATERIALS["text_019"] in joined, joined)
    assert_ok(_TEST_MATERIALS["text_020"] not in joined, joined)
    assert_ok(_TEST_MATERIALS["text_021"] in joined, joined)

    normalized = normalize_markup(_TEST_MATERIALS["text_007"])
    assert_ok(_TEST_MATERIALS["text_022"] in normalized, normalized)
    assert_ok("\n\n\n" not in normalized, normalized)

    decoded = decode_bytes(_TEST_MATERIALS["text_033"].encode("gb18030"))
    assert_ok(decoded == _TEST_MATERIALS["text_023"], decoded)

    english = merge_hard_breaks(_TEST_MATERIALS["text_008"])
    assert_ok("agreement" in english, english)


def _check_chunks() -> None:
    alpha = _TEST_MATERIALS["text_024"] * 80 + _TEST_MATERIALS["text_009"]
    beta = _TEST_MATERIALS["text_025"] * 80 + _TEST_MATERIALS["text_010"]
    text = f"{_TEST_MATERIALS['text_001']}{alpha}{_TEST_MATERIALS['text_002']}{beta}{_TEST_MATERIALS['text_003']}"
    drafts = chunk_pages(
        _page(text), min_tokens=40, max_tokens=120, overlap_tokens=10, filename=_TEST_MATERIALS["text_026"]
    )
    assert_ok(drafts, "expected chunks")
    alpha_hits = [item for item in drafts if _TEST_MATERIALS["text_034"] in item.text]
    beta_hits = [item for item in drafts if _TEST_MATERIALS["text_035"] in item.text]
    assert_ok(alpha_hits and beta_hits, _TEST_MATERIALS["text_011"])
    assert_ok(all(_TEST_MATERIALS["text_037"] not in item.text for item in alpha_hits), _TEST_MATERIALS["text_012"])
    assert_ok(all("Alpha" in item.section_path for item in alpha_hits), alpha_hits[0].section_path)
    assert_ok(all("Beta" in item.section_path for item in beta_hits), beta_hits[0].section_path)
    assert_ok(all(item.token_count <= 120 for item in drafts), [item.token_count for item in drafts])

    fenced = "```\nkeep\n\nboth lines\n```\n\n" + (_TEST_MATERIALS["text_027"] * 30)
    fenced_drafts = chunk_pages(_page(fenced), min_tokens=20, max_tokens=80, overlap_tokens=0)
    assert_ok(any("keep" in item.text and "both lines" in item.text for item in fenced_drafts), "fence was split")

    table = _TEST_MATERIALS["text_013"] + (_TEST_MATERIALS["text_028"] * 40)
    table_drafts = chunk_pages(_page(table), min_tokens=20, max_tokens=80, overlap_tokens=0)
    assert_ok(
        any(
            _TEST_MATERIALS["text_039"] in item.text and _TEST_MATERIALS["text_040"] in item.text
            for item in table_drafts
        ),
        "table was split",
    )

    prose = "\n\n".join(f"{_TEST_MATERIALS['text_005']}{i}。" + (_TEST_MATERIALS["text_038"] * 25) for i in range(12))
    long_drafts = chunk_pages(
        _page(prose), min_tokens=80, max_tokens=140, overlap_tokens=20, filename=_TEST_MATERIALS["text_029"]
    )
    assert_ok(len(long_drafts) >= 2, _TEST_MATERIALS["text_014"])
    assert_ok(all(item.token_count <= 140 for item in long_drafts), [item.token_count for item in long_drafts])
    body = long_drafts[:-1]
    assert_ok(all(item.token_count >= 80 for item in body), [item.token_count for item in body])
    shared = long_drafts[0].text[-12:]
    assert_ok(shared and shared in long_drafts[1].text, _TEST_MATERIALS["text_015"])
    assert_ok(count_tokens(_TEST_MATERIALS["text_036"]) == 2, "token counter")


def _check_pdf() -> None:
    import fitz

    doc = fitz.open()
    for index in range(3):
        page = doc.new_page()
        page.insert_text((72, 40), "ACME HANDBOOK", fontsize=9)
        page.insert_text((72, 80), "Scope of Work", fontsize=18)
        page.insert_text((72, 120), "The agree-", fontsize=11)
        page.insert_text((72, 136), f"{_TEST_MATERIALS['text_004']}{index + 1}.", fontsize=11)
        page.insert_text((72, 760), "Confidential", fontsize=9)
    raw = doc.tobytes()
    doc.close()
    pages = parse_document("sample.pdf", raw)
    blob = "\n".join(page.text for page in pages)
    assert_ok("agreement" in blob, blob)
    assert_ok("ACME HANDBOOK" not in blob, blob)
    assert_ok("Confidential" not in blob, blob)
    assert_ok("Scope of Work" in blob, blob)
    assert_ok(any(page.text.lstrip().startswith("#") for page in pages), blob)


def _check_sparse() -> None:
    contract = tokenize(_TEST_MATERIALS["text_016"])
    swapped = tokenize(_TEST_MATERIALS["text_017"])
    assert_ok(_TEST_MATERIALS["text_030"] in contract, contract)
    assert_ok(_TEST_MATERIALS["text_031"] not in swapped, swapped)
    assert_ok(_TEST_MATERIALS["text_032"] in swapped, swapped)


def main() -> None:
    _check_clean()
    _check_chunks()
    _check_pdf()
    _check_sparse()
    print(_TEST_MATERIALS["text_018"])


if __name__ == "__main__":
    sys.exit(main())
