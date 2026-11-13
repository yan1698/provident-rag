"""Day 1 ingest acceptance: upload PDF/TXT/MD, check metadata and chunk sizes."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.accept import assert_ok
from lib.http import API_BASE, get_json, post_file

RAW = ROOT / "data" / "raw"
MIN_TOKENS = 140
MAX_TOKENS = 220


def _check_chunk_fields(payload: dict, *, expect_band: bool) -> None:
    assert_ok(payload.get("chunk_count", 0) >= 1, "chunk_count < 1")
    chunks = payload["chunks"]
    assert_ok(len(chunks) == payload["chunk_count"], "chunk_count mismatch")
    for chunk in chunks:
        for key in ("id", "page", "start", "end", "token_count", "hash", "preview"):
            assert_ok(key in chunk, f"chunk missing {key}")
        assert_ok(chunk["end"] >= chunk["start"], "end < start")
        assert_ok(isinstance(chunk["page"], int) and chunk["page"] >= 1, "invalid page")
        assert_ok(len(chunk["hash"]) == 64, _TEST_MATERIALS["text_007"])

    counts = [c["token_count"] for c in chunks]
    too_big = [n for n in counts if n > MAX_TOKENS]
    assert_ok(not too_big, f"chunks above {MAX_TOKENS}: {too_big}")
    body_counts = counts[:-1] if len(counts) > 1 else counts
    if expect_band:
        outliers = [n for n in body_counts if not (MIN_TOKENS <= n <= MAX_TOKENS)]
        assert_ok(not outliers, f"{_TEST_MATERIALS['text_002']}{MIN_TOKENS}-{MAX_TOKENS}: {outliers}")
        assert_ok(counts[-1] <= MAX_TOKENS, f"{_TEST_MATERIALS['text_003']}{counts[-1]}")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _make_pdf(paragraphs: list[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "48 780 Td"]
    for para in paragraphs:
        wrapped = [para[i : i + 86] for i in range(0, len(para), 86)] or [""]
        for line in wrapped:
            ascii_line = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in line)
            commands.append(f"({_pdf_escape(ascii_line)}) Tj")
            commands.append("0 -13 Td")
        commands.append("0 -8 Td")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        ),
        b"4 0 obj << /Length " + str(len(stream)).encode("ascii") + b" >> stream\n" + stream + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    header = b"%PDF-1.4\n"
    xref_positions = [0]
    body = b""
    offset = len(header)
    for obj in objects:
        xref_positions.append(offset)
        body += obj
        offset += len(obj)
    xref = [b"xref\n0 6\n0000000000 65535 f \n"]
    for pos in xref_positions[1:]:
        xref.append(f"{pos:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n" + str(len(header) + len(body)).encode("ascii") + b"\n%%EOF\n"
    )
    return header + body + b"".join(xref) + trailer


def _long_english() -> str:
    para = _TEST_MATERIALS["text_004"]
    return "\n\n".join(para for _ in range(28))


def main() -> None:
    health = get_json(f"{API_BASE}/health", timeout=60)
    assert_ok(health.get("status") == "ok", f"health not ok: {health}")
    print("health ok")

    md_path = RAW / "sample.md"
    txt_path = RAW / "sample.txt"
    assert_ok(md_path.is_file(), f"missing {md_path}")
    assert_ok(txt_path.is_file(), f"missing {txt_path}")

    md = post_file("sample.md", md_path.read_bytes(), "text/markdown", timeout=60)
    _check_chunk_fields(md, expect_band=False)
    print(f"md ok  doc_id={md['doc_id']} chunks={md['chunk_count']} stats={md['token_stats']}")

    md_again = post_file("sample.md", md_path.read_bytes(), "text/markdown", timeout=60)
    assert_ok(md_again.get("reused") is True, _TEST_MATERIALS["text_005"])
    assert_ok(md_again["doc_id"] == md["doc_id"], "idempotent upload changed doc_id")
    assert_ok(md_again["chunk_count"] == md["chunk_count"], "idempotent upload changed chunk_count")
    print("idempotency ok")

    txt = post_file("sample.txt", txt_path.read_bytes(), "text/plain", timeout=60)
    _check_chunk_fields(txt, expect_band=False)
    print(f"txt ok  doc_id={txt['doc_id']} chunks={txt['chunk_count']}")

    long_txt = post_file("generated.txt", _long_english().encode("utf-8"), "text/plain", timeout=60)
    _check_chunk_fields(long_txt, expect_band=True)
    print(f"{_TEST_MATERIALS['text_001']}{long_txt['chunk_count']} stats={long_txt['token_stats']}")

    pdf_paras = [_long_english()[i : i + 400] for i in range(0, 2400, 400)]
    pdf = post_file("sample.pdf", _make_pdf(pdf_paras), "application/pdf", timeout=60)
    _check_chunk_fields(pdf, expect_band=False)
    assert_ok(pdf["chunks"][0]["page"] >= 1, "pdf page missing")
    print(f"pdf ok  doc_id={pdf['doc_id']} chunks={pdf['chunk_count']}")

    print(_TEST_MATERIALS["text_006"])


if __name__ == "__main__":
    sys.exit(main())
