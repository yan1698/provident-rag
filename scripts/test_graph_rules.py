"""G0 skeleton graph: heading tree + quote gate. HTTP optional if API is up."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.corpus.catalog import headings_in_order, iter_headings
from app.graph.rules import build_document_graph
from app.graph.schema import quote_in_chunk

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
TREE = _TEST_MATERIALS["text_007"]


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _chunk(text: str, *, n: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        page=1,
        start_offset=n,
        text=text,
        chunk_hash=f"{n:064x}"[:64],
    )


def _doc(name: str = "handbook.md") -> SimpleNamespace:
    return SimpleNamespace(filename=name, owner_id=None)


def test_heading_stack() -> None:
    levels = [(lv, title) for lv, title, _ in iter_headings(TREE)]
    _assert(levels[0] == (1, _TEST_MATERIALS["text_020"]), f"top heading: {levels}")
    _assert(("3", "BM25") == (str(levels[3][0]), levels[3][1]), f"BM25 level: {levels}")
    drafts = build_document_graph(_doc(), [_chunk(TREE)])
    pairs = {(e.src_name, e.dst_name, e.dst_kind) for e in drafts.edges}
    _assert(("BM25", _TEST_MATERIALS["text_021"], "doc_section") in pairs, f"BM25 parent: {pairs}")
    _assert(
        (_TEST_MATERIALS["text_022"], _TEST_MATERIALS["text_023"], "doc_section") in pairs, f"sparse parent: {pairs}"
    )
    _assert(("Dense", _TEST_MATERIALS["text_024"], "doc_section") in pairs, f"dense parent: {pairs}")
    _assert(
        (_TEST_MATERIALS["text_025"], _TEST_MATERIALS["text_026"], "doc_section") in pairs, f"rerank parent: {pairs}"
    )
    _assert((_TEST_MATERIALS["text_027"], "handbook.md", "document") in pairs, f"root parent: {pairs}")
    _assert(("Dense", _TEST_MATERIALS["text_028"], "doc_section") not in pairs, f"{_TEST_MATERIALS['text_001']}{pairs}")
    _assert((_TEST_MATERIALS["text_029"], "Dense", "doc_section") not in pairs, f"siblings reversed: {pairs}")
    for edge in drafts.edges:
        _assert(edge.quote, f"edge missing quote: {edge}")
        _assert(quote_in_chunk(edge.quote, TREE) is not None, f"{_TEST_MATERIALS['text_003']}{edge.quote!r}")
    for ent in drafts.entities:
        _assert(quote_in_chunk(ent.quote, TREE) is not None, f"{_TEST_MATERIALS['text_004']}{ent.quote!r}")
    again = build_document_graph(_doc(), [_chunk(TREE)])
    _assert(len(again.edges) == len(drafts.edges), _TEST_MATERIALS["text_010"])
    print(
        f"{_TEST_MATERIALS['text_002']}{len(drafts.edges)} entities={len({(e.kind, e.name) for e in drafts.entities})}"
    )


def test_filename_fallback() -> None:
    text = _TEST_MATERIALS["text_008"]
    drafts = build_document_graph(_doc("notes.md"), [_chunk(text)])
    kinds = {e.kind for e in drafts.entities}
    _assert("document" in kinds and "doc_section" in kinds, f"fallback kinds: {kinds}")
    pairs = {(e.src_name, e.dst_name, e.relation) for e in drafts.edges}
    _assert(("notes.md", "notes.md", "part_of") in pairs, f"fallback edge: {pairs}")
    print("filename fallback ok")


def test_markdown_filename_is_atx_hint() -> None:
    from app.graph.schema import is_markdown_filename

    _assert(
        is_markdown_filename(_TEST_MATERIALS["text_030"]) and is_markdown_filename("notes.markdown"),
        _TEST_MATERIALS["text_011"],
    )
    _assert(
        not is_markdown_filename(_TEST_MATERIALS["text_034"]) and not is_markdown_filename("guide.pdf"),
        _TEST_MATERIALS["text_012"],
    )
    print("atx hint ok")


def test_quote_gate() -> None:
    text = _TEST_MATERIALS["text_009"]
    span = quote_in_chunk(_TEST_MATERIALS["text_013"], text)
    _assert(span is not None, _TEST_MATERIALS["text_014"])
    _assert(quote_in_chunk(_TEST_MATERIALS["text_031"], text) is not None, _TEST_MATERIALS["text_015"])
    _assert(quote_in_chunk(_TEST_MATERIALS["text_032"], text) is None, _TEST_MATERIALS["text_016"])
    print("quote gate ok")


def test_headings_shared_with_catalog() -> None:
    titles = headings_in_order([TREE], fallback="x")
    _assert(_TEST_MATERIALS["text_033"] in titles and "BM25" in titles, f"catalog headings: {titles}")
    print("catalog reuse ok")


def _request(url: str, data: bytes | None = None, method: str | None = None) -> tuple[int, dict]:
    req = Request(url, data=data, method=method or ("GET" if data is None else "POST"))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=60) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else {}
    except HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"detail": raw.decode("utf-8", errors="replace")}
        return exc.code, payload
    except URLError:
        return 0, {}


def test_http_contract() -> None:
    status, payload = _request(f"{BASE}/")
    if status == 0:
        print(_TEST_MATERIALS["text_017"])
        return
    if status == 401:
        print(_TEST_MATERIALS["text_018"])
        return
    _assert(status == 200, f"root HTTP {status}: {payload}")
    if payload.get("graph") != "/graph":
        print(f"HTTP skipped: API {payload.get('version')}{_TEST_MATERIALS['text_005']}")
        return
    code, graph = _request(f"{BASE}/graph")
    if code == 401:
        print(_TEST_MATERIALS["text_019"])
        return
    _assert("pending_docs" in graph and "entities" in graph, f"graph shape: {graph}")
    _assert("pending_extract" in graph and "concepts" in graph, f"extract status missing: {graph}")
    _assert(graph.get("source") in {"rules", "llm", "mixed"}, f"graph source: {graph}")
    metrics_code, metrics = _request(f"{BASE}/metrics")
    _assert(metrics_code == 200, f"metrics HTTP {metrics_code}")
    _assert("graph" in metrics, f"metrics missing graph: {metrics}")
    if graph.get("pending_docs", 0) > 0 or graph.get("stale_mentions", 0) > 0:
        post_code, posted = _request(f"{BASE}/graph", data=b"{}")
        _assert(post_code == 200, f"POST /graph HTTP {post_code}: {posted}")
        deadline = time.time() + 30
        while time.time() < deadline:
            _, cur = _request(f"{BASE}/graph")
            if not cur.get("running"):
                graph = cur
                break
            time.sleep(0.4)
        _assert(not graph.get("running"), f"{_TEST_MATERIALS['text_006']}{graph}")
        _assert(graph.get("error") in (None, ""), f"graph job error: {graph.get('error')}")
    print(f"HTTP pending={graph.get('pending_docs')} entities={graph.get('entities')} edges={graph.get('edges')}")


def main() -> None:
    test_quote_gate()
    test_headings_shared_with_catalog()
    test_heading_stack()
    test_filename_fallback()
    test_markdown_filename_is_atx_hint()
    test_http_contract()
    print("test_graph_rules ok")


if __name__ == "__main__":
    main()
