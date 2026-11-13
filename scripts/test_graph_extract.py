"""G1 extraction: quote gate, unknown labels dropped, JSON repair, no-key fallback."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import importlib.util

from types import SimpleNamespace

from app.graph.extract import (
    ALREADY_EXTRACTED_NOTE,
    NO_KEY_NOTE,
    SKIP_SHORT_NOTE,
    drafts_from_extraction,
    drafts_from_window,
    estimate_extract,
    extract_document,
    extract_token_budget,
    heading_tree_from_hits,
    pack_extract_windows,
    should_extract,
)
from app.graph.order import entity_sort_key
from app.graph.resolve import names_should_merge, resolve_drafts
from app.graph.schema import GraphDrafts, HeadingHit, prefix_compatible

_json_io = importlib.util.spec_from_file_location("provident_json_io", ROOT / "backend" / "app" / "llm" / "json_io.py")
_json_mod = importlib.util.module_from_spec(_json_io)
_json_io.loader.exec_module(_json_mod)
parse_json_object = _json_mod.parse_json_object


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


TEXT = _TEST_MATERIALS["text_003"]


def test_quote_mismatch_dropped() -> None:
    payload = {
        "entities": [
            {"name": "BM25", "kind": "method", "quote": _TEST_MATERIALS["text_038"]},
            {"name": _TEST_MATERIALS["text_039"], "kind": "concept", "quote": _TEST_MATERIALS["text_040"]},
        ],
        "edges": [
            {
                "src": "BM25",
                "dst": _TEST_MATERIALS["text_041"],
                "src_kind": "method",
                "dst_kind": "concept",
                "relation": "part_of",
                "quote": _TEST_MATERIALS["text_042"],
            },
            {
                "src": "BM25",
                "dst": _TEST_MATERIALS["text_043"],
                "relation": "causes",
                "quote": _TEST_MATERIALS["text_044"],
            },
        ],
    }
    drafts = drafts_from_extraction(payload, chunk_id="c1", chunk_hash="h", text=TEXT)
    names = {item.name for item in drafts.entities}
    _assert("BM25" in names, f"kept BM25: {names}")
    _assert(_TEST_MATERIALS["text_024"] not in names, f"ungrounded entity leaked: {names}")
    rels = {(e.src_name, e.relation, e.dst_name) for e in drafts.edges}
    _assert(("BM25", "part_of", _TEST_MATERIALS["text_045"]) in rels, f"kept grounded edge: {rels}")
    _assert(("BM25", "causes", _TEST_MATERIALS["text_046"]) not in rels, f"ungrounded edge leaked: {rels}")
    print("quote gate ok")


def test_unknown_relation_rejected() -> None:
    payload = {
        "entities": [{"name": "BM25", "kind": "widget", "quote": _TEST_MATERIALS["text_047"]}],
        "edges": [
            {
                "src": "BM25",
                "dst": _TEST_MATERIALS["text_048"],
                "relation": "invented_by",
                "quote": _TEST_MATERIALS["text_049"],
            }
        ],
    }
    drafts = drafts_from_extraction(payload, chunk_id="c1", chunk_hash="h", text=TEXT)
    _assert(not drafts.entities, f"bad kind kept: {drafts.entities}")
    _assert(not drafts.edges, f"open relation kept: {drafts.edges}")
    print("closed set ok")


def test_json_repair() -> None:
    raw = _TEST_MATERIALS["text_004"]
    data = parse_json_object(raw)
    _assert(isinstance(data, dict) and "entities" in data, f"json repair: {data}")
    print("json repair ok")


def test_resolve_prefix_veto() -> None:
    _assert(
        not prefix_compatible(_TEST_MATERIALS["text_050"], _TEST_MATERIALS["text_051"]), _TEST_MATERIALS["text_006"]
    )
    _assert(
        not names_should_merge(
            _TEST_MATERIALS["text_052"], _TEST_MATERIALS["text_053"], aliases=[_TEST_MATERIALS["text_100"]]
        ),
        "alias+prefix veto",
    )
    drafts = GraphDrafts()
    from app.graph.schema import DraftEntity

    drafts.entities = [
        DraftEntity(_TEST_MATERIALS["text_025"], "method", "q", "c", "h", aliases=["CE"]),
        DraftEntity(_TEST_MATERIALS["text_026"], "method", "q", "c", "h", aliases=[_TEST_MATERIALS["text_089"]]),
    ]
    resolved = resolve_drafts(drafts)
    names = {item.name for item in resolved.entities}
    _assert(len(names) == 2, f"{_TEST_MATERIALS['text_001']}{names}")
    print("resolve veto ok")


def test_no_key_note() -> None:
    _assert(_TEST_MATERIALS["text_054"] in NO_KEY_NOTE and _TEST_MATERIALS["text_055"] in NO_KEY_NOTE, NO_KEY_NOTE)
    print(_TEST_MATERIALS["text_007"])


def test_dropped_quotes_counted() -> None:
    payload = {
        "entities": [
            {"name": "BM25", "kind": "method", "quote": _TEST_MATERIALS["text_056"]},
            {"name": _TEST_MATERIALS["text_057"], "kind": "concept", "quote": _TEST_MATERIALS["text_058"]},
        ],
        "edges": [
            {
                "src": "BM25",
                "dst": _TEST_MATERIALS["text_059"],
                "relation": "causes",
                "quote": _TEST_MATERIALS["text_060"],
            }
        ],
    }
    drafts = drafts_from_extraction(payload, chunk_id="c1", chunk_hash="h", text=TEXT)
    _assert(drafts.dropped >= 2, f"dropped quotes: {drafts.dropped}")
    _assert(any(item.name == "BM25" for item in drafts.entities), "kept grounded entity")
    print("dropped count ok")


def test_extract_token_budget_ignores_compose_cap() -> None:
    settings = SimpleNamespace(llm_max_output_tokens=1024)
    _assert(extract_token_budget(settings) == 4096, _TEST_MATERIALS["text_008"])
    settings.graph_extract_max_tokens = 2000
    _assert(extract_token_budget(settings) == 2000, _TEST_MATERIALS["text_009"])
    print("extract budget ok")


def test_completion_text_uses_reasoning() -> None:
    empty = SimpleNamespace(
        message=SimpleNamespace(content=_TEST_MATERIALS["text_090"], reasoning_content='{"ok": true}')
    )
    _assert(_json_mod.completion_text(empty) == '{"ok": true}', _TEST_MATERIALS["text_010"])
    filled = SimpleNamespace(message=SimpleNamespace(content=_TEST_MATERIALS["text_091"], reasoning_content="ignored"))
    _assert(_json_mod.completion_text(filled) == '{"a":1}', _TEST_MATERIALS["text_011"])
    print("reasoning fallback ok")


def test_skip_short_body() -> None:
    short = [SimpleNamespace(text=_TEST_MATERIALS["text_061"], page=1, start_offset=0)]
    ok, reason = should_extract(short, SimpleNamespace(graph_extract_min_chars=800))
    _assert(not ok and reason == SKIP_SHORT_NOTE, f"short: {ok} {reason}")
    forced, _ = should_extract(short, SimpleNamespace(graph_extract_min_chars=800), force=True)
    _assert(forced, _TEST_MATERIALS["text_012"])
    long = [SimpleNamespace(text=_TEST_MATERIALS["text_092"] * 400, page=1, start_offset=0)]
    ok_long, _ = should_extract(long, SimpleNamespace(graph_extract_min_chars=800))
    _assert(ok_long, _TEST_MATERIALS["text_013"])
    print("short skip ok")


def test_pack_windows_by_section() -> None:
    chunks = [
        SimpleNamespace(id="1", text=_TEST_MATERIALS["text_093"] * 20, page=1, start_offset=0),
        SimpleNamespace(id="2", text=_TEST_MATERIALS["text_094"] * 20, page=1, start_offset=10),
        SimpleNamespace(id="3", text=_TEST_MATERIALS["text_095"] * 20, page=1, start_offset=20),
    ]
    windows, skipped = pack_extract_windows(chunks, filename=_TEST_MATERIALS["text_027"], cap=10, max_chars=6000)
    _assert(skipped == 0, f"skipped: {skipped}")
    _assert(len(windows) == 2, f"windows: {len(windows)}")
    _assert(windows[0].section == _TEST_MATERIALS["text_062"] and len(windows[0].chunks) == 2, windows[0])
    _assert(windows[1].section == _TEST_MATERIALS["text_063"] and len(windows[1].chunks) == 1, windows[1])
    print("pack windows ok")


def test_window_quote_aligns_to_any_chunk() -> None:
    chunks = [
        SimpleNamespace(id="a", chunk_hash="ha", text=_TEST_MATERIALS["text_064"]),
        SimpleNamespace(id="b", chunk_hash="hb", text=_TEST_MATERIALS["text_065"]),
    ]
    payload = {
        "entities": [{"name": "BM25", "kind": "method", "quote": _TEST_MATERIALS["text_066"]}],
    }
    drafts = drafts_from_window(payload, chunks)
    _assert(len(drafts.entities) == 1, drafts.entities)
    _assert(drafts.entities[0].chunk_id == "b", drafts.entities[0].chunk_id)
    print(_TEST_MATERIALS["text_014"])


def test_already_extracted_skipped() -> None:
    from datetime import datetime, timezone

    doc = SimpleNamespace(filename=_TEST_MATERIALS["text_028"], graph_extracted_at=datetime.now(timezone.utc))
    drafts = extract_document(
        doc, [SimpleNamespace(text=_TEST_MATERIALS["text_101"] * 400, page=1, start_offset=0)], None, force=True
    )
    _assert(drafts.note == ALREADY_EXTRACTED_NOTE, drafts.note)
    _assert(drafts.calls == 0 and not drafts.entities, _TEST_MATERIALS["text_015"])
    print(_TEST_MATERIALS["text_016"])


def test_extract_allows_plain_text() -> None:
    long = [SimpleNamespace(text=_TEST_MATERIALS["text_096"] * 400, page=1, start_offset=0)]
    for name in ("guide.txt", "handbook.pdf", "notes.md"):
        est = estimate_extract(SimpleNamespace(filename=name, graph_extracted_at=None), long, None)
        _assert(est["calls"] >= 1 and est["skip"] == "", f"{name}: {est}")
    print(_TEST_MATERIALS["text_017"])


def test_headings_must_be_in_text() -> None:
    text = _TEST_MATERIALS["text_005"]
    payload = {
        "entities": [],
        "edges": [],
        "headings": [
            {"title": _TEST_MATERIALS["text_067"], "level": 1, "quote": _TEST_MATERIALS["text_068"]},
            {"title": _TEST_MATERIALS["text_069"], "level": 1, "quote": _TEST_MATERIALS["text_070"]},
            {"title": _TEST_MATERIALS["text_071"], "level": 2, "quote": _TEST_MATERIALS["text_072"]},
        ],
    }
    drafts = drafts_from_extraction(payload, chunk_id="c1", chunk_hash="h", text=text)
    names = {item.name for item in drafts.entities if item.kind == "doc_section"}
    _assert(_TEST_MATERIALS["text_029"] in names, names)
    _assert(_TEST_MATERIALS["text_030"] not in names, names)
    _assert(_TEST_MATERIALS["text_031"] not in names, names)
    rels = {(e.src_name, e.relation, e.dst_name) for e in drafts.edges}
    _assert((_TEST_MATERIALS["text_073"], "part_of", _TEST_MATERIALS["text_074"]) in rels, rels)
    _assert((_TEST_MATERIALS["text_075"], "part_of", _TEST_MATERIALS["text_076"]) not in rels, rels)
    print("heading probe ok")


def test_window_entity_cap() -> None:
    chunks = [SimpleNamespace(id="c", chunk_hash="h", text=_TEST_MATERIALS["text_097"] * 3)]
    entities = [
        {
            "name": f"{_TEST_MATERIALS['text_002']}{i}",
            "kind": "concept",
            "quote": _TEST_MATERIALS["text_032"],
        }
        for i in range(12)
    ]
    drafts = drafts_from_window({"entities": entities, "edges": []}, chunks)
    _assert(len(drafts.entities) == 8, len(drafts.entities))
    _assert(drafts.dropped >= 4, drafts.dropped)
    print("entity cap ok")


def test_estimate_skips_extracted() -> None:
    stamped = SimpleNamespace(filename=_TEST_MATERIALS["text_033"], graph_extracted_at="done")
    est = estimate_extract(
        stamped, [SimpleNamespace(text=_TEST_MATERIALS["text_102"] * 400, page=1, start_offset=0)], None
    )
    _assert(est["calls"] == 0 and est["total_tokens"] == 0 and est["skip"] == "already", est)
    settings = SimpleNamespace(
        graph_extract_max_chunks=10,
        graph_extract_window_chars=6000,
        graph_extract_max_tokens=1600,
        graph_extract_min_chars=800,
    )
    fresh = SimpleNamespace(filename=_TEST_MATERIALS["text_034"], graph_extracted_at=None)
    chunks = [
        SimpleNamespace(
            id="1", text=_TEST_MATERIALS["text_098"] + (_TEST_MATERIALS["text_103"] * 80), page=1, start_offset=0
        ),
        SimpleNamespace(
            id="2", text=_TEST_MATERIALS["text_099"] + (_TEST_MATERIALS["text_104"] * 80), page=1, start_offset=10
        ),
    ]
    est2 = estimate_extract(fresh, chunks, settings)
    _assert(est2["calls"] >= 1, est2)
    _assert(est2["total_tokens"] >= est2["input_tokens"] + est2["output_tokens"], est2)
    print("estimate ok")


def test_heading_tree_document_order() -> None:
    headings = [
        HeadingHit(
            name=_TEST_MATERIALS["text_077"],
            level=1,
            quote=_TEST_MATERIALS["text_078"],
            chunk_id="c2",
            chunk_hash="h2",
            start_char=0,
            end_char=6,
        ),
        HeadingHit(
            name=_TEST_MATERIALS["text_079"],
            level=1,
            quote=_TEST_MATERIALS["text_080"],
            chunk_id="c1",
            chunk_hash="h1",
            start_char=0,
            end_char=6,
        ),
        HeadingHit(
            name=_TEST_MATERIALS["text_081"],
            level=1,
            quote=_TEST_MATERIALS["text_082"],
            chunk_id="c1",
            chunk_hash="h1",
            start_char=40,
            end_char=46,
        ),
    ]
    drafts = heading_tree_from_hits(
        headings,
        _TEST_MATERIALS["text_018"],
        chunk_rank={"c1": 0, "c2": 1},
    )
    sections = [item.name for item in drafts.entities if item.kind == "doc_section"]
    _assert(
        sections == [_TEST_MATERIALS["text_083"], _TEST_MATERIALS["text_084"], _TEST_MATERIALS["text_085"]], sections
    )
    print(_TEST_MATERIALS["text_019"])


def test_entity_sort_by_doc_pos() -> None:
    chunk_pos = {"c1": (1, 0), "c2": (2, 0)}

    class _Ent:
        def __init__(self, kind: str, name: str) -> None:
            self.kind = kind
            self.name = name

    class _Men:
        def __init__(self, chunk_id: str, start_char: int) -> None:
            self.chunk_id = chunk_id
            self.start_char = start_char
            self.edge_id = None

    entities = [
        _Ent("doc_section", _TEST_MATERIALS["text_035"]),
        _Ent("doc_section", _TEST_MATERIALS["text_036"]),
        _Ent("doc_section", _TEST_MATERIALS["text_037"]),
    ]
    mentions = {
        _TEST_MATERIALS["text_020"]: [_Men("c2", 0)],
        _TEST_MATERIALS["text_021"]: [_Men("c1", 0)],
        _TEST_MATERIALS["text_022"]: [_Men("c1", 120)],
    }
    ordered = sorted(
        entities,
        key=lambda item: entity_sort_key(item, mentions[item.name], chunk_pos),
    )
    names = [item.name for item in ordered]
    _assert(names == [_TEST_MATERIALS["text_086"], _TEST_MATERIALS["text_087"], _TEST_MATERIALS["text_088"]], names)
    print(_TEST_MATERIALS["text_023"])


def main() -> None:
    test_quote_mismatch_dropped()
    test_unknown_relation_rejected()
    test_json_repair()
    test_resolve_prefix_veto()
    test_no_key_note()
    test_dropped_quotes_counted()
    test_extract_token_budget_ignores_compose_cap()
    test_completion_text_uses_reasoning()
    test_skip_short_body()
    test_pack_windows_by_section()
    test_window_quote_aligns_to_any_chunk()
    test_already_extracted_skipped()
    test_extract_allows_plain_text()
    test_headings_must_be_in_text()
    test_window_entity_cap()
    test_heading_tree_document_order()
    test_entity_sort_by_doc_pos()
    test_estimate_skips_extracted()
    print("graph extract ok")


if __name__ == "__main__":
    main()
