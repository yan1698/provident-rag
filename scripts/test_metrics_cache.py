"""Day 14 acceptance: /metrics + Redis search cache short-circuit."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lib.accept import assert_ok, version_tuple
from lib.http import API_BASE, get_json, post_json


def main() -> None:
    root = get_json(f"{API_BASE}/")
    assert_ok(version_tuple(root.get("version", "0")) >= (0, 14), f"need API 0.14+: {root.get('version')}")
    assert_ok(root.get("metrics") == "/metrics", f"root missing metrics: {root}")

    stats = get_json(f"{API_BASE}/metrics")
    assert_ok("cache" in stats and "documents" in stats, f"metrics shape: {stats}")
    assert_ok(stats.get("api_version"), "metrics missing api_version")
    assert_ok("trace_quality" in stats, f"metrics missing trace_quality: {stats}")
    assert_ok("hits" in stats["cache"] and "misses" in stats["cache"], f"cache stats: {stats['cache']}")
    print(f"metrics docs={stats['documents']} chunks={stats['chunks']} traces={stats['traces']} cache={stats['cache']}")

    query = f"{_TEST_MATERIALS['text_001']}{time.time_ns()}"
    first = post_json(f"{API_BASE}/search/dense", {"query": query, "top_k": 3}, timeout=180)
    assert_ok(first.get("cached") is False, f"{_TEST_MATERIALS['text_002']}{first.get('cached')}")
    second = post_json(f"{API_BASE}/search/dense", {"query": query, "top_k": 3}, timeout=180)
    assert_ok(second.get("cached") is True, f"{_TEST_MATERIALS['text_003']}{second}")
    assert_ok(second.get("count") == first.get("count"), "cached payload drifted")

    after = get_json(f"{API_BASE}/metrics")
    assert_ok(after["cache"]["hits"] >= 1, f"{_TEST_MATERIALS['text_004']}{after['cache']}")
    print("search cache ok")

    index = ROOT / "web" / "index.html"
    assert_ok(index.is_file(), "web/index.html missing — /ui cannot be served")
    print("ship files ok")


if __name__ == "__main__":
    main()
