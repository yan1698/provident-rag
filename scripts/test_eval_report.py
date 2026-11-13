"""Day 11 acceptance: frozen eval report, trace replay without LLM, Dify compare protocol."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
GOLDEN = ROOT / "data" / "eval" / "golden.json"
BASELINE = ROOT / "data" / "eval" / "baseline.json"


def _request(url: str, data: bytes | None = None, *, timeout: int = 120) -> dict:
    method = "GET" if data is None else "POST"
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} {url}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc


def _post_json(url: str, payload: dict) -> dict:
    return _request(url, data=json.dumps(payload).encode("utf-8"))


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts = []
    for piece in str(raw).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _sample_report(**overrides):
    from app.eval.metrics import EvalReport

    payload = dict(
        eval_set="unit",
        query_count=2,
        traced_count=2,
        missing=[],
        avg_coverage_rate=0.9,
        avg_hallucination_rate=0.1,
        term_hit_rate=1.0,
        citation_rate=1.0,
        refusal_ok_rate=1.0,
        gold_labeled_count=1,
        retrieval_scored_count=1,
        avg_recall_at_k=0.8,
        mrr=0.8,
        avg_off_corpus_evidence_rate=0.0,
        results=[],
    )
    payload.update(overrides)
    return EvalReport(**payload)


def _check_gates_and_snapshot() -> None:
    # Scenario-specific notes are maintained with local materials.
    from app.eval.gates import CorpusSnapshot, TracesMeta, evaluate_gates, freeze_report
    from app.eval.golden import EvalGates, GoldenQuery, GoldenSet, load_golden

    golden = load_golden(GOLDEN)
    by_name = load_golden(name="golden-agent-notes-v1")
    _assert(golden.name == "golden-agent-notes-v1", f"{_TEST_MATERIALS['text_001']}{golden.name}")
    _assert(by_name.name == golden.name, f"{_TEST_MATERIALS['text_002']}{by_name.name}")
    _assert(len(by_name.queries) == len(golden.queries), _TEST_MATERIALS["text_017"])
    _assert(golden.gates.min_avg_recall_at_k == 0.5, f"unexpected default gate: {golden.gates}")
    _assert(golden.forbidden_files, _TEST_MATERIALS["text_018"])
    from app.eval.golden import list_golden_sets

    _assert("golden-agent-notes-v1" in list_golden_sets(), f"versioned sets: {list_golden_sets()}")
    print(f"golden gates ok eval_set={golden.name} sets={list_golden_sets()}")

    corpus = CorpusSnapshot(path="data/raw/sample.md", documents=[], chunk_count=9, off_corpus_count=0)
    passing = evaluate_gates(golden.gates, _sample_report(), corpus)
    _assert(all(item.passed for item in passing), f"expected pass, got {passing}")

    failing = evaluate_gates(
        EvalGates(min_avg_recall_at_k=0.9, max_avg_hallucination_rate=0.05),
        _sample_report(avg_recall_at_k=0.2, avg_hallucination_rate=0.4),
        corpus,
    )
    names = {item.name for item in failing if not item.passed}
    _assert("avg_recall_at_k" in names and "avg_hallucination_rate" in names, f"{_TEST_MATERIALS['text_003']}{failing}")

    # Scenario-specific notes are maintained with local materials.
    grown = evaluate_gates(golden.gates, _sample_report(), CorpusSnapshot("x", [], 9, 12))
    _assert(all(item.passed for item in grown), f"{_TEST_MATERIALS['text_004']}{grown}")
    _assert(
        not any(item.name == "off_corpus_documents" for item in grown),
        f"{_TEST_MATERIALS['text_005']}{grown}",
    )

    # Scenario-specific notes are maintained with local materials.
    strict = evaluate_gates(EvalGates(max_off_corpus_documents=0), _sample_report(), CorpusSnapshot("x", [], 9, 2))
    _assert(
        any(item.name == "off_corpus_documents" and not item.passed for item in strict),
        f"{_TEST_MATERIALS['text_006']}{strict}",
    )

    # Scenario-specific notes are maintained with local materials.
    fixtures = evaluate_gates(
        golden.gates,
        _sample_report(),
        CorpusSnapshot("x", [], 9, 1, forbidden_documents=("generated.txt",)),
    )
    _assert(
        any(item.name == "forbidden_documents" and not item.passed for item in fixtures),
        f"{_TEST_MATERIALS['text_007']}{fixtures}",
    )

    # Scenario-specific notes are maintained with local materials.
    crowded = evaluate_gates(golden.gates, _sample_report(avg_off_corpus_evidence_rate=0.8), corpus)
    _assert(
        any(item.name == "off_corpus_evidence_rate" and not item.passed for item in crowded),
        f"{_TEST_MATERIALS['text_008']}{crowded}",
    )

    # Scenario-specific notes are maintained with local materials.
    unmeasured = evaluate_gates(golden.gates, _sample_report(avg_off_corpus_evidence_rate=None), corpus)
    _assert(
        not any(item.name == "off_corpus_evidence_rate" for item in unmeasured),
        f"{_TEST_MATERIALS['text_009']}{unmeasured}",
    )
    print("gates unit ok")

    tiny = GoldenSet(
        name="unit",
        corpus="data/raw/sample.md",
        queries=(GoldenQuery(id="q1", query=_TEST_MATERIALS["text_043"], expect="grounded", relevant_terms=()),),
        gates=EvalGates(min_avg_recall_at_k=0.5),
    )
    snap = freeze_report(
        _sample_report(avg_recall_at_k=0.2),
        golden=tiny,
        corpus=corpus,
        traces=TracesMeta(providers=["stub"], llm_models=["stub"], trace_ids=["t1"], reverify=False, source="stored"),
        api_version="0.11.0",
    )
    _assert(snap["schema"] == "provident.eval.report.v1", f"bad schema {snap.get('schema')}")
    _assert(snap["passed"] is False, _TEST_MATERIALS["text_019"])
    _assert("avg_recall_at_k" in snap["gate_failures"], f"failures {snap['gate_failures']}")
    _assert(snap["traces"]["source"] == "stored", snap["traces"])
    print("snapshot unit ok")


def _check_off_corpus_metric() -> None:
    from app.eval.golden import GoldenQuery, GoldenSet, forbidden_matches
    from app.eval.metrics import aggregate, off_corpus_evidence, score_query

    outside, identified = off_corpus_evidence(["aaa", "zzz", ""], {"aaa"})
    _assert((outside, identified) == (1, 2), f"{_TEST_MATERIALS['text_010']}{outside}/{identified}")

    spec = GoldenQuery(id="q1", query=_TEST_MATERIALS["text_030"], expect="grounded", relevant_terms=())
    evidence = [{"chunk_hash": "aaa"}, {"chunk_hash": "zzz"}, {"chunk_hash": ""}]
    scored = score_query(spec, trace_id="t1", evidence=evidence, corpus_hashes={"aaa"})
    _assert(scored.off_corpus_evidence_rate == 0.5, f"expected 1/2 off corpus: {scored}")

    legacy = score_query(spec, trace_id="t1", evidence=evidence)
    _assert(legacy.off_corpus_evidence_rate is None, f"{_TEST_MATERIALS['text_011']}{legacy}")

    refusal = GoldenQuery(id="q2", query=_TEST_MATERIALS["text_031"], expect="refusal", relevant_terms=())
    report = aggregate(
        GoldenSet(name="unit", corpus="", queries=(spec, refusal)),
        [
            scored,
            score_query(refusal, trace_id="t2", evidence=[{"chunk_hash": "zzz"}], corpus_hashes={"aaa"}),
        ],
    )
    _assert(
        report.avg_off_corpus_evidence_rate == 0.5,
        f"{_TEST_MATERIALS['text_012']}{report.avg_off_corpus_evidence_rate}",
    )

    _assert(forbidden_matches(["a.md", "generated.txt"], ("generated.*",)) == ("generated.txt",), "glob match broken")
    _assert(forbidden_matches(["Sample.MD"], ("sample.md",)) == ("Sample.MD",), _TEST_MATERIALS["text_020"])
    _assert(forbidden_matches(["sample.md"], ()) == (), _TEST_MATERIALS["text_021"])
    print(_TEST_MATERIALS["text_022"])


def _check_baseline_protocol() -> None:
    from app.eval.baseline import align_evidence, compare_metrics, parse_baseline, score_baseline
    from app.eval.gates import CorpusSnapshot
    from app.eval.gold import GoldChunk, GoldResolution, LabeledChunk
    from app.eval.golden import GoldenQuery, GoldenSet

    raw = json.loads(BASELINE.read_text(encoding="utf-8"))
    _assert(raw.get("system") == "dify", _TEST_MATERIALS["text_023"])
    _assert(raw.get("eval_set"), "baseline template missing eval_set")
    _assert(isinstance(raw.get("protocol"), list) and len(raw["protocol"]) >= 4, _TEST_MATERIALS["text_024"])
    parsed = parse_baseline(raw)
    _assert(parsed.system == "dify" and parsed.records == {}, parsed)
    print("baseline template ok")

    chunk = LabeledChunk(
        "c1",
        "abcd1234" * 8,
        "d1",
        1,
        (_TEST_MATERIALS["text_032"],),
        _TEST_MATERIALS["text_025"],
        text=_TEST_MATERIALS["text_033"],
    )
    aligned = align_evidence([{"text": _TEST_MATERIALS["text_037"]}], [chunk])
    _assert(aligned[0]["chunk_hash"] == chunk.chunk_hash, f"text align missed: {aligned}")
    hashed = align_evidence([{"chunk_hash": chunk.chunk_hash[:12], "text": _TEST_MATERIALS["text_038"]}], [chunk])
    _assert(hashed[0]["chunk_hash"] == chunk.chunk_hash, f"{_TEST_MATERIALS['text_013']}{hashed}")
    print("evidence align ok")

    spec = GoldenQuery(
        id="q_file_types",
        query=_TEST_MATERIALS["text_034"],
        expect="grounded",
        relevant_terms=("PDF", "TXT"),
        gold_sections=(_TEST_MATERIALS["text_039"],),
    )
    golden = GoldenSet(name="unit", corpus="", queries=(spec,))
    gold = {
        spec.id: GoldResolution(
            chunks=(GoldChunk("c1", chunk.chunk_hash, "d1", 1, spec.gold_sections[0], chunk.preview),),
            missing_labels=(),
        )
    }
    baseline = parse_baseline(
        {
            "system": "dify",
            "eval_set": "unit",
            "results": [
                {
                    "eval_id": "q_file_types",
                    "answer": _TEST_MATERIALS["text_040"],
                    "evidence": [{"index": 1, "text": _TEST_MATERIALS["text_044"]}],
                    "verification": {"coverage_rate": 1.0, "hallucination_rate": 0.0, "claims": []},
                }
            ],
        }
    )
    scored = score_baseline(golden, baseline, chunks=[chunk], gold_by_id=gold, reverify=False)
    _assert(scored.results[0].recall_at_k == 1.0, f"{_TEST_MATERIALS['text_014']}{scored.results[0]}")
    _assert(scored.results[0].reciprocal_rank == 1.0, scored.results[0])
    print("baseline scoring ok")

    ours = {
        "eval_set": "golden-sample-v1",
        "avg_coverage_rate": 0.9,
        "avg_hallucination_rate": 0.1,
        "avg_recall_at_k": 0.8,
        "mrr": 0.8,
        "term_hit_rate": 1.0,
        "citation_rate": 1.0,
        "refusal_ok_rate": 1.0,
    }
    weaker = {**ours, "avg_recall_at_k": 0.3, "avg_hallucination_rate": 0.4, "mrr": 0.3}
    from app.eval.gates import CorpusFile

    corpus = CorpusSnapshot(
        "data/raw/sample.md",
        documents=[CorpusFile("sample.md", "aaa", 9, True)],
        chunk_count=9,
        off_corpus_count=0,
    )
    matched = parse_baseline(
        {
            "system": "dify",
            "eval_set": "golden-sample-v1",
            "corpus_files": [{"filename": _TEST_MATERIALS["text_041"], "file_hash": "aaa"}],
            "metrics": weaker,
        }
    )
    diff = compare_metrics(ours, weaker, ours_corpus=corpus, baseline=matched)
    _assert(diff["winner"]["avg_recall_at_k"] == "ours", diff)
    _assert(diff["winner"]["avg_hallucination_rate"] == "ours", diff)
    _assert(diff["corpus_match"] is True, diff)
    _assert(diff["eval_set_match"] is True, diff)

    mismatched = parse_baseline(
        {
            "system": "dify",
            "eval_set": "other-set",
            "corpus_files": [{"filename": _TEST_MATERIALS["text_042"], "file_hash": "bbb"}],
            "metrics": weaker,
        }
    )
    bad = compare_metrics(ours, weaker, ours_corpus=corpus, baseline=mismatched)
    _assert(bad["corpus_match"] is False, bad)
    _assert(bad["eval_set_match"] is False, bad)
    _assert(any("file_hash" in note or "eval_set" in note for note in bad["notes"]), bad["notes"])
    print("compare unit ok")


def _check_live() -> None:
    health = _request(f"{BASE}/health")
    _assert(health.get("status") == "ok", f"health not ok: {health}")
    root = _request(f"{BASE}/")
    _assert("eval_report" in root and "eval_compare" in root, f"{_TEST_MATERIALS['text_015']}{root}")
    _assert(_version_tuple(root.get("version", "")) >= (0, 11), f"expected >= 0.11.0, got {root.get('version')}")
    print(f"api version={root.get('version')}")

    report = _request(f"{BASE}/eval/report")
    for key in (
        "schema",
        "generated_at",
        "api_version",
        "eval_set",
        "corpus",
        "traces",
        "metrics",
        "gates",
        "passed",
        "results",
    ):
        _assert(key in report, f"/eval/report missing {key}: {list(report)}")
    _assert(report["schema"] == "provident.eval.report.v1", report["schema"])
    _assert(report["metrics"]["traced_count"] >= 1, f"{_TEST_MATERIALS['text_016']}{report['metrics']}")
    _assert(report["corpus"]["chunk_count"] >= 1, f"empty corpus: {report['corpus']}")
    _assert(report["traces"]["source"] == "stored", report["traces"])
    _assert("avg_recall_at_k" in (report.get("metrics") or {}), report["metrics"])
    _assert("avg_off_corpus_evidence_rate" in (report.get("metrics") or {}), report["metrics"])
    _assert("forbidden_documents" in (report.get("corpus") or {}), report["corpus"])
    print(
        f"report ok passed={report['passed']} "
        f"recall={report['metrics']['avg_recall_at_k']} "
        f"mrr={report['metrics']['mrr']} "
        f"hallu={report['metrics']['avg_hallucination_rate']} "
        f"off_corpus_docs={report['corpus']['off_corpus_count']} "
        f"off_corpus_evidence={report['metrics']['avg_off_corpus_evidence_rate']}"
    )

    replay = _post_json(f"{BASE}/eval/replay", {"reverify": False})
    _assert(replay["traces"]["source"] == "stored", replay["traces"])
    _assert(replay["metrics"]["traced_count"] == report["metrics"]["traced_count"], _TEST_MATERIALS["text_026"])
    print(_TEST_MATERIALS["text_027"])

    expected = [
        {"filename": item["filename"], "file_hash": item["file_hash"]}
        for item in report["corpus"]["documents"]
        if item.get("expected")
    ]
    if not expected and report["corpus"]["documents"]:
        first = report["corpus"]["documents"][0]
        expected = [{"filename": first["filename"], "file_hash": first["file_hash"]}]
    weaker = {
        "system": "dify",
        "name": "dify-stub",
        "eval_set": report["eval_set"],
        "k": 6,
        "corpus_files": expected,
        "metrics": {
            "avg_coverage_rate": 0.4,
            "avg_hallucination_rate": 0.5,
            "avg_recall_at_k": 0.2,
            "mrr": 0.2,
            "term_hit_rate": 0.4,
            "citation_rate": 0.4,
            "refusal_ok_rate": 0.0,
        },
        "results": [],
    }
    compared = _post_json(f"{BASE}/eval/compare", weaker)
    _assert(compared["baseline"]["scored"] is False, _TEST_MATERIALS["text_028"])
    _assert(compared["comparison"]["corpus_match"] is True, compared["comparison"])
    _assert(compared["comparison"]["winner"]["avg_recall_at_k"] == "ours", compared["comparison"])
    _assert(compared["comparison"]["winner"]["avg_hallucination_rate"] == "ours", compared["comparison"])
    print(
        f"compare ok corpus_match={compared['comparison']['corpus_match']} "
        f"recall_winner={compared['comparison']['winner']['avg_recall_at_k']}"
    )


def main() -> None:
    _check_gates_and_snapshot()
    _check_off_corpus_metric()
    _check_baseline_protocol()
    if os.environ.get("SKIP_EVAL_LLM") == "1":
        print(_TEST_MATERIALS["text_035"])
        print(_TEST_MATERIALS["text_036"])
        return
    _check_live()
    print(_TEST_MATERIALS["text_029"])


if __name__ == "__main__":
    main()
