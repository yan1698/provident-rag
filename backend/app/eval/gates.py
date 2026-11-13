"""Day 11: the report shape and the quality gates, as pure functions.

Split out of ``snapshot.py`` so CI can import and test the gate logic without
SQLAlchemy or a live Postgres. Everything that needs a DB session stays in
``snapshot.py``; this module only knows dataclasses and arithmetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.version import API_VERSION
from app.eval.golden import EvalGates, GoldenSet
from app.eval.metrics import EvalReport, QueryScore


@dataclass(frozen=True)
class CorpusFile:
    filename: str
    file_hash: str
    chunk_count: int
    expected: bool


@dataclass(frozen=True)
class CorpusSnapshot:
    path: str
    documents: list[CorpusFile]
    chunk_count: int
    off_corpus_count: int
    forbidden_documents: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    name: str
    op: str
    threshold: float
    actual: float | None
    passed: bool


@dataclass(frozen=True)
class TracesMeta:
    providers: list[str]
    llm_models: list[str]
    trace_ids: list[str]
    reverify: bool
    source: str


# 这些闸门算不出数值时静默跳过，而不是判失败：一次全是拒答题的回归不该因为
# 「没有 grounded 题可统计语料外证据」而红灯。质量闸门仍然保持 None 即失败。
_SKIP_WHEN_UNMEASURED = {"off_corpus_evidence_rate"}


def _as_float(value: int | float | None) -> float | None:
    return None if value is None else float(value)


def evaluate_gates(gates: EvalGates, report: EvalReport, corpus: CorpusSnapshot) -> list[GateResult]:
    checks: list[tuple[str, str, float | None, float | None]] = [
        ("avg_coverage_rate", ">=", gates.min_avg_coverage_rate, report.avg_coverage_rate),
        ("avg_hallucination_rate", "<=", gates.max_avg_hallucination_rate, report.avg_hallucination_rate),
        ("avg_recall_at_k", ">=", gates.min_avg_recall_at_k, report.avg_recall_at_k),
        ("mrr", ">=", gates.min_mrr, report.mrr),
        ("refusal_ok_rate", ">=", gates.min_refusal_ok_rate, report.refusal_ok_rate),
        ("term_hit_rate", ">=", gates.min_term_hit_rate, report.term_hit_rate),
        ("off_corpus_documents", "<=", _as_float(gates.max_off_corpus_documents), float(corpus.off_corpus_count)),
        ("off_corpus_evidence_rate", "<=", gates.max_off_corpus_evidence_rate, report.avg_off_corpus_evidence_rate),
        ("forbidden_documents", "<=", _as_float(gates.max_forbidden_documents), float(len(corpus.forbidden_documents))),
    ]
    results: list[GateResult] = []
    for name, op, threshold, actual in checks:
        if threshold is None:
            continue
        if actual is None and name in _SKIP_WHEN_UNMEASURED:
            continue
        if actual is None:
            passed = False
        elif op == ">=":
            passed = actual >= threshold
        else:
            passed = actual <= threshold
        results.append(GateResult(name=name, op=op, threshold=threshold, actual=actual, passed=passed))
    return results


def score_dict(item: QueryScore) -> dict[str, Any]:
    return asdict(item)


def metrics_dict(report: EvalReport) -> dict[str, Any]:
    payload = asdict(report)
    payload.pop("results", None)
    return payload


def freeze_report(
    report: EvalReport,
    *,
    golden: GoldenSet,
    corpus: CorpusSnapshot,
    traces: TracesMeta,
    api_version: str = API_VERSION,
) -> dict[str, Any]:
    gates = evaluate_gates(golden.gates, report, corpus)
    failures = [item.name for item in gates if not item.passed]
    return {
        "schema": "provident.eval.report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_version": api_version,
        "eval_set": golden.name,
        "corpus": asdict(corpus),
        "traces": asdict(traces),
        "metrics": metrics_dict(report),
        "gates": [asdict(item) for item in gates],
        "passed": not failures,
        "gate_failures": failures,
        "results": [score_dict(item) for item in report.results],
    }
