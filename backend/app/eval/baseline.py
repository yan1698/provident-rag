"""Day 11: score a foreign system (Dify 等) with the same judge as Provident.

对照口径（必须同时满足，否则对比没有测量意义）：

1. 同一批源文档：baseline.corpus_files[].file_hash 必须对得上当前 documents.file_hash。
2. 同一批问题：eval_id 对齐 golden.json，禁止各写各的 query 文本。
3. 同一个 k：baseline.k 应等于当时 /query 的 top_k（默认写在报告 traces 里）。
4. 同一套 gold chunk：Recall@k / MRR 用 golden.json 的 gold_sections / gold_chunk_hashes。
5. 同一个 Verifier：用我们的 Cross-Encoder + 词面兜底给 baseline 的 answer+evidence 打
   coverage / hallucination。Dify 自己的「引用」格式不参与计分，避免两套法官。

Dify 通常给不出我们的 chunk_hash。对齐规则：

- 证据自带 chunk_hash（或前缀）→ 直接用
- 否则在语料 chunk 全文里找：证据文本是 chunk 的子串，或 chunk 是证据的子串
- 再否则按空白折叠后的字符重合率，≥ 0.5 取最高者
- 对不上就留空 hash，这条证据对 Recall 计 0，不假装命中
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.eval.gates import CorpusSnapshot, metrics_dict
from app.eval.gold import LabeledChunk
from app.eval.golden import GoldenSet, eval_data_file
from app.eval.metrics import EvalReport, QueryScore, aggregate, score_query

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.core.config import Settings

# 对齐与对比是纯函数，只有 score_baseline_from_db 和重新验证需要 DB / 本地模型，
# 那几个 import 因此放进函数体，让协议测试能在无 DB 的 CI 里跑。

_MIN_OVERLAP = 0.5
DEFAULT_BASELINE_PATH = eval_data_file("baseline.json")

_HIGHER_BETTER = {
    "avg_coverage_rate",
    "term_hit_rate",
    "citation_rate",
    "refusal_ok_rate",
    "avg_recall_at_k",
    "mrr",
}
_LOWER_BETTER = {"avg_hallucination_rate"}


@dataclass
class BaselineRecord:
    eval_id: str
    query: str = ""
    answer: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] | None = None
    retrieved_chunk_hashes: list[str] = field(default_factory=list)


@dataclass
class BaselineSet:
    system: str
    name: str
    eval_set: str
    k: int | None
    corpus_files: list[dict[str, str]]
    metrics: dict[str, Any]
    records: dict[str, BaselineRecord]


def load_baseline(path: Path | None = None) -> BaselineSet:
    target = path or eval_data_file("baseline.json")
    if not target.is_file():
        raise FileNotFoundError(f"baseline file not found: {target}")
    return parse_baseline(json.loads(target.read_text(encoding="utf-8")))


def parse_baseline(raw: dict[str, Any]) -> BaselineSet:
    records: dict[str, BaselineRecord] = {}
    for item in raw.get("results") or []:
        eval_id = str(item.get("eval_id") or "").strip()
        if not eval_id:
            continue
        hashes = [str(h) for h in (item.get("retrieved_chunk_hashes") or []) if str(h).strip()]
        records[eval_id] = BaselineRecord(
            eval_id=eval_id,
            query=str(item.get("query") or ""),
            answer=str(item.get("answer") or ""),
            evidence=list(item.get("evidence") or []),
            citations=list(item.get("citations") or []),
            verification=item.get("verification") if isinstance(item.get("verification"), dict) else None,
            retrieved_chunk_hashes=hashes,
        )
    return BaselineSet(
        system=str(raw.get("system") or "baseline"),
        name=str(raw.get("name") or raw.get("system") or "baseline"),
        eval_set=str(raw.get("eval_set") or ""),
        k=raw.get("k"),
        corpus_files=[
            {"filename": str(item.get("filename") or ""), "file_hash": str(item.get("file_hash") or "")}
            for item in (raw.get("corpus_files") or [])
        ],
        metrics=dict(raw.get("metrics") or {}),
        records=records,
    )


def align_evidence(evidence: list[dict[str, Any]], chunks: list[LabeledChunk]) -> list[dict[str, Any]]:
    """Fill chunk_hash on each evidence row so Recall@k 能对上 gold 集合。"""
    by_hash = {item.chunk_hash: item for item in chunks if item.chunk_hash}
    aligned: list[dict[str, Any]] = []
    for position, raw in enumerate(evidence, start=1):
        row = dict(raw)
        if not row.get("index"):
            row["index"] = position
        existing = str(row.get("chunk_hash") or "").strip().lower()
        if existing:
            match = _hash_lookup(existing, by_hash)
            if match:
                row["chunk_hash"] = match.chunk_hash
                row.setdefault("chunk_id", match.chunk_id)
                aligned.append(row)
                continue
        text = str(row.get("text") or "")
        match = _text_lookup(text, chunks)
        if match:
            row["chunk_hash"] = match.chunk_hash
            row.setdefault("chunk_id", match.chunk_id)
            row.setdefault("page", match.page)
        aligned.append(row)
    return aligned


def score_baseline(
    golden: GoldenSet,
    baseline: BaselineSet,
    *,
    chunks: list[LabeledChunk],
    gold_by_id: dict[str, Any] | None = None,
    reverify: bool = False,
    settings: Settings | None = None,
) -> EvalReport:
    """Score baseline records with Provident's gold labels + Verifier. No generator LLM."""
    gold_by_id = gold_by_id or {}
    corpus_hashes = frozenset(item.chunk_hash for item in chunks if item.chunk_hash)
    scores: list[QueryScore] = []
    for spec in golden.queries:
        record = baseline.records.get(spec.id)
        resolution = gold_by_id.get(spec.id)
        gold_hashes = set(resolution.hashes) if resolution is not None else set()
        missing = list(resolution.missing_labels) if resolution is not None else []
        if record is None:
            scores.append(score_query(spec, trace_id=None, gold_hashes=gold_hashes, gold_missing_labels=missing))
            continue
        evidence = align_evidence(_with_retrieved_hashes(record), chunks)
        verification = record.verification
        if reverify or verification is None:
            from app.core.config import get_settings
            from app.eval.replay import to_evidence_items
            from app.verify.service import verify_answer

            settings = settings or get_settings()
            result = verify_answer(record.answer, to_evidence_items(evidence), settings=settings)
            verification = asdict(result)
        scores.append(
            score_query(
                spec,
                trace_id=f"{baseline.system}:{spec.id}",
                answer=record.answer,
                evidence=evidence,
                citations=record.citations,
                verification=verification,
                gold_hashes=gold_hashes,
                gold_missing_labels=missing,
                corpus_hashes=corpus_hashes,
            )
        )
    return aggregate(golden, scores)


def score_baseline_from_db(
    db: Session,
    golden: GoldenSet,
    baseline: BaselineSet,
    *,
    reverify: bool = False,
    settings: Settings | None = None,
    owner_id=None,
) -> EvalReport:
    from app.eval.gold import corpus_chunks, resolve_golden_set

    chunks = corpus_chunks(db, golden.corpus, owner_id=owner_id)
    gold = resolve_golden_set(db, golden, owner_id=owner_id)
    return score_baseline(
        golden,
        baseline,
        chunks=chunks,
        gold_by_id=gold,
        reverify=reverify,
        settings=settings,
    )


def compare_metrics(
    ours: dict[str, Any],
    theirs: dict[str, Any],
    *,
    ours_corpus: CorpusSnapshot | None = None,
    baseline: BaselineSet | None = None,
) -> dict[str, Any]:
    keys = (
        "avg_coverage_rate",
        "avg_hallucination_rate",
        "avg_recall_at_k",
        "mrr",
        "term_hit_rate",
        "citation_rate",
        "refusal_ok_rate",
    )
    delta: dict[str, float | None] = {}
    winner: dict[str, str] = {}
    for key in keys:
        left = ours.get(key)
        right = theirs.get(key)
        if left is None or right is None:
            delta[key] = None
            winner[key] = "n/a"
            continue
        delta[key] = round(float(left) - float(right), 6)
        if abs(delta[key]) < 1e-9:
            winner[key] = "tie"
        elif key in _LOWER_BETTER:
            winner[key] = "ours" if left < right else "baseline"
        else:
            winner[key] = "ours" if left > right else "baseline"

    notes = _protocol_notes(ours_corpus, baseline)
    eval_set_match = True
    if baseline and baseline.eval_set and ours.get("eval_set"):
        eval_set_match = baseline.eval_set == ours.get("eval_set")
        if not eval_set_match:
            notes.append(f"eval_set mismatch: ours={ours.get('eval_set')} baseline={baseline.eval_set}")

    return {
        "ours": {key: ours.get(key) for key in keys},
        "baseline": {key: theirs.get(key) for key in keys},
        "delta": delta,
        "winner": winner,
        "eval_set_match": eval_set_match,
        "corpus_match": not any(item.startswith("corpus") for item in notes),
        "notes": notes,
    }


def precomputed_metrics(baseline: BaselineSet) -> dict[str, Any]:
    """Use numbers the baseline file already carried, if any."""
    return dict(baseline.metrics)


def report_metrics(report: EvalReport) -> dict[str, Any]:
    payload = metrics_dict(report)
    payload["eval_set"] = report.eval_set
    return payload


def _with_retrieved_hashes(record: BaselineRecord) -> list[dict[str, Any]]:
    if record.evidence:
        return list(record.evidence)
    return [
        {"chunk_hash": item, "text": "", "index": i} for i, item in enumerate(record.retrieved_chunk_hashes, start=1)
    ]


def _hash_lookup(prefix: str, by_hash: dict[str, LabeledChunk]) -> LabeledChunk | None:
    if prefix in by_hash:
        return by_hash[prefix]
    if len(prefix) < 6:
        return None
    hits = [item for item in by_hash.values() if item.chunk_hash.lower().startswith(prefix)]
    return hits[0] if len(hits) == 1 else None


def _text_lookup(text: str, chunks: list[LabeledChunk]) -> LabeledChunk | None:
    needle = _fold(text)
    if len(needle) < 12:
        return None
    contained = [item for item in chunks if needle in _fold(_chunk_text(item)) or _fold(_chunk_text(item)) in needle]
    if len(contained) == 1:
        return contained[0]
    if contained:
        return max(contained, key=lambda item: _overlap(needle, _fold(_chunk_text(item))))

    ranked: list[tuple[float, LabeledChunk]] = []
    for item in chunks:
        ratio = _overlap(needle, _fold(_chunk_text(item)))
        if ratio >= _MIN_OVERLAP:
            ranked.append((ratio, item))
    if not ranked:
        return None
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked[0][1]


def _chunk_text(chunk: LabeledChunk) -> str:
    return chunk.text or chunk.preview


def _fold(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _overlap(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if shorter in longer:
        return len(shorter) / len(longer)
    hits = 0
    window = max(4, min(12, len(shorter) // 8 or 4))
    seen: set[str] = set()
    for i in range(0, len(shorter) - window + 1, window):
        piece = shorter[i : i + window]
        if piece in seen:
            continue
        seen.add(piece)
        if piece in longer:
            hits += 1
    return hits / max(len(seen), 1)


def _protocol_notes(corpus: CorpusSnapshot | None, baseline: BaselineSet | None) -> list[str]:
    notes: list[str] = []
    if corpus is None or baseline is None:
        return notes
    if not baseline.corpus_files:
        notes.append("corpus: baseline 未声明 file_hash，无法证明吃到同一批源文档")
        return notes
    ours = {item.filename: item.file_hash for item in corpus.documents if item.expected}
    theirs = {item["filename"]: item["file_hash"] for item in baseline.corpus_files if item.get("filename")}
    for name, digest in ours.items():
        other = theirs.get(name)
        if not other:
            notes.append(f"corpus: baseline 缺少 {name}")
        elif other != digest:
            notes.append(f"corpus: {name} file_hash 不一致")
    return notes
