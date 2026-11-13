from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, fields
from fnmatch import fnmatch
from pathlib import Path

_EVAL_DIRNAME = Path("data") / "eval"
DEFAULT_EVAL_SET = "golden-agent-notes-v1"


def eval_roots() -> tuple[Path, ...]:
    """Host ``data/eval`` directories plus the Docker ``/data/eval`` mount, first match wins."""
    found: list[Path] = []
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _EVAL_DIRNAME
        if candidate.is_dir() and candidate not in found:
            found.append(candidate)
    mounted = Path("/data/eval")
    if mounted.is_dir() and mounted not in found:
        found.append(mounted)
    if found:
        return tuple(found)
    if Path("/.dockerenv").exists():
        return (mounted,)
    return (here.parents[3] / _EVAL_DIRNAME,)


def eval_data_file(name: str) -> Path:
    """Locate ``data/eval/<name>`` on the host repo or the Docker ``/data/eval`` mount."""
    rel = Path(name)
    for root in eval_roots():
        candidate = root / rel
        if candidate.is_file():
            return candidate
    roots = eval_roots()
    if Path("/.dockerenv").exists():
        return Path("/data/eval") / rel
    return (roots[0] if roots else Path("data") / "eval") / rel


def _set_stem(name: str) -> str:
    text = str(name or "").strip()
    if text.endswith(".json"):
        return Path(text).stem
    return Path(text).name or DEFAULT_EVAL_SET


def resolve_golden_path(*, name: str | None = None, path: Path | None = None) -> Path:
    """Resolve a versioned set ``data/eval/golden/<name>.json``, then the legacy pointer."""
    if path is not None:
        return Path(path)
    set_name = _set_stem(name or DEFAULT_EVAL_SET)
    versioned = eval_data_file(f"golden/{set_name}.json")
    if versioned.is_file():
        return versioned
    return eval_data_file("golden.json")


def list_golden_sets() -> tuple[str, ...]:
    names: list[str] = []
    for root in eval_roots():
        folder = root / "golden"
        if not folder.is_dir():
            continue
        for item in sorted(folder.glob("*.json")):
            if item.stem not in names:
                names.append(item.stem)
    return tuple(names)


DEFAULT_GOLDEN_PATH = resolve_golden_path()


@dataclass(frozen=True)
class EvalGates:
    """Day 11 quality gates. None means the gate is skipped."""

    min_avg_coverage_rate: float | None = 0.7
    max_avg_hallucination_rate: float | None = 0.25
    min_avg_recall_at_k: float | None = 0.5
    min_mrr: float | None = 0.5
    min_refusal_ok_rate: float | None = 1.0
    min_term_hit_rate: float | None = 0.8
    # 知识库里存在 corpus 之外的文档是常态，光看份数会随语料正常增长误杀，默认不拦。
    # 真正让指标失真的是「语料外的 chunk 挤进 Top-K」和「测试夹具入库」，由下面两项负责。
    max_off_corpus_documents: int | None = None
    max_off_corpus_evidence_rate: float | None = 0.5
    max_forbidden_documents: int | None = 0


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    expect: str
    relevant_terms: tuple[str, ...]
    gold_sections: tuple[str, ...] = ()
    gold_chunk_hashes: tuple[str, ...] = ()

    @property
    def has_gold(self) -> bool:
        return bool(self.gold_sections or self.gold_chunk_hashes)


@dataclass(frozen=True)
class GoldenSet:
    name: str
    corpus: str
    queries: tuple[GoldenQuery, ...]
    gates: EvalGates = EvalGates()
    forbidden_files: tuple[str, ...] = ()


def _str_tuple(values: object) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(v).strip() for v in values if str(v).strip())  # type: ignore[union-attr]


def _read_raw(target: Path, *, seen: set[Path] | None = None) -> tuple[Path, dict]:
    walked = seen if seen is not None else set()
    resolved = target.resolve() if target.exists() else target
    if resolved in walked:
        raise ValueError(f"golden alias loop at {target}")
    walked.add(resolved)
    if not target.is_file():
        raise FileNotFoundError(f"golden eval set not found: {target}")
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"golden file is not an object: {target}")
    if raw.get("queries"):
        return target, raw
    nxt = raw.get("moved_to") or raw.get("alias_of") or raw.get("name")
    if not nxt:
        raise ValueError(f"golden eval set has no queries: {target}")
    hint = str(nxt).strip()
    if "/" in hint or "\\" in hint or hint.endswith(".json"):
        rel = Path(hint)
        candidate = rel if rel.is_absolute() else target.parent / rel
        if not candidate.is_file():
            candidate = resolve_golden_path(name=rel.stem)
    else:
        candidate = resolve_golden_path(name=hint)
    return _read_raw(candidate, seen=walked)


def load_golden(path: Path | None = None, name: str | None = None) -> GoldenSet:
    target = resolve_golden_path(name=name, path=path)
    target, raw = _read_raw(target)
    queries = []
    seen: set[str] = set()
    for item in raw.get("queries") or []:
        qid = str(item["id"]).strip()
        if not qid or qid in seen:
            raise ValueError(f"duplicate or empty golden query id: {qid!r}")
        seen.add(qid)
        expect = str(item.get("expect") or "grounded").strip().lower()
        if expect not in {"grounded", "refusal"}:
            raise ValueError(f"query {qid}: expect must be grounded or refusal")
        queries.append(
            GoldenQuery(
                id=qid,
                query=str(item["query"]).strip(),
                expect=expect,
                relevant_terms=_str_tuple(item.get("relevant_terms")),
                gold_sections=_str_tuple(item.get("gold_sections")),
                gold_chunk_hashes=_str_tuple(item.get("gold_chunk_hashes")),
            )
        )
    if not queries:
        raise ValueError("golden eval set has no queries")
    set_name = str(raw.get("name") or target.stem or "golden")
    if target.parent.name == "golden" and target.stem != set_name:
        raise ValueError(f"golden file {target.name} must be named {set_name}.json")
    return GoldenSet(
        name=set_name,
        corpus=_corpus_field(raw.get("corpus")),
        queries=tuple(queries),
        gates=_load_gates(raw.get("gates")),
        forbidden_files=_str_tuple(raw.get("forbidden_files")),
    )


def corpus_filenames(corpus: str) -> tuple[str, ...]:
    """``golden.corpus`` 可以是单路径，也可以是逗号分隔 / JSON 数组展开后的多个文件名。"""
    names: list[str] = []
    for part in (corpus or "").split(","):
        name = Path(part.strip()).name
        if name:
            names.append(name)
    return tuple(names)


def forbidden_matches(filenames: Iterable[str], patterns: tuple[str, ...]) -> tuple[str, ...]:
    """挑出命中夹具黑名单的文件名。``patterns`` 支持 glob，大小写不敏感。"""
    if not patterns:
        return ()
    lowered = [item.lower() for item in patterns]
    hits = [name for name in filenames if any(fnmatch(name.lower(), pattern) for pattern in lowered)]
    return tuple(dict.fromkeys(hits))


def _corpus_field(raw: object) -> str:
    if isinstance(raw, list):
        names = [Path(str(item).strip()).name for item in raw if str(item).strip()]
        return ", ".join(names)
    return str(raw or "").strip()


def _load_gates(raw: object) -> EvalGates:
    if not isinstance(raw, dict):
        return EvalGates()
    allowed = {item.name for item in fields(EvalGates)}
    kwargs = {key: raw[key] for key in raw if key in allowed}
    return EvalGates(**kwargs)
