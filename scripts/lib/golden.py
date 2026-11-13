"""Read versioned golden sets without importing backend."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_EVAL_SET = "golden-agent-notes-v1"


def resolve_golden_path(root: Path, name: str | None = None, path: Path | None = None) -> Path:
    """Prefer ``data/eval/golden/<name>.json``; fall back to the legacy pointer file."""
    if path is not None:
        return Path(path)
    set_name = Path(str(name or DEFAULT_EVAL_SET).strip()).stem
    versioned = root / "data" / "eval" / "golden" / f"{set_name}.json"
    if versioned.is_file():
        return versioned
    return root / "data" / "eval" / "golden.json"


def _follow(target: Path, *, seen: set[Path] | None = None) -> tuple[Path, dict]:
    walked = seen if seen is not None else set()
    resolved = target.resolve() if target.exists() else target
    if resolved in walked:
        raise SystemExit(f"golden alias loop at {target}")
    walked.add(resolved)
    if not target.is_file():
        raise SystemExit(f"golden file not found: {target}")
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"golden file is not an object: {target}")
    if raw.get("queries"):
        return target, raw
    nxt = raw.get("moved_to") or raw.get("alias_of") or raw.get("name")
    if not nxt:
        raise SystemExit(f"no queries in {target}")
    hint = str(nxt).strip()
    if "/" in hint or "\\" in hint or hint.endswith(".json"):
        rel = Path(hint)
        candidate = rel if rel.is_absolute() else target.parent / rel
        if not candidate.is_file():
            candidate = resolve_golden_path(
                target.parents[2] if target.parent.name == "eval" else target.parents[3], rel.stem
            )
    else:
        root = target.parents[2] if target.parent.name == "eval" else target.parents[3]
        candidate = resolve_golden_path(root, hint)
    return _follow(candidate, seen=walked)


def load_golden_json(
    path: Path | None = None, *, root: Path | None = None, name: str | None = None
) -> tuple[str, list[dict]]:
    """Return (eval_set_name, queries). Each query is a dict with at least ``id``."""
    if path is None:
        if root is None:
            raise SystemExit("load_golden_json needs path or root")
        path = resolve_golden_path(root, name)
    target, raw = _follow(Path(path))
    set_name = str(raw.get("name") or "eval")
    queries = [item for item in (raw.get("queries") or []) if isinstance(item, dict) and item.get("id")]
    if not queries:
        raise SystemExit(f"no queries in {target}")
    return set_name, queries
