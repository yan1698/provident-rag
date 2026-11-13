"""Golden eval set, gold chunk labels, and trace-level metric aggregation.

Re-exports are lazy on purpose. ``app.eval.golden`` / ``app.eval.metrics`` are pure
functions with no DB dependency, and CI needs to import them on a machine that has
neither SQLAlchemy 2.x nor Postgres. An eager ``from app.eval.gold import ...`` here
would drag ``app.models`` into every one of those imports and take that away.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "EvalGates": "app.eval.golden",
    "GoldenQuery": "app.eval.golden",
    "GoldenSet": "app.eval.golden",
    "load_golden": "app.eval.golden",
    "list_golden_sets": "app.eval.golden",
    "resolve_golden_path": "app.eval.golden",
    "DEFAULT_EVAL_SET": "app.eval.golden",
    "EvalReport": "app.eval.metrics",
    "aggregate": "app.eval.metrics",
    "score_query": "app.eval.metrics",
    "GoldChunk": "app.eval.gold",
    "GoldResolution": "app.eval.gold",
    "resolve_gold": "app.eval.gold",
    "resolve_golden_set": "app.eval.gold",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
