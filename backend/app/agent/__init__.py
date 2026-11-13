"""Agent package."""

from __future__ import annotations

__all__ = ["classify_task_kind", "run_agent"]


def __getattr__(name: str):
    if name == "classify_task_kind":
        from app.agent.classify import classify_task_kind

        return classify_task_kind
    if name == "run_agent":
        from app.agent.service import run_agent

        return run_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
