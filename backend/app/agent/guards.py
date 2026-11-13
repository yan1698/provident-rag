"""Host guards G1–G6. Rewrite the plan before the tool runs."""

from __future__ import annotations

from typing import Any

from app.agent.classify import LONGFORM_KINDS
from app.agent.planner import Plan
from app.agent.state import AgentState

# 只在没传入 registry 时当兜底。新动作应注册进 Tool.task_kinds，不要往这里加名字。
_FALLBACK_ACTIONS = frozenset(
    {
        "analyze",
        "survey",
        "decompose",
        "digest",
        "read",
        "graph_walk",
        "retrieve",
        "compose",
        "verify",
        "finish",
        "refuse",
        "answer",
        "error",
    }
)


def _analyzed(state: AgentState, history: list[dict[str, Any]]) -> bool:
    return state.task_spec is not None or any(item.get("action") == "analyze" for item in history)


def _surveyed(state: AgentState, history: list[dict[str, Any]]) -> bool:
    return bool(state.catalog) or any(item.get("action") == "survey" for item in history)


def _composed(history: list[dict[str, Any]]) -> bool:
    return any(item.get("action") == "compose" for item in history)


def _has_retrieved_node(state: AgentState) -> bool:
    return any(node.status == "retrieved" for node in state.nodes) or any(
        isinstance(item, dict) and item.get("bullets") for item in state.digests
    )


def apply_guards(
    plan: Plan,
    state: AgentState,
    history: list[dict[str, Any]],
    *,
    registry: Any | None = None,
) -> Plan:
    """Return plan, or a rewritten plan with source=guard."""
    action = (plan.action or "").strip().lower()
    known = _FALLBACK_ACTIONS
    if registry is not None:
        known = frozenset(registry.names()) | {"error"}
    if action not in known:
        return Plan(f"G6: 未知动作 {action}。", "error", action, "guard")

    if state.task_kind in LONGFORM_KINDS:
        return _synthesis_guards(plan, state, history, action)
    return _factoid_guards(plan, state, history, action)


def _synthesis_guards(
    plan: Plan,
    state: AgentState,
    history: list[dict[str, Any]],
    action: str,
) -> Plan:
    if action != "analyze" and not _analyzed(state, history):
        if action in {
            "survey",
            "decompose",
            "digest",
            "read",
            "retrieve",
            "compose",
            "finish",
            "answer",
            "verify",
            "graph_walk",
        }:
            return Plan("G1: 综合任务必须先理解任务。", "analyze", "", "guard")

    if state.skips_retrieve() and action in {"survey", "decompose", "digest", "read", "retrieve", "graph_walk"}:
        return Plan("G1: 复述或澄清不扫描目录、不检索。", "compose", "", "guard")

    if state.catalog_inventory() and action in {"decompose", "digest", "retrieve", "graph_walk"}:
        if not _surveyed(state, history):
            return Plan("G1: 整库盘点必须先扫描目录。", "survey", "", "guard")
        return Plan("G1: 整库盘点按目录列出，不按主题检索。", "compose", "", "guard")

    if action in {"retrieve", "digest", "read", "decompose", "graph_walk"} and not _surveyed(state, history):
        if action == "graph_walk":
            return Plan("G7: 未扫描目录不得沿图谱走关系。", "survey", "", "guard")
        return Plan("G1: 综合任务必须先扫描知识库目录。", "survey", "", "guard")

    if (
        action in {"compose", "finish", "answer", "verify"}
        and not state.skips_retrieve()
        and not _surveyed(state, history)
    ):
        return Plan("G1: 综合任务必须先扫描知识库目录。", "survey", "", "guard")

    if action in {"compose", "finish"} and not _has_retrieved_node(state):
        if state.skips_retrieve() and (state.prior_answer or state.memory_brief or state.turn_kind == "offtopic"):
            return plan
        if state.catalog_inventory() and state.catalog:
            return plan
        return Plan("G2: 没有检索到可用节点，拒绝编造。", "refuse", "", "guard")

    composed = _composed(history)
    over_retrieves = state.subtopic_retrieves >= max(1, state.max_retrieves)
    last_step = state.total_steps >= max(1, state.max_steps) - 1
    if (
        (over_retrieves or last_step)
        and not composed
        and action
        not in {"compose", "refuse", "finish", "survey", "decompose", "digest", "verify", "analyze", "graph_walk"}
    ):
        if _has_retrieved_node(state):
            return Plan("G5: 预算用尽，改为基于已有证据起草。", "compose", "", "guard")
        return Plan("G5: 预算用尽且没有检索到节点。", "refuse", "", "guard")

    return plan


def _factoid_guards(
    plan: Plan,
    state: AgentState,
    history: list[dict[str, Any]],
    action: str,
) -> Plan:
    if (state.reference or {}).get("action") == "ask" and action in {"compose", "finish"}:
        return plan
    retrieved = state.retrieve_attempts > 0 or any(item.get("action") == "retrieve" for item in history)
    if action == "finish" and not retrieved:
        return Plan("G4: 尚未检索，先查语料。", "retrieve", state.goal, "guard")
    return plan
