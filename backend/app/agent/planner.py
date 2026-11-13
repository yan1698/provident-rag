"""Planner: rules + optional LLM tool-calling over the local corpus."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agent.classify import LONGFORM_KINDS, TASK_KIND_DIGEST, TASK_KIND_SYNTHESIS

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.agent.state import AgentState

ACTIONS = ("retrieve", "answer", "verify", "finish", "refuse")

PLANNER_SYSTEM = (
    "你是 Provident 的检索规划器。只根据本地语料作答，禁止编造。"
    "每次只输出一个 JSON 对象，不要 Markdown，不要解释。"
    '格式：{"thought":"...","action":"retrieve|answer|verify|finish|refuse","input":"..."}。'
    "retrieve：input 是检索 query，可多次用不同子问题检索。"
    "answer：基于已检索证据生成带 [S<n>] 引用的回答，input 可为空。"
    "verify：对上一轮回答做蕴含校验。"
    "finish：结束，忽略 input，沿用 answer/compose 已写出的回答。"
    "refuse：证据不足，必须拒绝而不是猜测。"
    "没有检索过就不要 finish。verify 后若 coverage_rate 偏低应再 retrieve 或 refuse。"
    "不要调用 web_search。"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class Plan:
    thought: str
    action: str
    input: str
    source: str  # rules | llm | guard
    # 动作专属参数。以前这里是一个 retrieve 专用的 node_id 字段，结果每加一个需要
    # 额外入参的工具都要动 Plan 和执行器；现在一律走这里。
    args: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        thought: str,
        action: str,
        input: str = "",
        source: str = "rules",
        args: Mapping[str, Any] | None = None,
        node_id: str = "",
    ) -> None:
        object.__setattr__(self, "thought", thought)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "source", source)
        payload = dict(args or {})
        if node_id:
            payload.setdefault("node_id", node_id)
        object.__setattr__(self, "args", payload)

    @property
    def node_id(self) -> str:
        return str(self.args.get("node_id") or "")


def choose_planner(settings: Settings, override: str | None = None) -> str:
    mode = (override or settings.agent_planner or "auto").strip().lower()
    if mode in {"rules", "llm"}:
        return mode
    # auto: planning is deterministic; the generator LLM is only used in `answer`.
    return "rules"


def _refuse_live_gap(state: AgentState | None) -> Plan | None:
    """Skip retrieve only for a live (not stale) gap whose key matches this turn's query."""
    if state is None or not state.gap_hints:
        return None
    from app.memory.promote import should_skip_retrieve_for_gap
    from app.memory.query import build_retrieve_query

    query = build_retrieve_query(state.retrieve_slots) if state.retrieve_slots else (state.goal or "").strip()
    surface = str((state.retrieve_slots or {}).get("goal_surface") or "")
    for item in state.gap_hints:
        key = str(item.get("key") or "")
        if key not in {query, surface}:
            continue
        if should_skip_retrieve_for_gap(item, digest_unchanged=True):
            return Plan("同一空检索且语料未变，跳过检索。", "refuse", "", "rules")
    return None


def next_plan(
    goal: str,
    history: list[dict[str, Any]],
    settings: Settings,
    *,
    planner: str | None = None,
    state: AgentState | None = None,
) -> Plan:
    from app.agent.partial import only_unverified_gaps

    if history and history[-1].get("action") == "verify" and only_unverified_gaps(history[-1].get("observation")):
        return Plan("保留未核实提示并交付，不因支持状态未知自动重新生成。", "finish", "", "rules")
    if (
        state is not None
        and state.reference
        and (
            state.reference.get("action") == "ask"
            or state.reference.get("kind") == "step"
            and state.reference.get("retrieve")
        )
    ):
        return rule_plan(state.goal, history, settings=settings, state=state)
    is_longform = state is not None and state.task_kind in LONGFORM_KINDS
    if not is_longform:
        from app.agent.analyze import split_factoid_queries

        max_retrieves = max(1, settings.agent_max_retrieves, len(split_factoid_queries(goal)))
        retrieve_count = sum(1 for item in history if item.get("action") == "retrieve")
        answered = any(item.get("action") == "answer" for item in history)
        last = history[-1] if history else {}
        # Safety for LLM planner that keeps retrieving.
        if retrieve_count >= max_retrieves and not answered and last.get("action") == "retrieve":
            obs = last.get("observation") if isinstance(last.get("observation"), dict) else {}
            if int(obs.get("count") or 0) > 0:
                return Plan("已达检索上限，改为基于现有证据生成。", "answer", goal, "rules")

    chosen = choose_planner(settings, planner)
    if chosen == "llm" and not is_longform:
        from app.llm import LLMError

        try:
            return llm_plan(goal, history, settings)
        except (LLMError, ValueError, json.JSONDecodeError):
            return rule_plan(goal, history, settings=settings, state=state)
    return rule_plan(goal, history, settings=settings, state=state)


def rule_plan(
    goal: str,
    history: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    min_coverage: float | None = None,
    max_retrieves: int | None = None,
    state: AgentState | None = None,
) -> Plan:
    """Deterministic planner driven by the last observation (ReAct-style)."""
    from app.agent.partial import only_unverified_gaps

    if history and history[-1].get("action") == "verify" and only_unverified_gaps(history[-1].get("observation")):
        return Plan("保留未核实提示并交付，不因支持状态未知自动重新生成。", "finish", "", "rules")
    if state is not None and (state.reference or {}).get("action") == "ask":
        return Plan(
            "明确来源指代后再继续。",
            "finish" if any(step.get("action") == "compose" for step in history) else "compose",
            "",
            "rules",
        )
    if (
        state is not None
        and state.reading.get("mode") == "bounded"
        and history
        and history[-1].get("action") == "verify"
    ):
        return Plan("分别交付阅读、正文及引用检查结果；不以单一覆盖率追加生成循环。", "finish", "", "rules")
    if (
        state is not None
        and history
        and history[-1].get("action") == "verify"
        and state.reading.get("documents")
        and any(not d.get("complete") for d in state.reading["documents"].values())
    ):
        return Plan("必要阅读范围未覆盖，带未读和预算缺口交付部分内容，不再重复同一预算生成。", "finish", "", "rules")
    if state is not None and state.task_kind in {TASK_KIND_SYNTHESIS, TASK_KIND_DIGEST}:
        from app.agent.batch_reading import eligible, pending

        last_action = str(history[-1].get("action") or "") if history else ""
        if pending(state):
            return Plan("继续下一批任务原文，不调用中间生成模型。", "read", "", "rules")
        if last_action == "read":
            return Plan("读取已结束，按独立写稿预算交付并列出缺口。", "compose", "", "rules")
        if eligible(state, settings) and not state.reading.get("mode"):
            # Synthesis still runs its normal Hybrid search first. Digest can
            # directly read a named section rather than call an intermediate LLM.
            if last_action == "retrieve" or (
                state.task_kind == TASK_KIND_DIGEST and last_action in {"survey", "analyze"}
            ):
                return Plan("按步骤、条件和风险读取目标章节的有限原文批次。", "read", "", "rules")
    if state is not None and state.task_kind == TASK_KIND_DIGEST:
        return digest_rule_plan(goal, history, state, settings=settings)
    if state is not None and state.task_kind == TASK_KIND_SYNTHESIS:
        return synthesis_rule_plan(goal, history, state, settings=settings)

    coverage_floor = 0.5 if min_coverage is None else min_coverage
    retrieve_limit = 2 if max_retrieves is None else max_retrieves
    if settings is not None:
        coverage_floor = settings.agent_min_coverage_rate if min_coverage is None else min_coverage
        retrieve_limit = settings.agent_max_retrieves if max_retrieves is None else max_retrieves
    coverage_floor = max(0.0, min(1.0, float(coverage_floor)))
    retrieve_limit = max(1, int(retrieve_limit))
    from app.agent.analyze import split_factoid_queries

    queries = (
        [state.goal]
        if state is not None and (state.reference or {}).get("kind") == "step"
        else split_factoid_queries(goal)
    )
    retrieve_limit = max(retrieve_limit, len(queries))

    if not history:
        skipped = _refuse_live_gap(state)
        if skipped is not None:
            return skipped
        query = queries[0]
        if state is not None and state.retrieve_slots:
            from app.memory.query import build_retrieve_query

            query = build_retrieve_query(state.retrieve_slots) or query
        return Plan("先在语料里检索，再决定能否回答。", "retrieve", query, "rules")

    last = history[-1]
    last_action = str(last.get("action") or "")
    last_obs = last.get("observation") if isinstance(last.get("observation"), dict) else {}
    retrieve_count = sum(1 for item in history if item.get("action") == "retrieve")

    if last_action == "retrieve":
        hit_count = int(last_obs.get("count") or 0)
        if retrieve_count < len(queries) and retrieve_count < retrieve_limit:
            nxt = queries[retrieve_count]
            return Plan(
                f"复合问句：继续检索「{nxt}」。",
                "retrieve",
                nxt,
                "rules",
            )
        if hit_count <= 0:
            if retrieve_count < retrieve_limit and retrieve_count == 1 and len(queries) == 1:
                from app.agent.rewrite import rewrite_search_query

                last_q = str(last.get("input") or last_obs.get("query") or goal)
                rewritten = rewrite_search_query(goal, last_q, settings=settings)
                return Plan(
                    "检索没有命中，改写 query 后再检一次。",
                    "retrieve",
                    rewritten or goal,
                    "rules",
                )
            return Plan("检索没有命中，拒绝编造。", "refuse", "", "rules")
        return Plan("用已检索证据生成带引用的回答。", "answer", goal, "rules")

    if last_action == "answer":
        return Plan("核对每句声明是否被证据覆盖。", "verify", "", "rules")

    if last_action == "verify":
        coverage = _coverage_rate(last_obs)
        if coverage >= coverage_floor:
            return Plan(
                f"校验通过（coverage={coverage:.2f}），交付带引用的回答。",
                "finish",
                "",
                "rules",
            )
        if retrieve_count < retrieve_limit:
            retry_query = _retry_retrieve_query(goal, state)
            return Plan(
                f"覆盖率偏低（coverage={coverage:.2f}），补充检索。",
                "retrieve",
                retry_query,
                "rules",
            )
        from app.agent.partial import supported_count

        if supported_count(last_obs) > 0:
            return Plan(
                f"覆盖率仍偏低（coverage={coverage:.2f}），部分交付已覆盖内容。",
                "finish",
                "",
                "rules",
            )
        return Plan(
            f"覆盖率仍偏低（coverage={coverage:.2f}）且已达检索上限，拒绝编造。",
            "refuse",
            "",
            "rules",
        )

    if last_action in {"refuse", "finish", "error"}:
        return Plan("已结束。", "finish", "", "rules")

    # Unknown last action: restart from retrieve if needed.
    if retrieve_count == 0:
        return Plan("先在语料里检索，再决定能否回答。", "retrieve", _retrieval_query(goal), "rules")
    return Plan("状态异常，基于现有证据尝试回答。", "answer", goal, "rules")


def digest_rule_plan(
    goal: str,
    history: list[dict[str, Any]],
    state: AgentState,
    *,
    settings: Settings | None = None,
) -> Plan:
    """analyze → survey → digest(doc)* → compose → verify → finish."""
    from app.agent.digest import pending_digest_docs

    coverage_floor = 0.5
    if settings is not None:
        coverage_floor = max(0.0, min(1.0, float(settings.agent_synthesis_min_module_coverage)))
    last = history[-1] if history else {}
    last_action = str(last.get("action") or "")
    last_obs = last.get("observation") if isinstance(last.get("observation"), dict) else {}
    composed = any(item.get("action") == "compose" for item in history)
    retrieved = [node for node in state.nodes if node.status == "retrieved"]

    if not history:
        return Plan("先理解用户任务，再扫描知识库目录。", "analyze", "", "rules")
    if last_action == "analyze":
        return _after_analyze_plan(state)
    if last_action == "survey" or (last_action == "digest" and not last_obs.get("done")):
        if pending_digest_docs(state):
            return Plan("用模型阅读与任务最相关的下一篇文档。", "digest", "", "rules")
    if not composed:
        if retrieved or any(item.get("bullets") for item in state.digests):
            return Plan("基于已读懂的文档起草长回答。", "compose", "", "rules")
        return Plan("没有读到可用文档，拒绝编造。", "refuse", "", "rules")
    if last_action == "compose":
        if state.skips_retrieve() or state.compose_mode in {"memory", "catalog"}:
            return Plan("会话复述或目录盘点无需核对检索证据。", "finish", "", "rules")
        return Plan("核对长回答是否被证据覆盖。", "verify", "", "rules")
    if last_action == "verify":
        coverage = _coverage_rate(last_obs)
        if coverage >= coverage_floor or state.revised:
            return Plan(
                f"校验完成（coverage={coverage:.2f}），交付。",
                "finish",
                "",
                "rules",
            )
        state.revised = True
        return Plan(
            f"覆盖率偏低（coverage={coverage:.2f}），按缺口再起草一次。",
            "compose",
            "",
            "rules",
        )
    if last_action in {"refuse", "finish", "error"}:
        return Plan("已结束。", "finish", "", "rules")
    if retrieved:
        return Plan("基于已读文档结束本轮。", "finish", "", "rules")
    return Plan("无法继续，拒绝编造。", "refuse", "", "rules")


def synthesis_rule_plan(
    goal: str,
    history: list[dict[str, Any]],
    state: AgentState,
    *,
    settings: Settings | None = None,
) -> Plan:
    """analyze → survey → decompose → retrieve(question)* → compose → verify → revise(once) → finish."""
    coverage_floor = 0.5
    if settings is not None:
        coverage_floor = max(0.0, min(1.0, float(settings.agent_synthesis_min_module_coverage)))
    retrieve_limit = max(1, int(state.max_retrieves))
    last = history[-1] if history else {}
    last_action = str(last.get("action") or "")
    last_obs = last.get("observation") if isinstance(last.get("observation"), dict) else {}
    composed = any(item.get("action") == "compose" for item in history)
    pending = [node for node in state.nodes if node.status == "pending"]
    retrieved = [node for node in state.nodes if node.status == "retrieved"]

    if not history:
        return Plan("先理解用户任务，再扫描知识库目录。", "analyze", "", "rules")

    if last_action == "analyze":
        return _after_analyze_plan(state)

    if last_action == "survey":
        if state.catalog_inventory():
            return Plan("按目录盘点知识库覆盖面，不按主题检索。", "compose", "", "rules")
        intent = state.task_spec.intent if state.task_spec else ""
        if intent == "compare" and not any(item.get("action") == "graph_walk" for item in history):
            return Plan("对比任务先看图谱上的共用关系类型，没有图就跳过。", "graph_walk", "", "rules")
        return Plan("按用户子问题规划检索，禁止整库前 N 章。", "decompose", "", "rules")

    if last_action == "graph_walk":
        return Plan("按对比维度或原问题规划检索。", "decompose", "", "rules")

    if last_action == "decompose" and last_obs.get("probe_needed") and state.subtopic_retrieves < retrieve_limit:
        if not any(
            isinstance(item.get("observation"), dict) and item.get("observation", {}).get("probe") for item in history
        ):
            return Plan("词面命中不足，用主题做一次探测检索。", "retrieve", _topic_query(goal, state), "rules")

    if last_action == "retrieve" and int(last_obs.get("fresh_count") or 0) == 0:
        node_id = str(last_obs.get("node_id") or last.get("node_id") or "")
        node = next((item for item in state.nodes if item.id == node_id), None)
        node_tries = sum(
            1
            for item in history
            if item.get("action") == "retrieve"
            and (
                (isinstance(item.get("observation"), dict) and item["observation"].get("node_id") == node_id)
                or item.get("node_id") == node_id
            )
        )
        if node is not None and node_tries == 1 and state.subtopic_retrieves < retrieve_limit:
            from app.agent.rewrite import rewrite_search_query

            filenames = [doc.filename for doc in state.catalog]
            rewritten = rewrite_search_query(
                node.question or node.title or goal,
                node.query,
                settings=settings,
                filenames=filenames,
            )
            node.status = "pending"
            node.query = rewritten or node.query
            return Plan(
                f"子问题未命中，改写后再检「{node.question or node.title}」。",
                "retrieve",
                node.query,
                "rules",
                args={"node_id": node.id},
            )

    if pending and state.subtopic_retrieves < retrieve_limit:
        node = pending[0]
        return Plan(
            f"检索子问题「{node.question or node.title}」。",
            "retrieve",
            node.query,
            "rules",
            args={"node_id": node.id},
        )

    if not composed:
        if retrieved:
            return Plan("基于已检索节点起草结构化结果。", "compose", "", "rules")
        return Plan("没有检索到可用节点，拒绝编造。", "refuse", "", "rules")

    if last_action == "compose":
        if state.skips_retrieve() or state.compose_mode in {"memory", "catalog"}:
            return Plan("会话复述或目录盘点无需核对检索证据。", "finish", "", "rules")
        return Plan("核对结构化结果是否被证据覆盖。", "verify", "", "rules")

    if last_action == "verify":
        coverage = _coverage_rate(last_obs)
        if coverage >= coverage_floor:
            return Plan(
                f"校验通过（coverage={coverage:.2f}），交付。",
                "finish",
                "",
                "rules",
            )
        if state.subtopic_retrieves < retrieve_limit:
            retry_node = _retry_synthesis_node(state, settings)
            if retry_node is not None:
                retry_node.retried = True
                retry_node.status = "pending"
                retry_node.query = _retry_retrieve_query(retry_node.question or retry_node.title or goal, state)
                kind = "空节点" if not retry_node.chunk_ids else "弱节点"
                return Plan(
                    f"覆盖率偏低（coverage={coverage:.2f}），补检索{kind}「{retry_node.question or retry_node.title}」。",
                    "retrieve",
                    retry_node.query,
                    "rules",
                    args={"node_id": retry_node.id},
                )
        if not state.revised:
            state.revised = True
            return Plan(
                f"覆盖率偏低（coverage={coverage:.2f}），按缺口再起草一次。",
                "compose",
                "",
                "rules",
            )
        return Plan(
            f"覆盖率偏低（coverage={coverage:.2f}），带缺口交付。",
            "finish",
            "",
            "rules",
        )

    if last_action in {"refuse", "finish", "error"}:
        return Plan("已结束。", "finish", "", "rules")

    if retrieved:
        return Plan("基于已有节点结束本轮。", "finish", "", "rules")
    return Plan("无法继续，拒绝编造。", "refuse", "", "rules")


def llm_plan(goal: str, history: list[dict[str, Any]], settings: Settings) -> Plan:
    from app.llm import get_llm_client

    client = get_llm_client(settings)
    compact = []
    for item in history[-8:]:
        compact.append(
            {
                "action": item.get("action"),
                "input": item.get("input"),
                "observation": _compact_observation(item.get("observation")),
            }
        )
    prompt = (
        f"用户目标：{goal}\n"
        f"已执行步骤：{json.dumps(compact, ensure_ascii=False)}\n"
        f"最低覆盖率阈值：{settings.agent_min_coverage_rate}\n"
        f"最大检索次数：{settings.agent_max_retrieves}\n"
        "请输出下一步 JSON。"
    )
    raw = client.generate(
        system=PLANNER_SYSTEM,
        prompt=prompt,
        max_output_tokens=min(400, settings.llm_max_output_tokens),
        temperature=0.0,
        purpose="planner",
    ).text
    data = _parse_json_object(raw)
    action = str(data.get("action") or "").strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"unknown planner action: {action}")
    thought = str(data.get("thought") or "").strip()
    payload = str(data.get("input") or data.get("query") or "").strip()
    return Plan(thought or "llm", action, payload, "llm")


def _after_analyze_plan(state: AgentState) -> Plan:
    if state.skips_retrieve():
        return Plan("根据上一轮回答直接复述或澄清，不检索。", "compose", "", "rules")
    if state.catalog_inventory():
        if state.catalog:
            return Plan("按目录盘点知识库覆盖面，不按主题检索。", "compose", "", "rules")
        return Plan("先扫描知识库目录，再按文档盘点。", "survey", "", "rules")
    intent = state.task_spec.intent if state.task_spec else ""
    if state.catalog:
        if state.task_kind == TASK_KIND_DIGEST:
            return Plan("沿用上轮目录，继续按篇阅读。", "digest", "", "rules")
        if intent == "compare":
            return Plan("对比任务先看图谱上的共用关系类型，没有图就跳过。", "graph_walk", "", "rules")
        return Plan("沿用上轮目录，按修正后的主题规划检索。", "decompose", "", "rules")
    if state.task_kind == TASK_KIND_DIGEST:
        return Plan("先扫描知识库目录，再逐篇读懂文档。", "survey", "", "rules")
    return Plan("先扫描知识库目录，再按子问题检索。", "survey", "", "rules")


def _coverage_rate(observation: dict[str, Any]) -> float:
    raw = observation.get("module_coverage_rate")
    if raw is None:
        raw = observation.get("coverage_rate")
    try:
        grounded = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        grounded = 0.0
    task = observation.get("task_coverage")
    if task is None:
        return grounded
    try:
        return min(grounded, float(task))
    except (TypeError, ValueError):
        return grounded


def _topic_query(goal: str, state: AgentState | None = None) -> str:
    """Prefer corrected subjects / sub-questions over the raw user utterance."""
    if state is not None and state.task_spec is not None:
        questions = [str(item).strip() for item in (state.task_spec.questions or []) if str(item).strip()]
        if questions:
            return questions[0][:160]
        subjects = [str(item).strip() for item in (state.task_spec.subjects or []) if str(item).strip()]
        if subjects:
            return "、".join(subjects)[:160]
    return _retrieval_query(goal)


def _retrieval_query(goal: str) -> str:
    """Search with the topic, not the chat wrapper（「我是一名…，为我…」）."""
    from app.corpus.relevance import strip_chat

    cleaned = strip_chat(goal)
    return (cleaned or (goal or "").strip())[:160]


def _retry_synthesis_node(state: AgentState, settings: Settings | None) -> Any:
    """Empty nodes first, then retrieved-but-weak nodes that have not been retried."""
    empty = next((node for node in state.nodes if node.status == "empty" and not node.retried), None)
    if empty is not None:
        return empty
    floor = 0.15
    if settings is not None:
        floor = float(getattr(settings, "agent_weak_node_score", 0.15) or 0.15)
    return next(
        (
            node
            for node in state.nodes
            if node.status == "retrieved" and not node.retried and float(node.score or 0.0) < floor
        ),
        None,
    )


def _retry_retrieve_query(goal: str, state: AgentState | None = None) -> str:
    """Follow-up query from the user's own uncovered wording.

    Never build it from rejected claims: those may be invented, and searching
    with them drifts further away from the corpus.
    """
    from app.corpus.relevance import uncovered_terms

    base = _retrieval_query(goal)
    blob = " ".join((item.text or "") for item in state.evidence) if state is not None else ""
    missing = uncovered_terms(goal, blob, limit=3) if blob else []
    if missing:
        return f"{base} {' '.join(missing)}"[:160]
    return f"{base} 定义 要点"[:160]


def _compact_observation(observation: Any) -> Any:
    if not isinstance(observation, dict):
        return observation
    hits = observation.get("hits")
    if isinstance(hits, list):
        trimmed = []
        for hit in hits[:5]:
            if not isinstance(hit, dict):
                continue
            trimmed.append(
                {
                    "index": hit.get("index"),
                    "page": hit.get("page"),
                    "score": hit.get("score"),
                    "snippet": str(hit.get("snippet") or hit.get("text") or "")[:180],
                }
            )
        return {
            "count": observation.get("count", len(trimmed)),
            "hits": trimmed,
            "coverage_rate": observation.get("coverage_rate"),
            "module_coverage_rate": observation.get("module_coverage_rate"),
            "task_coverage": observation.get("task_coverage"),
            "hallucination_rate": observation.get("hallucination_rate"),
            "answer_preview": str(observation.get("answer") or "")[:200],
        }
    return {
        key: observation.get(key)
        for key in (
            "count",
            "coverage_rate",
            "module_coverage_rate",
            "task_coverage",
            "hallucination_rate",
            "status",
            "answer",
            "claims",
        )
        if key in observation
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    from app.llm import parse_json_object

    data = parse_json_object(raw)
    if isinstance(data, dict):
        return data
    raise ValueError("planner did not return a JSON object")
