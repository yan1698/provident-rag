"""Agent tools. Every action the planner can emit is a Tool in the registry.

There are no privileged actions: ``analyze`` and ``decompose`` are tools like any
other, so the executor never needs to know their names, and per-action parameters
travel in ``plan.args`` instead of widening the ``execute`` signature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from app.agent.planner import Plan
from app.agent.state import AgentPhase, AgentState, GapItem
from app.agent.store import AgentStore
from app.agent.survey import apply_survey
from app.generate.prompt import EvidenceItem

if TYPE_CHECKING:
    from app.core.config import Settings

_REFUSE_ANSWER = "现有证据不足以回答该问题。"
_NO_EVIDENCE_ANSWER = "现有证据不足以回答该问题（检索未命中任何相关片段）。"


def _source_refusal(state, message):
    state.answer = message
    state.evidence = []
    state.citations = []
    state.compose_mode = "refused"
    state.outline = {
        **(state.outline or {}),
        "modules": [],
        "evidence": [],
        "source_error": message,
        "evidence_selection": state.evidence_selection,
    }
    return ToolResult({"status": "refused", "reason": message}, terminal=True)


def _stamp_retrieve(observation: dict[str, Any], state: AgentState, store: Any) -> None:
    try:
        from app.corpus.index_state import retrieve_index_stamp

        session = getattr(store, "session", None)
        if session is not None:
            observation.update(retrieve_index_stamp(session, state.owner_id))
        else:
            observation.setdefault("search_executed", True)
            observation.setdefault("downstream_ok", True)
    except Exception:  # noqa: BLE001 — stamp must not fail retrieve
        observation.setdefault("search_executed", True)
        observation.setdefault("downstream_ok", True)
    note = str(getattr(store, "scope_note", "") or "")
    if note:
        observation["scope_note"] = note
        if isinstance(state.scope, dict):
            state.scope["widened"] = True
    state.last_retrieve_obs = dict(observation)


def _extract_citations(answer: str, evidence: list[EvidenceItem]):
    from app.generate.service import _extract_citations as extract

    return extract(answer, evidence)


ALL_TASK_KINDS = frozenset({"factoid", "synthesis", "digest"})


@dataclass(frozen=True)
class ToolResult:
    observation: dict[str, Any]
    terminal: bool = False


@dataclass
class ToolContext:
    """Everything a tool may touch. Widen this, not the ``execute`` signature."""

    state: AgentState
    settings: Settings
    store: AgentStore
    deadline: float | None = None

    @property
    def db(self):
        """Transitional: digest / neighbors still accept a Session. Prefer ``store``."""
        return getattr(self.store, "session", None)


class Tool(ABC):
    name: str
    description: str
    # 哪些任务线可以发这个动作。guards 的动作白名单由此推导，不再手写第二份清单。
    task_kinds: frozenset[str] = ALL_TASK_KINDS

    @abstractmethod
    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        """Run the tool. ``plan.input`` is the main payload, ``plan.args`` the rest."""


class AnalyzeTool(Tool):
    name = "analyze"
    description = "把目标拆成意图、主体与待答子问题"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        from app.agent.analyze import apply_analyze

        observation = apply_analyze(
            ctx.state,
            requested_intent=ctx.state.requested_intent,
            settings=ctx.settings,
        )
        return ToolResult(observation)


class SurveyTool(Tool):
    name = "survey"
    description = "扫描当前知识库的文档目录与章节标题，不走向量检索"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        settings = ctx.settings
        if ctx.state.catalog:
            from app.corpus.catalog import CatalogBuild

            built = CatalogBuild(documents=tuple(ctx.state.catalog), truncated=False)
            observation, terminal = apply_survey(ctx.state, built)
            observation["cached"] = True
            return ToolResult(observation, terminal)
        max_sections = int(getattr(settings, "agent_synthesis_max_catalog_sections", 80) or 80)
        built = ctx.store.load_catalog(goal=ctx.state.goal, max_sections=max_sections)
        observation, terminal = apply_survey(ctx.state, built)
        return ToolResult(observation, terminal)


class DecomposeTool(Tool):
    name = "decompose"
    description = "把综合任务拆成可分别检索的子问题节点"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        from app.agent.decompose import apply_decompose

        settings = ctx.settings
        observation = apply_decompose(
            ctx.state,
            max_subtopics=int(getattr(settings, "agent_synthesis_max_subtopics", 6) or 6),
            score_min=float(getattr(settings, "agent_synthesis_section_score_min", 0.1) or 0.1),
            allow_model=True,
        )
        return ToolResult(observation)


class RetrieveTool(Tool):
    name = "retrieve"
    description = "从语料库检索相关文档"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        state, settings, store = ctx.state, ctx.settings, ctx.store
        node_id = plan.node_id
        state.phase = AgentPhase.RETRIEVING
        query = plan.input.strip() or state.goal
        if (state.reference or {}).get("retrieve"):
            query += "；指定操作步骤：" + str(state.reference.get("text") or "")
        if state.retrieve_slots and not plan.node_id and not plan.input.strip():
            from app.memory.query import build_retrieve_query

            built = build_retrieve_query(state.retrieve_slots)
            if built:
                query = built
        hits = store.search(query, settings=settings)
        floor = float(getattr(settings, "agent_retrieve_min_score", 0.0) or 0.0)
        dropped = 0
        if floor > 0.0:
            strong = [hit for hit in hits if float(getattr(hit, "score", 0.0) or 0.0) >= floor]
            dropped = len(hits) - len(strong)
            hits = strong
        scores = {str(hit.chunk_id): float(getattr(hit, "score", 0.0) or 0.0) for hit in hits}
        fresh = _evidence_from_hits(store, hits, scores=scores)
        anchored = []
        if state.reference and state.reference.get("retrieve"):
            from app.agent.reading import reference_rows, step_context_rows

            loader = (
                step_context_rows
                if state.task_kind == "factoid" and state.reference.get("kind") == "step"
                else reference_rows
            )
            anchored, error = loader(state, store)
            if error:
                state.answer = error
                state.evidence = []
                state.citations = []
                state.gaps.append(GapItem(topic="步骤来源", reason=error))
                return ToolResult({"status": "refused", "reason": error}, terminal=True)
        if state.task_kind == "synthesis":
            observation, terminal = _execute_synthesis_retrieve(
                state, settings, query, fresh, node_id, store=store, scores=scores
            )
            if anchored:
                for row in anchored:
                    if not any(e.chunk_id == row.id for e in state.evidence):
                        state.evidence.append(
                            EvidenceItem(
                                index=len(state.evidence) + 1,
                                chunk_id=row.id,
                                doc_id=row.doc_id,
                                page=row.page,
                                text=row.text,
                                chunk_hash=row.chunk_hash,
                                start_offset=row.start_offset,
                                section_path=row.section_path,
                                role="reference",
                            )
                        )
                node = next((n for n in state.nodes if n.id == node_id), None)
                if node:
                    node.chunk_ids = list(dict.fromkeys(node.chunk_ids + [r.id for r in anchored]))
                    node.status = "retrieved"
                observation["fresh_count"] = max(1, observation.get("fresh_count", 0))
                observation["reference_reread"] = [r.id for r in anchored]
                terminal = False
            _stamp_retrieve(observation, state, store)
            return ToolResult(observation, terminal)

        if anchored:
            bound = [
                EvidenceItem(
                    index=i + 1,
                    chunk_id=row.id,
                    doc_id=row.doc_id,
                    page=row.page,
                    text=row.text,
                    chunk_hash=row.chunk_hash,
                    section_path=row.section_path,
                    start_offset=row.start_offset,
                    role="reference",
                )
                for i, row in enumerate(anchored)
            ]
            fresh = _merge_evidence(bound, fresh)
        fresh, neighbor_obs = store.expand_neighbors(fresh, settings=settings)
        fresh, graph_obs = store.expand_graph(fresh, goal=query, settings=settings)

        state.retrieve_attempts += 1

        if state.retrieve_attempts <= 1 or not state.evidence:
            state.evidence = fresh
        else:
            state.evidence = _merge_evidence(state.evidence, fresh)

        state.hit_count = len(state.evidence)
        observation = {
            "count": len(state.evidence),
            "fresh_count": len(fresh),
            "dropped_low_score": dropped,
            "query": query,
            "hits": [
                {
                    "index": item.index,
                    "chunk_id": item.chunk_id,
                    "doc_id": item.doc_id,
                    "page": item.page,
                    "snippet": item.text[:240],
                    "chunk_hash": item.chunk_hash,
                    "role": getattr(item, "role", "hit"),
                }
                for item in state.evidence
            ],
        }
        observation.update(neighbor_obs)
        observation.update(graph_obs)
        if anchored:
            observation["reference_reread"] = [row.id for row in anchored]
        _stamp_retrieve(observation, state, store)
        return ToolResult(observation)


class AnswerTool(Tool):
    name = "answer"
    description = "基于证据生成带引用的回答"
    task_kinds = frozenset({"factoid"})

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        from app.generate.service import answer_from_evidence

        state = ctx.state
        state.phase = AgentPhase.ANSWERING
        result = answer_from_evidence(state.goal, state.evidence, settings=ctx.settings)
        if result.evidence_selection.get("invalid_citation_indices"):
            state.evidence_selection["answer"] = result.evidence_selection
            return _source_refusal(state, "模型引用了未投喂的来源，已拒绝交付该回答。")
        state.answer = result.answer
        state.evidence = result.evidence
        # Persist the generator's actual evidence whitelist, including for factoid
        # follow-ups whose outline was seeded from a previous long-form answer.
        from dataclasses import asdict

        state.outline = {**(state.outline or {}), "evidence": [asdict(item) for item in result.evidence]}
        state.hit_count = len(state.evidence)
        state.citations = [
            {
                "index": item.index,
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "page": item.page,
            }
            for item in result.citations
        ]
        return ToolResult(
            {
                "answer": result.answer,
                "citation_count": len(state.citations),
                "provider": result.provider,
                "evidence_selection": result.evidence_selection,
            }
        )


class DigestTool(Tool):
    name = "digest"
    description = "用 DeepSeek 阅读一篇上传文档并抽出与任务相关的要点"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        from app.agent.digest import apply_digest

        return ToolResult(apply_digest(ctx.state, ctx.store, ctx.settings))


def _stamp_paragraphs(state: AgentState, reference: dict | None = None, *, carried_anchor: str = "") -> None:
    from app.agent.memory import index_paragraphs, index_operation_steps

    if not isinstance(state.outline, dict):
        state.outline = {}
    state.outline["paragraphs"] = index_paragraphs(state.answer)
    by_index = {item.index: item for item in state.evidence}
    for paragraph in state.outline["paragraphs"]:
        text = state.answer[paragraph["start"] : paragraph["end"]]
        sources = _extract_citations(text, state.evidence)
        paragraph["chunk_ids"] = [item.chunk_id for item in sources]
        paragraph["source_proofs"] = []
        import hashlib

        for source in sources:
            item = by_index.get(source.index)
            if item:
                paragraph["source_proofs"].append(
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "chunk_hash": item.chunk_hash,
                        "text_sha256": hashlib.sha256(item.text.encode()).hexdigest(),
                    }
                )
    steps = index_operation_steps(
        state.outline.get("modules") or [],
        state.evidence,
        operation_plan=bool(state.task_spec and state.task_spec.intent == "plan"),
    )
    state.outline["steps"] = steps
    if steps and not (state.operation_anchor and state.task_spec and state.task_spec.intent != "plan"):
        state.outline["operation_anchor"] = {"fresh": True}
        state.outline["operation_steps"] = steps
    elif state.operation_anchor:
        state.outline["operation_anchor"] = {"run_id": state.operation_anchor}
        state.outline["operation_steps"] = state.operation_steps
    if state.reference:
        state.outline["reference"] = {
            k: v
            for k, v in state.reference.items()
            if k in {"kind", "index", "indices", "anchors", "action", "anchor_run_id"}
        }
        state.outline["anchor_run_id"] = (
            state.operation_anchor
            if state.reference.get("kind") == "step"
            else state.reference.get("anchor_run_id", "")
        )
    if state.compose_mode in {"memory", "reference"} and carried_anchor:
        state.outline["answer_anchor"] = {"run_id": carried_anchor}
    elif state.compose_mode in {"written", "catalog"}:
        state.outline["answer_anchor"] = {"fresh": True}
    if not reference or reference.get("action") != "ask":
        return
    goal = state.goal or ""
    if "展开" in goal:
        pending_action = "expand"
    elif "为什么" in goal or "为何" in goal or "什么意思" in goal:
        pending_action = "explain"
    else:
        pending_action = "recite"
    state.outline["pending_paragraphs"] = list(reference.get("candidates") or [])
    state.outline["source_answer"] = str(reference.get("source_answer") or "")
    state.outline["pending_action"] = pending_action
    state.outline["pending_reference_kind"] = reference.get("kind", "paragraph")


def _compose_reference(state: AgentState, settings: Any, reference: dict) -> str:
    action = str(reference.get("action") or "")
    if action == "ask":
        return str(reference.get("message") or "")
    text = str(reference.get("text") or "")
    if action == "recite" or not text:
        return text
    # Legacy answers without source identity may be recited, never expanded into
    # new facts solely from answer memory. Source-bound expansion uses retrieval.
    if not reference.get("chunk_ids"):
        return text
    system = "你只处理用户指定的这一段原文。不要检索，不要引入原文没有的主题，不要把内容改写成实施方案。"
    if action == "expand":
        prompt = f"请展开下面这一段，仍然只讨论这一段里的内容。\n\n原文：\n{text}\n\n用户要求：{state.goal}"
    else:
        prompt = f"用户在问下面这一段为什么这样写。只根据这一段解释。\n\n原文：\n{text}\n\n用户原话：{state.goal}"
    try:
        from app.llm import LLMError, get_llm_client

        return (
            get_llm_client(settings)
            .generate(
                system=system,
                prompt=prompt,
                max_output_tokens=min(800, int(getattr(settings, "llm_max_output_tokens", 1024) or 1024)),
                temperature=0.2,
                purpose="reference",
            )
            .text
            or text
        )
    except (LLMError, ValueError, TypeError, AttributeError):
        return text


class ReadTool(Tool):
    name = "read"
    description = "在有限字符和批次预算内读取下一批任务原文"
    task_kinds = frozenset({"synthesis", "digest"})

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        from app.agent.batch_reading import read_batch
        from app.llm.meter import _STOP

        ctx.state.phase = AgentPhase.DIGESTING
        result = read_batch(ctx.state, ctx.store, ctx.settings, stop=_STOP.get(), deadline=ctx.deadline)
        return ToolResult(result)


class ComposeTool(Tool):
    name = "compose"
    description = "基于已检索证据起草可读长回答"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        from app.agent.evidence import select_global_evidence
        from app.generate.synthesis import compose_synthesis, evidence_snapshot, reindex_evidence_state

        state, settings = ctx.state, ctx.settings
        state.phase = AgentPhase.COMPOSING
        if state.skips_retrieve():
            from app.agent.memory import compose_followup_answer

            carried_anchor = ""
            if isinstance(state.outline, dict):
                raw_anchor = state.outline.get("answer_anchor")
                if isinstance(raw_anchor, dict) and raw_anchor.get("run_id"):
                    carried_anchor = str(raw_anchor["run_id"])
                elif state.outline.get("anchor_run_id"):
                    carried_anchor = str(state.outline["anchor_run_id"])
            reference = state.outline.get("reference") if isinstance(state.outline, dict) else None
            if isinstance(reference, dict) and reference.get("hit"):
                text = _compose_reference(state, settings, reference)
                compose_mode = "reference"
            else:
                reference = None
                text = compose_followup_answer(
                    state.goal,
                    state.memory_brief,
                    state.prior_answer,
                    state.turn_kind,
                    catalog=state.catalog,
                )
                compose_mode = "memory"
            state.answer = text
            state.compose_mode = compose_mode
            state.outline = {
                "mode": state.task_spec.intent if state.task_spec else "explain",
                "modules": [{"heading": "上一轮", "paragraphs": [{"text": text, "sources": []}]}],
                "order_note": "",
                "evidence": [],
                "task_spec": state.task_spec.to_dict() if state.task_spec else None,
                "compose_mode": compose_mode,
                "turn_kind": state.turn_kind,
            }
            if isinstance(reference, dict) and reference.get("index"):
                state.outline["reference"] = {
                    "index": reference.get("index"),
                    "action": str(reference.get("action") or ""),
                }
            _stamp_paragraphs(
                state,
                reference if isinstance(reference, dict) else None,
                carried_anchor=carried_anchor,
            )
            state.citations = []
            return ToolResult(
                {
                    "modules": state.outline["modules"],
                    "gaps": [],
                    "answer": state.answer,
                    "compose_mode": compose_mode,
                    "turn_kind": state.turn_kind,
                    "citation_count": 0,
                }
            )
        if state.catalog_inventory():
            from app.agent.memory import compose_catalog_listing

            text = compose_catalog_listing(state.catalog)
            state.answer = text
            state.compose_mode = "catalog"
            state.outline = {
                "mode": "inventory",
                "modules": [{"heading": "知识库目录", "paragraphs": [{"text": text, "sources": []}]}],
                "order_note": "",
                "evidence": [],
                "task_spec": state.task_spec.to_dict() if state.task_spec else None,
                "compose_mode": "catalog",
            }
            state.citations = []
            _stamp_paragraphs(state)
            return ToolResult(
                {
                    "modules": state.outline["modules"],
                    "gaps": [],
                    "answer": state.answer,
                    "compose_mode": "catalog",
                    "citation_count": 0,
                }
            )
        from app.agent.reading import read_for_synthesis

        if state.reading.get("mode") == "bounded":
            from app.agent.batch_reading import revalidate

            revalidate(state, ctx.store)
        else:
            read_for_synthesis(state, ctx.store, settings)
        cull = {"kept": len(state.evidence), "dropped": 0, "note": ""}
        if state.task_kind == "synthesis" and not state.reading:
            cull = select_global_evidence(
                state,
                cap=int(getattr(settings, "agent_synthesis_max_evidence", 16) or 16),
            )
        reindex_evidence_state(state)
        from app.agent.reading import evidence_identity_error

        error = evidence_identity_error(state.evidence, ctx.store)
        if error:
            return _source_refusal(state, error)
        draft = compose_synthesis(state, settings)
        error = evidence_identity_error(state.evidence, ctx.store)
        if error:
            return _source_refusal(state, error)
        state.answer = draft.markdown
        state.compose_mode = draft.compose_mode
        state.fallback_reason = draft.fallback_reason
        state.gaps = [GapItem(topic=item["topic"], reason=item["reason"]) for item in draft.gaps]
        state.outline = {
            "mode": draft.mode,
            "modules": draft.modules,
            "order_note": draft.order_note,
            "evidence": evidence_snapshot(state),
            "task_spec": state.task_spec.to_dict() if state.task_spec else None,
            "compose_mode": draft.compose_mode,
            "fallback_reason": draft.fallback_reason,
            "evidence_selection": state.evidence_selection,
            "reading": state.reading,
            "run_budget": state.run_budget,
            "task_update": state.task_spec.update if state.task_spec else "new",
        }
        _stamp_paragraphs(state)
        if state.reading:
            from app.verify.service import reconcile_delivery

            state.reading["delivery"] = reconcile_delivery(state, ctx.store)
        state.citations = [
            {
                "index": item.index,
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "page": item.page,
            }
            for item in _extract_citations(state.answer, state.evidence)
        ]
        observation = {
            "modules": draft.modules,
            "gaps": draft.gaps,
            "order_note": draft.order_note,
            "stripped": draft.stripped,
            "answer": state.answer,
            "citation_count": len(state.citations),
            "compose_mode": draft.compose_mode,
            "fallback_reason": draft.fallback_reason,
            "evidence_kept": cull.get("kept"),
        }
        if cull.get("note"):
            observation["note"] = cull["note"]
        if draft.compose_mode == "refused":
            observation["status"] = "refused"
        return ToolResult(observation, terminal=draft.compose_mode == "refused")


class VerifyTool(Tool):
    name = "verify"
    description = "对回答做蕴含校验"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        state, settings = ctx.state, ctx.settings
        state.phase = AgentPhase.VERIFYING
        if not state.answer:
            return ToolResult({"error": "no answer to verify", "error_kind": "precondition"})
        from app.agent.reading import evidence_identity_error
        from app.generate.synthesis import CitationContractError, check_source_contract

        error = evidence_identity_error(state.evidence, ctx.store)
        if error:
            return _source_refusal(state, error)
        try:
            check_source_contract(state.answer, {item.index for item in state.evidence})
        except CitationContractError as exc:
            return _source_refusal(state, str(exc))
        if state.task_kind in {"synthesis", "digest"} and state.outline:
            from app.verify.service import reconcile_delivery, verify_synthesis

            questions = list(state.task_spec.questions) if state.task_spec else []
            result = verify_synthesis(
                modules=list(state.outline.get("modules") or []),
                order_note=str(state.outline.get("order_note") or ""),
                gaps=[{"topic": gap.topic, "reason": gap.reason} for gap in state.gaps],
                evidence=state.evidence,
                settings=settings,
                task_questions=questions,
            )
            payload_out = asdict(result)
            payload_out["module_coverage_rate"] = result.coverage_rate
            payload_out["task_coverage"] = result.task_coverage
            payload_out["meta_count"] = sum(1 for item in result.claims if item.status == "meta")
            payload_out.update(_graph_verify_meta(state))
            state.verification = payload_out
            if state.reading:
                complete = all(d.get("complete") for d in state.reading.get("documents", {}).values())
                payload_out["reading_complete"] = complete
                if state.reading.get("mode") == "bounded":
                    payload_out["answer_task_coverage"] = payload_out.get("task_coverage")
                    payload_out["task_coverage"] = None
                    payload_out["task_completeness"] = (
                        "unknown: evaluate reading, delivered facts, and source support separately"
                    )
                elif not complete:
                    payload_out["answer_task_coverage"] = payload_out.get("task_coverage")
                    payload_out["task_coverage"] = 0.0  # full requested reading is not achieved
            state.task_coverage = payload_out["task_coverage"]
            if state.reading:
                state.reading["delivery"] = reconcile_delivery(state, ctx.store)
                state.outline["reading"] = state.reading
            return ToolResult(state.verification)
        from app.verify.service import verify_answer

        result = verify_answer(state.answer, state.evidence, settings=settings)
        state.verification = asdict(result)
        state.verification.update(_graph_verify_meta(state))
        return ToolResult(state.verification)


class GraphWalkTool(Tool):
    name = "graph_walk"
    description = "对比任务：用图谱邻居关系类型作为维度，图未构建则跳过"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        state, settings = ctx.state, ctx.settings
        if not bool(getattr(settings, "graph_enabled", True)):
            return ToolResult({"skipped": True, "reason": "图谱已关闭", "dimensions": [], "edges": []})
        counts = ctx.store.graph_counts()
        if int(counts.get("entities") or 0) <= 0:
            return ToolResult({"skipped": True, "reason": "图谱未构建", "dimensions": [], "edges": []})
        subjects = list((state.task_spec.subjects if state.task_spec else []) or [])
        result = ctx.store.graph_walk(
            subjects,
            hops=max(1, min(int(getattr(settings, "graph_expand_hops", 1) or 1), 2)),
        )
        if result.get("skipped"):
            return ToolResult(result)
        state.graph_dimensions = list(result.get("dimensions") or [])
        state.graph_used_edges = list(result.get("edges") or [])
        return ToolResult(result)


class RefuseTool(Tool):
    name = "refuse"
    description = "证据不足时拒绝回答"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        state = ctx.state
        state.phase = AgentPhase.DONE
        state.answer = _NO_EVIDENCE_ANSWER if not state.hit_count else _REFUSE_ANSWER
        state.citations = []
        return ToolResult({"answer": state.answer, "status": "refused"}, terminal=True)


class FinishTool(Tool):
    name = "finish"
    description = "交付最终回答并结束"

    def execute(self, plan: Plan, ctx: ToolContext) -> ToolResult:
        state, settings = ctx.state, ctx.settings
        state.phase = AgentPhase.DONE
        extra: dict[str, Any] = {}
        dropped = bool(plan.input.strip())
        if state.task_kind == "factoid" and state.verification:
            from app.agent.partial import apply_partial_answer, should_partial

            floor = float(getattr(settings, "agent_min_coverage_rate", 0.5) or 0.5)
            if should_partial(state.verification, coverage_floor=floor):
                extra = apply_partial_answer(state)
        if state.answer:
            state.citations = [
                {
                    "index": item.index,
                    "chunk_id": item.chunk_id,
                    "doc_id": item.doc_id,
                    "page": item.page,
                }
                for item in _extract_citations(state.answer, state.evidence)
            ]
        if not state.answer:
            state.answer = _REFUSE_ANSWER
        result = {"answer": state.answer, "citation_count": len(state.citations)}
        result.update(extra)
        if dropped:
            result["payload_ignored"] = True
        return ToolResult(result, terminal=True)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> frozenset[str]:
        """每个已注册动作。guards 拿它当白名单，所以是 frozenset 而非有序元组。"""
        return frozenset(self._tools)

    def names_for(self, task_kind: str) -> frozenset[str]:
        return frozenset(name for name, tool in self._tools.items() if task_kind in tool.task_kinds)


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        AnalyzeTool(),
        SurveyTool(),
        DecomposeTool(),
        ReadTool(),
        DigestTool(),
        RetrieveTool(),
        GraphWalkTool(),
        ComposeTool(),
        AnswerTool(),
        VerifyTool(),
        RefuseTool(),
        FinishTool(),
    ):
        registry.register(tool)
    return registry


def _execute_synthesis_retrieve(
    state: AgentState,
    settings: Settings,
    query: str,
    fresh: list[EvidenceItem],
    node_id: str,
    *,
    store: AgentStore | None = None,
    scores: dict[str, float] | None = None,
) -> tuple[dict[str, Any], bool]:
    from app.agent.decompose import (
        append_probe_nodes,
        count_section_hits,
        filter_hits_for_question,
        map_hit_to_sections,
        question_binds_section,
    )

    state.subtopic_retrieves += 1
    node = next((item for item in state.nodes if item.id == node_id), None) if node_id else None
    probe = node is None
    if node is not None:
        question = node.question or node.title or query
        fresh = filter_hits_for_question(
            question,
            fresh,
            scores=scores,
            semantic_min=float(getattr(settings, "agent_hit_semantic_min", 0.1) or 0.1),
            exempt_doc_ids={node.doc_id} if node.doc_id else None,
        )
    neighbor_obs = {"neighbors_added": 0, "neighbors_dropped": 0, "note": ""}
    graph_obs = {"graph_added": 0, "graph_dropped": 0, "note": ""}
    if store is not None and fresh:
        fresh, neighbor_obs = store.expand_neighbors(fresh, settings=settings)
        fresh, graph_obs = store.expand_graph(fresh, goal=query, settings=settings)

    if state.evidence:
        state.evidence = _merge_evidence(state.evidence, fresh)
    else:
        state.evidence = list(fresh)
    state.hit_count = len(state.evidence)

    added: list[str] = []
    if node is not None:
        hit_ids = [item.chunk_id for item in fresh if getattr(item, "role", "hit") != "neighbor"]
        if not hit_ids:
            if node.chunk_ids:
                node.status = "retrieved"
            else:
                node.status = "empty"
                node.chunk_ids = []
                state.gaps.append(GapItem(topic=node.question or node.title, reason="检索未命中与该子问题相关的章节"))
        else:
            node.status = "retrieved"
            node.chunk_ids = list(dict.fromkeys(list(node.chunk_ids) + hit_ids))
            scored = [float((scores or {}).get(cid, 0.0) or 0.0) for cid in node.chunk_ids]
            scored.extend(float(item.score or 0.0) for item in fresh if getattr(item, "role", "hit") != "neighbor")
            if scored:
                node.score = max(float(node.score or 0.0), max(scored))
            wanted = set(node.chunk_ids)
            state.evidence_by_node[node.id] = [item.index for item in state.evidence if item.chunk_id in wanted]
            mapped: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for item in fresh:
                for doc, section in map_hit_to_sections(item.text, item.doc_id, state.catalog):
                    key = (doc.doc_id, section)
                    if key in seen:
                        continue
                    seen.add(key)
                    mapped.append({"doc_id": doc.doc_id, "section": section, "filename": doc.filename})
            bound = [
                item
                for item in mapped
                if question_binds_section(
                    node.question or node.title,
                    item["section"],
                    item["filename"],
                )
            ]
            chosen = bound or mapped[:3]
            if chosen:
                node.source_sections = chosen
                node.section = node.section or chosen[0]["section"]
                node.doc_id = node.doc_id or chosen[0]["doc_id"]
    elif probe:
        extra = append_probe_nodes(
            state,
            fresh,
            max_subtopics=int(getattr(settings, "agent_synthesis_max_subtopics", 6) or 6),
        )
        added = [item.id for item in extra]

    fresh_ids = {item.chunk_id for item in fresh}
    hit_rows = [
        {
            "index": item.index,
            "chunk_id": item.chunk_id,
            "doc_id": item.doc_id,
            "page": item.page,
            "snippet": item.text[:240],
            "chunk_hash": item.chunk_hash,
            "role": getattr(item, "role", "hit"),
        }
        for item in state.evidence
        if item.chunk_id in fresh_ids
    ]
    observation = {
        "query": query,
        "node_id": node.id if node is not None else None,
        "fresh_count": len(fresh),
        "count": len(state.evidence),
        "hits": hit_rows,
        "probe": probe,
        "added_nodes": added,
        **count_section_hits(
            node.section if node is not None else None,
            fresh,
            state.catalog,
        ),
    }
    observation.update(neighbor_obs)
    observation.update(graph_obs)
    return observation, False


def _evidence_from_hits(store: AgentStore, hits, *, scores: dict[str, float] | None = None) -> list[EvidenceItem]:
    if not hits:
        return []
    rows = store.load_chunks([str(hit.chunk_id) for hit in hits])
    evidence: list[EvidenceItem] = []
    for index, hit in enumerate(hits, start=1):
        row = rows.get(str(hit.chunk_id))
        text = (row.text if row is not None else getattr(hit, "snippet", "")) or ""
        chunk_hash = row.chunk_hash if row is not None else ""
        start_offset = int(row.start_offset) if row is not None else 0
        page = int(row.page) if row is not None else int(getattr(hit, "page", 1) or 1)
        evidence.append(
            EvidenceItem(
                index=index,
                chunk_id=str(hit.chunk_id),
                doc_id=str(hit.doc_id) if row is None else row.doc_id,
                page=page,
                text=text,
                chunk_hash=chunk_hash or "",
                score=float((scores or {}).get(str(hit.chunk_id), 0.0) or 0.0),
                start_offset=start_offset,
            )
        )
    return evidence


def _merge_evidence(existing: list[EvidenceItem], fresh: list[EvidenceItem]) -> list[EvidenceItem]:
    from app.agent.evidence import merge_evidence

    return merge_evidence(existing, fresh)


def _graph_verify_meta(state: AgentState) -> dict[str, Any]:
    if not state.graph_used_edges:
        return {}
    from app.verify.graph import relation_grounded_rate

    texts = {item.chunk_id: item.text for item in state.evidence}
    meta = relation_grounded_rate(state.graph_used_edges, chunk_text=texts)
    return {
        "relation_grounded_rate": meta.get("relation_grounded_rate"),
        "relation_unbacked": meta.get("relation_unbacked") or [],
    }
