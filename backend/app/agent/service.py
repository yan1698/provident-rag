"""Agent loop: Planner → Tool → Observation → audit log."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.agent.budget import next_failure_streak, past_deadline, resolve_retrieve_limit, resolve_step_limit
from app.agent.classify import LONGFORM_KINDS, classify_task_kind, has_strong_synthesis, is_point_how_why
from app.agent.guards import apply_guards
from app.agent.planner import Plan, choose_planner, next_plan
from app.agent.state import AgentPhase, AgentState
from app.agent.store import SqlAgentStore
from app.agent.tools import ToolContext, ToolRegistry, ToolResult, default_registry
from app.core.config import Settings, get_settings
from app.llm import LLMError, public_llm_failure, answer_provider_name
from app.models import AgentRun
from app.trace.service import llm_model_name

_REFUSE_ANSWER = "现有证据不足以回答该问题。"
_TERMINAL = frozenset({"finished", "refused", "error", "cancelled"})
_CANCELLED_ANSWER = "任务已取消。"


def _committed_status(db: Session, run_id: UUID) -> str | None:
    """Read the committed status. The identity map can hide another transaction's update."""
    from sqlalchemy import text

    value = db.execute(text("SELECT status FROM agent_runs WHERE id = :id"), {"id": run_id}).scalar()
    return None if value is None else str(value)


def _stop_if_closed(db: Session, row: AgentRun, run_id: UUID) -> AgentRun | None:
    if _committed_status(db, run_id) == "running":
        return None
    db.rollback()
    found = db.get(AgentRun, run_id)
    if found is not None:
        db.refresh(found)
    return found if found is not None else row


def _merge_run_metrics(row: AgentRun, status: str, previous: dict[str, Any]) -> None:
    """Keep queue wait and record how long execution took. No document text."""
    from sqlalchemy.orm.attributes import flag_modified

    metrics = dict(previous.get("_metrics") or {}) if isinstance(previous, dict) else {}
    started = row.created_at
    claimed = metrics.get("claimed_at")
    if isinstance(claimed, str):
        try:
            started = datetime.fromisoformat(claimed)
        except ValueError:
            started = row.created_at
    if started is not None:
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        metrics["run_ms"] = max(0, int((datetime.now(timezone.utc) - started).total_seconds() * 1000))
    metrics["terminal"] = status
    outline = dict(row.outline or {})
    outline["_metrics"] = metrics
    row.outline = outline
    flag_modified(row, "outline")


def _owner_key(owner_id: UUID | None) -> str:
    return "anon" if owner_id is None else str(owner_id)


def _session_key(session_id: UUID | None) -> str:
    return "-" if session_id is None else str(session_id)


def _idempotent_replay(
    db: Session,
    *,
    owner_id: UUID | None,
    session_id: UUID | None,
    idempotency_key: str,
    goal_text: str,
) -> AgentRun | None:
    """Return the run already stored for this key. None when the client may create one."""
    from datetime import timedelta

    from sqlalchemy import delete, select

    from app.models.agent import AgentIdempotency

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    owner_key = _owner_key(owner_id)
    session_key = _session_key(session_id)
    db.execute(
        delete(AgentIdempotency).where(
            AgentIdempotency.owner_key == owner_key,
            AgentIdempotency.session_key == session_key,
            AgentIdempotency.idempotency_key == idempotency_key,
            AgentIdempotency.created_at < cutoff,
        )
    )
    row = db.execute(
        select(AgentIdempotency).where(
            AgentIdempotency.owner_key == owner_key,
            AgentIdempotency.session_key == session_key,
            AgentIdempotency.idempotency_key == idempotency_key,
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.goal != goal_text:
        raise ValueError("幂等键已用于另一个问题")
    found = db.get(AgentRun, row.run_id)
    if found is None:
        return None
    found._idempotent_replay = True  # type: ignore[attr-defined]
    return found


def create_agent_run(
    db: Session,
    goal: str,
    *,
    planner: str | None = None,
    task_kind: str | None = None,
    owner_id: UUID | None = None,
    session_id: UUID | None = None,
    settings: Settings | None = None,
    scope_mode: str | None = None,
    collection_ids: list[str] | None = None,
    pin_scope: bool = False,
    clear_pin: bool = False,
    max_steps: int | None = None,
    intent: str | None = None,
    idempotency_key: str | None = None,
    answer_source: str = "document_only",
    source_run_id: UUID | None = None,
    selected_context: str = "",
    analysis_mode: str | None = None,
    target_document_id: UUID | None = None,
) -> AgentRun:
    """Create a queued row and commit. The worker claims it before execution."""
    settings = settings or get_settings()
    goal_text = goal.strip()
    if not goal_text:
        raise ValueError("goal must not be empty")
    from app.generate.general import source_mode

    if answer_source not in {"document_only", "general_explanation"}:
        raise ValueError("invalid answer_source")
    source = get_agent_run(db, source_run_id, owner_id=owner_id) if source_run_id else None
    if source_run_id and (source is None or source.user_id != owner_id or source.session_id != session_id):
        raise ValueError("source run not found")
    if source is not None and source_mode(source) != "document_only":
        raise ValueError("source run must be a document answer")
    if source is not None and source.status not in _TERMINAL:
        raise ValueError("source run must be completed")
    if selected_context and (source is None or selected_context not in (source.answer or "")):
        raise ValueError("selected_context must be selected from the source answer")
    if answer_source == "general_explanation" and source is None:
        raise ValueError("general explanation requires source_run_id")
    if source is not None and session_id is None:
        raise ValueError("source run requires an existing session")
    if source is not None and target_document_id is None and answer_source == "document_only":
        stored_target = (source.outline or {}).get("target_document_id")
        if stored_target:
            target_document_id = UUID(stored_target)
    from app.agent.memory import is_answer_pointer

    reference_context = source or (
        next(iter(reversed(_recent_session_runs(db, session_id, owner_id=owner_id))), None) if session_id else None
    )
    reference_first = bool(reference_context and is_answer_pointer(goal_text) and answer_source == "document_only")
    if reference_first and (task_kind in {None, "auto"} or analysis_mode == "auto"):
        task_kind = "factoid"
    routing = None
    if analysis_mode is not None:
        from app.agent.routing import decide, validate_scope

        routing = decide(
            goal_text,
            analysis_mode,
            answer_source=answer_source,
            target_document_id=target_document_id,
            has_answer_context=bool(reference_context),
        )
        if answer_source == "document_only":
            validate_scope(db, owner_id, scope_mode, collection_ids)
            if routing["actual"] == "quick":
                # Existing-session follow-ups retain real Agent identity and resolver.
                if not source_run_id and not reference_first:
                    raise ValueError("quick questions must use /query")
                task_kind = "factoid"
                routing.update(
                    actual="agent_factoid",
                    reason="answer_reference_first" if reference_first else "explicit_session_followup",
                )
            else:
                task_kind = routing["actual"]
    key = (idempotency_key or "").strip()[:80]
    if key:
        from app.agent.lifecycle import lock_owner

        lock_owner(db, owner_id)
        replay = _idempotent_replay(
            db,
            owner_id=owner_id,
            session_id=session_id,
            idempotency_key=key,
            goal_text=goal_text,
        )
        if replay is not None:
            stored = replay.outline or {}
            if (
                source_mode(replay) != answer_source
                or str(stored.get("source_run_id") or "") != str(source_run_id or "")
                or str(stored.get("selected_context") or "") != selected_context
                or stored.get("routing", {}).get("requested") != analysis_mode
                or str(stored.get("target_document_id") or "") != str(target_document_id or "")
            ):
                raise ValueError("幂等键已用于另一来源模式或上下文")
            return replay

    from app.agent.classify import TASK_KIND_AUTO
    from app.agent.memory import compose_session_memory
    from app.llm.ledger import attach_call_refs, bind_call_context

    priors = (
        [source]
        if source is not None
        else (_recent_session_runs(db, session_id, owner_id=owner_id) if session_id else [])
    )
    if target_document_id and (
        not priors or str((priors[-1].outline or {}).get("target_document_id") or "") != str(target_document_id)
    ):
        priors = []  # A new reading target must not reuse old answer/operation anchors.
    prior = priors[-1] if priors else None
    memory = compose_session_memory(priors) if priors else ""
    if prior is not None:
        prev_kind = str(prior.task_kind or "")
        requested = (task_kind or "").strip() or TASK_KIND_AUTO
        if (
            prev_kind in LONGFORM_KINDS
            and requested in {TASK_KIND_AUTO, ""}
            and not has_strong_synthesis(goal_text)
            and not is_point_how_why(goal_text)
        ):
            task_kind = prev_kind

    with bind_call_context(user_id=owner_id):
        resolved_kind = (
            "factoid"
            if answer_source == "general_explanation"
            else classify_task_kind(goal_text, task_kind, settings=settings)
        )
    chosen_planner = "rules" if answer_source == "general_explanation" else choose_planner(settings, planner)
    now = datetime.now(timezone.utc)
    session = _ensure_session(db, session_id, owner_id=owner_id, title=goal_text, now=now)
    if session is not None:
        from app.agent.session_runtime import acquire_session

        try:
            acquire_session(db, session.id, owner_id=owner_id)
        except Exception as exc:  # noqa: BLE001 — schema 0006 not applied yet
            if "agent_session_state" not in str(exc) and "does not exist" not in str(exc).lower():
                raise
    outline = {"memory": memory} if memory else {}
    dispatch: dict[str, Any] = {"planner": chosen_planner}
    if max_steps is not None:
        dispatch["max_steps"] = int(max_steps)
    if intent:
        dispatch["intent"] = str(intent)
    outline["_dispatch"] = dispatch
    if routing is not None:
        outline["routing"] = routing
    if target_document_id:
        outline["target_document_id"] = str(target_document_id)
    outline.update(
        answer_source=answer_source, source_run_id=str(source_run_id or ""), selected_context=selected_context
    )
    if answer_source == "general_explanation":
        outline.pop("memory", None)
        outline["topic"] = source.goal
    scope_payload = (
        dict(source.scope or {})
        if source is not None
        else _resolve_scope(
            db,
            owner_id,
            goal_text,
            session=session,
            scope_mode=scope_mode,
            collection_ids=collection_ids,
            pin_scope=pin_scope,
            clear_pin=clear_pin,
        )
    )
    if target_document_id:
        from app.agent.routing import validate_target
        from app.corpus.scope import Scope

        validate_target(db, owner_id, target_document_id, Scope.from_dict(scope_payload))
    row = AgentRun(
        user_id=owner_id,
        session_id=session.id if session is not None else None,
        goal=goal_text,
        status="queued",
        provider=answer_provider_name(settings),
        llm_model=llm_model_name(settings),
        planner=chosen_planner,
        task_kind=resolved_kind,
        steps=[],
        step_count=0,
        outline=outline,
        scope=scope_payload,
        heartbeat_at=None,
    )
    db.add(row)
    db.flush()
    if key:
        from app.models.agent import AgentIdempotency

        db.add(
            AgentIdempotency(
                owner_key=_owner_key(owner_id),
                session_key=_session_key(session_id),
                idempotency_key=key,
                run_id=row.id,
                goal=goal_text,
            )
        )
    db.commit()
    db.refresh(row)
    attach_call_refs(run_id=row.id)
    return row


def execute_agent_run(
    db: Session,
    run_id: UUID,
    *,
    max_steps: int | None = None,
    planner: str | None = None,
    intent: str | None = None,
    owner_id: UUID | None = None,
    settings: Settings | None = None,
    tools: ToolRegistry | None = None,
) -> AgentRun:
    """Run the planner loop for an existing AgentRun, committing after each step."""
    settings = settings or get_settings()
    registry = tools or default_registry()
    row = db.get(AgentRun, run_id)
    if row is None:
        raise ValueError(f"agent run {run_id} not found")
    if owner_id is not None and row.user_id != owner_id:
        raise ValueError(f"agent run {run_id} not found")
    from app.agent.lifecycle import CLOSED_STATUSES, RUNNING, heartbeat_scope

    if row.status in CLOSED_STATUSES or row.status != RUNNING:
        return row
    with heartbeat_scope(run_id, int(settings.agent_heartbeat_interval_seconds)):
        from app.generate.general import source_mode

        if source_mode(row) == "general_explanation":
            from app.agent.general import execute_explanation

            return execute_explanation(db, row, settings=settings)
        return _execute_running_run(
            db,
            row,
            run_id,
            max_steps=max_steps,
            planner=planner,
            intent=intent,
            owner_id=owner_id,
            settings=settings,
            registry=registry,
        )


def _execute_running_run(
    db: Session,
    row: AgentRun,
    run_id: UUID,
    *,
    max_steps: int | None,
    planner: str | None,
    intent: str | None,
    owner_id: UUID | None,
    settings: Settings,
    registry: ToolRegistry,
) -> AgentRun:
    goal_text = row.goal
    resolved_kind = str(row.task_kind or "factoid")
    memory_brief = ""
    if isinstance(row.outline, dict):
        memory_brief = str(row.outline.get("memory") or "")
    priors: list = []
    if row.session_id:
        priors = _recent_session_runs(db, row.session_id, owner_id=owner_id, before=row.id)
        selected_source = (row.outline or {}).get("source_run_id")
        if selected_source:
            source = get_agent_run(db, UUID(selected_source), owner_id=row.user_id)
            if source is None or source.session_id != row.session_id:
                raise ValueError("source run not found")
            priors = [source]
        target = str((row.outline or {}).get("target_document_id") or "")
        if target and (not priors or str((priors[-1].outline or {}).get("target_document_id") or "") != target):
            priors = []
            memory_brief = ""
        if not memory_brief and priors:
            from app.agent.memory import compose_session_memory

            memory_brief = compose_session_memory(priors)
    prior_answer = str(getattr(priors[-1], "answer", "") or "") if priors else ""
    anchor_run = _load_anchor_run(db, priors, session_id=row.session_id)
    outline_seed: dict[str, Any] | None = None
    if anchor_run is not None:
        anchor_outline = anchor_run.outline if isinstance(anchor_run.outline, dict) else {}
        stored = anchor_outline.get("paragraphs") if isinstance(anchor_outline, dict) else None
        outline_seed = {
            "paragraphs": list(stored) if isinstance(stored, list) else [],
            "source_answer": str(anchor_run.answer or ""),
            "anchor_run_id": str(anchor_run.id),
        }
        latest_outline = priors[-1].outline if priors and isinstance(priors[-1].outline, dict) else {}
        operation = latest_outline.get("operation_anchor") or anchor_outline.get("operation_anchor") or {}
        operation_row = _run_by_id(db, str(operation.get("run_id") or ""), session_id=row.session_id)
        if operation_row is not None and operation_row.user_id == owner_id:
            from app.agent.memory import run_operation_steps

            outline_seed["steps"] = run_operation_steps(operation_row)
            outline_seed["operation_anchor_id"] = str(operation_row.id)
        else:
            from app.agent.memory import run_operation_steps

            outline_seed["steps"] = run_operation_steps(anchor_run)
            outline_seed["operation_anchor_id"] = str(anchor_run.id) if outline_seed["steps"] else ""
        if isinstance(latest_outline, dict) and latest_outline.get("pending_paragraphs"):
            outline_seed["paragraphs"] = list(latest_outline.get("pending_paragraphs") or [])
            outline_seed["source_answer"] = str(latest_outline.get("source_answer") or outline_seed["source_answer"])
            outline_seed["pending_action"] = str(latest_outline.get("pending_action") or "")
    from app.agent.analyze import split_factoid_queries
    from app.memory.query import retrieve_slots_from_user
    from app.memory.slots import apply_ltm_to_state, resolve_turn_kind

    owner = owner_id if owner_id is not None else row.user_id
    try:
        from app.memory.store import list_items

        ltm_items = list_items(db, owner)
    except Exception as exc:  # noqa: BLE001 — schema 0006 not applied yet
        if "memory_items" not in str(exc) and "does not exist" not in str(exc).lower():
            raise
        ltm_items = []
    retrieve_slots = retrieve_slots_from_user(goal_text, questions=split_factoid_queries(goal_text))
    turn_kind = resolve_turn_kind(goal_text, memory_brief)
    ltm_extra = apply_ltm_to_state(turn_kind=turn_kind, retrieve_slots=retrieve_slots, items=ltm_items)
    from app.corpus.catalog import catalog_from_digest

    prior_catalog = catalog_from_digest(getattr(priors[-1], "catalog_digest", None) if priors else None)
    if priors:
        from app.corpus.scope import Scope

        if Scope.from_dict(getattr(priors[-1], "scope", None)).key() != Scope.from_dict(row.scope).key():
            prior_catalog = []
    limit = resolve_step_limit(
        resolved_kind,
        max_steps,
        factoid_default=settings.agent_max_steps,
        synthesis_default=settings.agent_synthesis_max_steps,
        synthesis_cap=settings.agent_synthesis_max_steps,
    )
    chosen_planner = choose_planner(settings, planner or row.planner)

    state = AgentState(
        goal=goal_text,
        task_kind=resolved_kind,
        requested_intent=intent,
        owner_id=owner_id if owner_id is not None else row.user_id,
        max_steps=limit,
        max_retrieves=resolve_retrieve_limit(
            resolved_kind,
            factoid_default=settings.agent_max_retrieves,
            synthesis_default=settings.agent_synthesis_max_retrieves,
        ),
        memory_brief=memory_brief,
        prior_answer=prior_answer,
        outline=outline_seed,
        catalog=list(prior_catalog),
        retrieve_slots=retrieve_slots,
        gap_hints=list(ltm_extra.get("gap_hints") or []),
        ltm_items=list(ltm_items),
        turn_kind=turn_kind,
        scope=dict(row.scope or {}),
    )
    from app.agent.memory import is_answer_pointer, resolve_answer_reference

    if resolved_kind == "factoid" and is_answer_pointer(goal_text):
        outline_seed = outline_seed or {}
        state.reference = resolve_answer_reference(
            goal_text, answer=prior_answer, paragraphs=outline_seed.get("paragraphs"), steps=outline_seed.get("steps")
        )
        if state.reference:
            state.reference["anchor_run_id"] = str(
                outline_seed.get("operation_anchor_id")
                if state.reference.get("kind") == "step"
                else outline_seed.get("anchor_run_id")
            )
            state.operation_steps = list(outline_seed.get("steps") or [])
            state.operation_anchor = str(outline_seed.get("operation_anchor_id") or "")
            state.outline = {**(state.outline or {}), "reference": state.reference}
            if state.reference.get("action") == "ask":
                state.turn_kind = "clarify"
    steps: list[dict[str, Any]] = []
    status = "finished"
    error_detail = ""
    usage_mark = 0

    from app.core.request_id import set_step_index
    from app.llm.ledger import bind_call_context
    from app.llm.meter import bind_llm_stop, collect_usage
    from app.queue.client import is_cancelled

    def _aborted() -> bool:
        return is_cancelled(run_id) or _committed_status(db, run_id) != "running"

    run_budget = float(getattr(settings, "agent_run_timeout_seconds", 0) or 0)
    run_deadline = time.perf_counter() + run_budget if run_budget > 0 else None
    tool_budget = float(getattr(settings, "agent_tool_timeout_seconds", 0) or 0)
    failure_streak = 0
    failure_limit = int(getattr(settings, "agent_max_consecutive_tool_errors", 3) or 0)

    try:
        with bind_call_context(user_id=state.owner_id, run_id=run_id):
            from app.agent.run_budget import bind as bind_run_budget

            with (
                bind_llm_stop(_aborted),
                bind_run_budget(state, settings, checkpoint=lambda: _persist_progress(row, steps, state)),
            ):
                for _ in range(limit):
                    closed = _stop_if_closed(db, row, run_id)
                    if closed is not None:
                        return closed
                    if is_cancelled(run_id):
                        status = "cancelled"
                        if state.reading.get("mode") == "bounded":
                            state.reading["status"] = "cancelled"
                        if not state.answer:
                            state.answer = _CANCELLED_ANSWER
                        break
                    if past_deadline(run_deadline, time.perf_counter()):
                        status = "error"
                        if state.reading.get("mode") == "bounded":
                            state.reading["status"] = "timeout"
                        error_detail = "run timeout"
                        break
                    from app.corpus.scope import Scope

                    store = SqlAgentStore(db, owner_id=state.owner_id)
                    store.scope = Scope.from_dict(state.scope)
                    store.target_document_id = str((row.outline or {}).get("target_document_id") or "")
                    tool_deadline = time.perf_counter() + tool_budget if tool_budget > 0 else None
                    ctx = ToolContext(
                        state=state,
                        settings=settings,
                        store=store,
                        deadline=tool_deadline,
                    )
                    started = time.perf_counter()
                    with collect_usage(state.usage):
                        plan = next_plan(state.working_goal(), steps, settings, planner=chosen_planner, state=state)
                        plan = apply_guards(plan, state, steps, registry=registry)
                        result = _execute(plan, ctx, registry)
                    observation, terminal = result.observation, result.terminal
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    state.total_steps += 1
                    set_step_index(state.total_steps)
                    step = {
                        "thought": plan.thought,
                        "action": plan.action,
                        "input": plan.input,
                        "source": plan.source,
                        "phase": state.phase.value,
                        "observation": observation,
                        "elapsed_ms": elapsed_ms,
                    }
                    if plan.node_id:
                        step["node_id"] = plan.node_id
                    if state.usage:
                        step["usage"] = [item.to_dict() for item in state.usage[usage_mark:]]
                        usage_mark = len(state.usage)
                    steps.append(step)
                    if not _persist_progress(row, steps, state):
                        closed = _stop_if_closed(db, row, run_id)
                        return closed if closed is not None else row
                    failure_streak, tripped = next_failure_streak(
                        failure_streak,
                        observation if isinstance(observation, dict) else None,
                        limit=failure_limit,
                    )
                    if tripped:
                        status = "error"
                        error_detail = "连续工具失败"
                        break
                    if terminal:
                        obs_status = str(observation.get("status") or "") if isinstance(observation, dict) else ""
                        if plan.action == "refuse" or obs_status == "refused":
                            status = "refused"
                        break
                else:
                    if not state.answer:
                        status = "error"
                        error_detail = "max_steps reached without an answer"
                    else:
                        status = "finished"
    except LLMError as exc:
        if exc.kind == "cancelled" or is_cancelled(run_id):
            status = "cancelled"
            if state.reading.get("mode") == "bounded":
                state.reading["status"] = "cancelled"
            if not state.answer:
                state.answer = _CANCELLED_ANSWER
            error_detail = ""
            thought = "任务已取消"
            action = "cancel"
            observation = {"status": "cancelled"}
        else:
            status = "error"
            error_detail = public_llm_failure(exc)
            thought = "LLM unavailable"
            action = "error"
            observation = {"error": error_detail, "error_kind": exc.kind}
        state.phase = AgentPhase.DONE
        steps.append(
            {
                "thought": thought,
                "action": action,
                "input": "",
                "source": "system",
                "phase": state.phase.value,
                "observation": observation,
                "elapsed_ms": 0,
            }
        )
        if not _persist_progress(row, steps, state):
            closed = _stop_if_closed(db, row, run_id)
            return closed if closed is not None else row
    except Exception as exc:  # noqa: BLE001 — surface unexpected failures on the run row
        status = "error"
        error_detail = str(exc)
        state.phase = AgentPhase.DONE
        steps.append(
            {
                "thought": "agent crashed",
                "action": "error",
                "input": "",
                "source": "system",
                "phase": state.phase.value,
                "observation": {"error": error_detail, "error_kind": "internal"},
                "elapsed_ms": 0,
            }
        )
        if not _persist_progress(row, steps, state):
            closed = _stop_if_closed(db, row, run_id)
            return closed if closed is not None else row

    if not state.answer and status == "finished":
        state.answer = _REFUSE_ANSWER
        status = "refused"

    from sqlalchemy import select

    from app.agent.lifecycle import WORKER_ID
    from app.queue.client import writer_owns

    if not writer_owns(run_id, WORKER_ID):
        db.rollback()
        found = db.get(AgentRun, run_id)
        return found if found is not None else row

    held = db.execute(
        select(AgentRun).where(AgentRun.id == run_id, AgentRun.status == "running").with_for_update()
    ).scalar_one_or_none()
    if held is None:
        db.rollback()
        found = db.get(AgentRun, run_id)
        return found if found is not None else row
    row = held

    if state.task_spec is not None or state.fallback_reason or state.partial or state.memory_brief:
        outline = dict(state.outline or {})
        outline["reading"] = state.reading
        outline["run_budget"] = state.run_budget
        if state.memory_brief:
            outline.setdefault("memory", state.memory_brief)
        if state.task_spec is not None:
            outline.setdefault("task_spec", state.task_spec.to_dict())
        if state.compose_mode:
            outline.setdefault("compose_mode", state.compose_mode)
        if state.fallback_reason:
            outline.setdefault("fallback_reason", state.fallback_reason)
        if state.partial:
            outline["partial"] = True
        state.outline = outline

    row.status = status
    row.answer = state.answer
    row.planner = chosen_planner
    row.task_kind = state.task_kind
    _seal_answer_anchor(row, state)
    previous_outline = dict(row.outline or {})
    row.outline = state.outline
    row.outline = {
        **(row.outline or {}),
        **{
            k: previous_outline[k]
            for k in ("answer_source", "source_run_id", "selected_context", "routing", "target_document_id")
            if k in previous_outline
        },
    }
    _merge_run_metrics(row, status, previous_outline)
    row.gaps = [{"topic": gap.topic, "reason": gap.reason} for gap in state.gaps] or None
    from app.corpus.catalog import catalog_digest_payload

    row.catalog_digest = catalog_digest_payload(state.catalog) or None
    row.scope = dict(state.scope or {}) or None
    row.step_count = len(steps)
    row.steps = list(steps)
    flag_modified(row, "steps")
    row.verification = state.verification
    row.heartbeat_at = datetime.now(timezone.utc)
    row.finished_at = datetime.now(timezone.utc)
    if error_detail and status == "error" and not row.answer:
        row.answer = error_detail
    if status == "cancelled":
        row.rolled_back = True
    if row.session_id is not None:
        from app.agent.session_runtime import complete_session_turn, record_turn_events
        from app.memory.extractor import events_from_steps

        try:
            snap = complete_session_turn(
                db,
                row.session_id,
                cancelled=(status == "cancelled"),
                slots=dict(state.retrieve_slots or {}),
                catalog_digest=list(row.catalog_digest or []),
                gaps=list(row.gaps or []),
            )
            record_turn_events(
                db,
                row.session_id,
                row.id,
                events_from_steps(str(row.id), steps),
                revision=int(snap.get("revision") or 0),
            )
        except Exception as exc:  # noqa: BLE001 — schema 0006 not applied yet
            if "agent_session" not in str(exc) and "does not exist" not in str(exc).lower():
                raise
    try:
        from app.memory.store import persist_turn_memory

        persist_turn_memory(
            db,
            run=row,
            steps=steps,
            retrieve_obs=state.last_retrieve_obs or None,
            owner_id=state.owner_id,
            session_id=row.session_id,
            answered=status == "finished" and int((state.last_retrieve_obs or {}).get("count") or 0) > 0,
        )
    except Exception as exc:  # noqa: BLE001 — schema 0006 not applied yet
        if "memory_items" not in str(exc) and "does not exist" not in str(exc).lower():
            raise
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def run_agent(
    db: Session,
    goal: str,
    *,
    max_steps: int | None = None,
    planner: str | None = None,
    task_kind: str | None = None,
    intent: str | None = None,
    owner_id: UUID | None = None,
    session_id: UUID | None = None,
    settings: Settings | None = None,
    tools: ToolRegistry | None = None,
) -> AgentRun:
    """Synchronous create + execute (unit tests / offline scripts)."""
    settings = settings or get_settings()
    row = create_agent_run(
        db,
        goal,
        planner=planner,
        task_kind=task_kind,
        owner_id=owner_id,
        session_id=session_id,
        settings=settings,
    )
    from app.agent.lifecycle import claim_queued_run

    if not claim_queued_run(row.id):
        db.refresh(row)
        return row
    db.refresh(row)
    return execute_agent_run(
        db,
        row.id,
        max_steps=max_steps,
        planner=planner,
        intent=intent,
        owner_id=owner_id,
        settings=settings,
        tools=tools,
    )


def _resolve_scope(db, owner_id, goal, *, session, scope_mode, collection_ids, pin_scope, clear_pin) -> dict:
    pinned = None
    try:
        if session is not None and clear_pin:
            session.pinned_scope = None
        elif session is not None:
            pinned = session.pinned_scope if isinstance(session.pinned_scope, dict) else None
        from app.corpus.router import resolve_scope

        scope = resolve_scope(
            db,
            owner_id,
            goal,
            mode=scope_mode or "auto",
            collection_ids=collection_ids,
            pinned=pinned,
        )
        if session is not None and pin_scope and scope.mode in {"soft", "hard"}:
            session.pinned_scope = scope.to_dict()
        return scope.to_dict()
    except Exception as exc:  # noqa: BLE001 — collections table may be absent
        if "collections" not in str(exc) and "does not exist" not in str(exc).lower():
            raise
        return {"mode": "all", "collection_ids": [], "labels": [], "widened": False}


def _ensure_session(
    db: Session,
    session_id: UUID | None,
    *,
    owner_id: UUID | None,
    title: str,
    now: datetime,
):
    from app.models.agent import AgentSession

    if session_id is not None:
        session = db.get(AgentSession, session_id)
        if session is None:
            raise ValueError(f"agent session {session_id} not found")
        if owner_id is not None and session.user_id != owner_id:
            raise ValueError(f"agent session {session_id} not found")
        session.updated_at = now
        db.add(session)
        return session
    session = AgentSession(user_id=owner_id, title=(title or "")[:160], created_at=now, updated_at=now)
    db.add(session)
    db.flush()
    return session


def _seal_answer_anchor(row: AgentRun, state: AgentState) -> None:
    outline = dict(state.outline or {})
    operation = outline.get("operation_anchor")
    if isinstance(operation, dict) and operation.get("fresh"):
        outline["operation_anchor"] = {"run_id": str(row.id)}
    anchor = outline.get("answer_anchor")
    if isinstance(anchor, dict) and anchor.get("fresh"):
        outline["answer_anchor"] = {"run_id": str(row.id)}
    elif state.compose_mode in {"written", "catalog"} and not (isinstance(anchor, dict) and anchor.get("run_id")):
        outline["answer_anchor"] = {"run_id": str(row.id)}
    state.outline = outline


def _run_by_id(db: Session, raw: str, *, session_id: UUID | None = None) -> AgentRun | None:
    try:
        run_id = UUID(str(raw))
    except (TypeError, ValueError):
        return None
    row = db.get(AgentRun, run_id)
    if row is None:
        return None
    # A one-shot run has no session. Reject only when this turn is in a session
    # and the loaded row belongs to another one.
    if session_id is not None and row.session_id != session_id:
        return None
    return row


def _load_anchor_run(db: Session, priors: list, *, session_id: UUID | None = None) -> AgentRun | None:
    """Substantive answer a later paragraph pointer should use."""
    from app.agent.memory import select_answer_anchor

    if not priors:
        return None
    latest = priors[-1]
    outline = latest.outline if isinstance(latest.outline, dict) else {}
    anchor = outline.get("answer_anchor") if isinstance(outline, dict) else None
    if isinstance(anchor, dict) and anchor.get("run_id") and not anchor.get("fresh"):
        target = str(anchor["run_id"])
        for row in priors:
            if str(row.id) == target:
                return row
        loaded = _run_by_id(db, target, session_id=session_id)
        if loaded is not None:
            return loaded
    picked = select_answer_anchor(priors)
    target = str(picked.get("run_id") or "")
    for row in priors:
        if str(row.id) == target:
            return row
    return latest


def _recent_session_runs(
    db: Session,
    session_id: UUID | None,
    *,
    owner_id: UUID | None = None,
    before: UUID | None = None,
    limit: int = 3,
) -> list:
    from sqlalchemy import select, func

    if session_id is None:
        return []
    stmt = (
        select(AgentRun)
        .where(AgentRun.session_id == session_id)
        .where(AgentRun.status.in_(tuple(_TERMINAL)))
        .where(func.coalesce(AgentRun.outline["answer_source"].as_string(), "document_only") != "general_explanation")
        .order_by(AgentRun.created_at.desc())
    )
    if owner_id is not None:
        stmt = stmt.where(AgentRun.user_id == owner_id)
    if before is not None:
        stmt = stmt.where(AgentRun.id != before)
    rows = list(db.execute(stmt.limit(max(int(limit), 1))).scalars().all())
    rows.reverse()
    return rows


def list_session_runs(
    db: Session,
    session_id: UUID,
    *,
    owner_id: UUID | None = None,
) -> list:
    from sqlalchemy import select

    from app.models.agent import AgentSession

    session = db.get(AgentSession, session_id)
    if session is None:
        return []
    if owner_id is not None and session.user_id != owner_id:
        return []
    stmt = select(AgentRun).where(AgentRun.session_id == session_id).order_by(AgentRun.created_at.asc())
    if owner_id is not None:
        stmt = stmt.where(AgentRun.user_id == owner_id)
    return list(db.execute(stmt).scalars().all())


def list_agent_sessions(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    owner_id: UUID | None = None,
) -> list[dict]:
    from sqlalchemy import select

    from app.models.agent import AgentSession

    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    stmt = select(AgentSession).order_by(AgentSession.updated_at.desc()).offset(offset).limit(limit)
    if owner_id is not None:
        stmt = stmt.where(AgentSession.user_id == owner_id)
    sessions = list(db.execute(stmt).scalars().all())
    out: list[dict] = []
    for session in sessions:
        runs = list_session_runs(db, session.id, owner_id=owner_id)
        if not runs:
            continue
        first = runs[0]
        last = runs[-1]
        out.append(
            {
                "id": session.id,
                "title": (session.title or first.goal or "").strip() or first.goal,
                "latest_goal": last.goal or "",
                "answer_source": "general_explanation"
                if (last.outline or {}).get("answer_source") == "general_explanation"
                else "document_only",
                "turn_count": len(runs),
                "status": last.status,
                "task_kind": str(getattr(last, "task_kind", None) or "factoid"),
                "gap_count": len(getattr(last, "gaps", None) or []),
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }
        )
    return out


def get_session(db: Session, session_id: UUID, *, owner_id: UUID | None = None):
    from app.models.agent import AgentSession

    session = db.get(AgentSession, session_id)
    if session is None:
        return None
    if owner_id is not None and session.user_id != owner_id:
        return None
    return session


def get_agent_run(db: Session, run_id: UUID, *, owner_id: UUID | None = None) -> AgentRun | None:
    row = db.get(AgentRun, run_id)
    if row is None:
        return None
    if owner_id is not None and row.user_id != owner_id:
        return None
    return row


def delete_agent_run(db: Session, run_id: UUID, *, owner_id: UUID | None = None) -> AgentRun | None:
    """Delete an agent run and any feedback rows pointing at it."""
    from types import SimpleNamespace

    from sqlalchemy import delete

    from app.models.feedback import Feedback

    row = get_agent_run(db, run_id, owner_id=owner_id)
    if row is None:
        return None
    snapshot = SimpleNamespace(id=row.id, goal=row.goal)
    db.execute(delete(Feedback).where(Feedback.agent_run_id == run_id))
    db.delete(row)
    db.commit()
    return snapshot


def delete_agent_session(db: Session, session_id: UUID, *, owner_id: UUID | None = None):
    """Delete a conversation and every run inside it."""
    from types import SimpleNamespace

    from sqlalchemy import delete

    from app.models.feedback import Feedback

    session = get_session(db, session_id, owner_id=owner_id)
    if session is None:
        return None
    runs = list_session_runs(db, session_id, owner_id=owner_id)
    run_ids = [row.id for row in runs]
    snapshot = SimpleNamespace(id=session.id, title=session.title or "", turn_count=len(runs))
    if run_ids:
        db.execute(delete(Feedback).where(Feedback.agent_run_id.in_(run_ids)))
        db.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
    db.delete(session)
    db.commit()
    return snapshot


def list_agent_runs(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    owner_id: UUID | None = None,
) -> list[AgentRun]:
    from sqlalchemy import select

    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    stmt = select(AgentRun).order_by(AgentRun.created_at.desc()).offset(offset).limit(limit)
    if owner_id is not None:
        stmt = stmt.where(AgentRun.user_id == owner_id)
    return list(db.execute(stmt).scalars().all())


def count_running_agent_runs(
    db: Session,
    *,
    owner_id: UUID | None,
    stale_seconds: int = 1800,
) -> int:
    """Queued jobs always count. Running jobs count only while the heartbeat is fresh."""
    from datetime import timedelta

    from sqlalchemy import func, or_, select

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(int(stale_seconds), 1))
    fresh_running = (
        (AgentRun.status == "running") & (AgentRun.heartbeat_at.is_not(None)) & (AgentRun.heartbeat_at >= cutoff)
    )
    stmt = select(func.count()).select_from(AgentRun).where(or_(AgentRun.status == "queued", fresh_running))
    if owner_id is not None:
        stmt = stmt.where(AgentRun.user_id == owner_id)
    else:
        stmt = stmt.where(AgentRun.user_id.is_(None))
    return int(db.execute(stmt).scalar_one() or 0)


def _persist_progress(row: AgentRun, steps: list[dict[str, Any]], state: AgentState | None = None) -> bool:
    """Commit steps only while this worker still holds a running lease.

    The UPDATE requires ``status='running'``. A missing worker lease, or an
    owner key that names someone else, returns before that UPDATE.
    """
    from sqlalchemy import update

    from app.agent.lifecycle import RUNNING, WORKER_ID
    from app.queue.client import writer_owns

    db = Session.object_session(row)
    if db is None:
        return False
    run_id = row.id
    if not writer_owns(run_id, WORKER_ID):
        return False
    values: dict[str, Any] = {
        "steps": list(steps),
        "step_count": len(steps),
        "heartbeat_at": datetime.now(timezone.utc),
    }
    if state is not None:
        values["task_kind"] = state.task_kind
        if state.answer:
            values["answer"] = state.answer
        if state.verification:
            values["verification"] = state.verification
        outline = dict(state.outline or {})
        outline["reading"] = state.reading
        outline["run_budget"] = state.run_budget
        if state.memory_brief:
            outline["memory"] = state.memory_brief
        if state.task_spec is not None:
            outline["task_spec"] = state.task_spec.to_dict()
        if state.compose_mode:
            outline["compose_mode"] = state.compose_mode
        if state.fallback_reason:
            outline["fallback_reason"] = state.fallback_reason
        for key in ("answer_source", "source_run_id", "selected_context", "routing", "target_document_id"):
            if key in (row.outline or {}):
                outline[key] = row.outline[key]
        if outline:
            values["outline"] = outline
        if state.catalog:
            from app.corpus.catalog import catalog_digest_payload

            values["catalog_digest"] = catalog_digest_payload(state.catalog)
        if state.gaps:
            values["gaps"] = [{"topic": gap.topic, "reason": gap.reason} for gap in state.gaps]
    result = db.execute(
        update(AgentRun)
        .where(AgentRun.id == run_id, AgentRun.status == RUNNING)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    if int(result.rowcount or 0) != 1:
        db.rollback()
        return False
    db.commit()
    db.refresh(row)
    from app.core.log import get_logger

    last = steps[-1] if steps else {}
    get_logger("agent").info(
        "step %s %s",
        last.get("action") or "",
        last.get("phase") or "",
    )
    return True


def _execute(plan: Plan, ctx: ToolContext, registry: ToolRegistry) -> ToolResult:
    if past_deadline(ctx.deadline, time.perf_counter()):
        return ToolResult({"error": "tool deadline exceeded", "error_kind": "timeout"})
    if plan.action == "error":
        return ToolResult(
            {"error": f"unsupported action {plan.input or plan.action}", "error_kind": "invalid_action", "guard": "G6"}
        )
    tool = registry.get(plan.action)
    if tool is None:
        return ToolResult({"error": f"unsupported action {plan.action}", "error_kind": "invalid_action", "guard": "G6"})
    return tool.execute(plan, ctx)
