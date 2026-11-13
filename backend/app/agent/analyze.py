"""Turn a user goal into a TaskSpec. Rules first; optional LLM polish."""

from __future__ import annotations

import re
from typing import Any

from app.agent.classify import COMPARE_DIFFERENT_RE, HOW_WHY_RE, has_strong_synthesis, is_point_how_why
from app.agent.state import AgentState, TaskSpec

INTENTS = ("explain", "compare", "plan", "inventory")

_COMPARE = ("对比", "比较", "异同", "区别", "差异", "对照", "相比")
_PLAN = ("规划", "路径", "步骤", "方案", "制定", "指南", "教程")
_PLAN_DELIVERABLE = ("制定", "指南", "教程", "方案", "路径", "步骤")
_INVENTORY = ("清单", "盘点", "覆盖", "汇总")
_EXPLAIN_HINTS = ("梳理", "整理", "要点", "说明", "解释")

_SPLIT = re.compile(r"与|和|、|/|，|,")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_RESPECTIVE_RE = re.compile(r"分别|各自")
_ASK_TAIL_RE = re.compile(r"(是什么|有哪些|如何定义|的定义|怎么定义).*$")

ANALYZE_SYSTEM = (
    "你是知识库任务分析器。只根据用户目标判断任务形态。"
    "只输出一个 JSON 对象，不要 Markdown。"
    '形状：{"intent":"explain|compare|plan|inventory","subjects":["主题"],'
    '"questions":["检索子问题"]}。'
    "compare 的 questions 只能有两条：每侧各自「是什么、有哪些要点」。"
    "不要单独出「异同/区别」检索题，对照在写稿阶段完成。"
    "subjects 必须来自用户目标里的主题词（如 agent、学习、发票报销），"
    "不要把规划/指南/制定当主题，也不要发明用户没点名的领域（前端、设计）。"
    "questions 会直接当作检索句。每条必须含 subjects 里的主题词，不要改写成空泛的「目标与适用范围」。"
    "若 subjects 已从跑题原句修正（例如原句在问前端设计，主题实为编排范式），"
    "questions 必须覆盖修正后的主题，禁止再把跑题原句当下检索词。"
    "用户在问整库覆盖面、有哪些文档、盘点知识库且没有点名领域时，intent=inventory，subjects 必须为空，"
    "禁止发明 Agent、前端、设计等主题。"
    "不要扩成质量保障、验证、技术决策、核心逻辑等相邻章。"
)

_GENERIC_SUBJECTS = frozenset({"知识库", "覆盖面", "文档", "语料", "内容", "材料", "章节", "文件"})
_CORPUS_ASK_RE = re.compile(r"覆盖面|整个知识库|全部文档|所有文档|知识库(里)?有哪些|盘点知识库|有哪些文档")


def detect_intent(goal: str, requested: str | None = None) -> str:
    raw = (requested or "").strip().lower()
    if raw in INTENTS:
        return raw
    text = goal or ""
    if any(word in text for word in _COMPARE) or COMPARE_DIFFERENT_RE.search(text):
        return "compare"
    if any(word in text for word in _PLAN):
        if is_point_how_why(text) and not any(word in text for word in _PLAN_DELIVERABLE):
            return "explain"
        return "plan"
    if any(word in text for word in _INVENTORY) or _CORPUS_ASK_RE.search(text):
        return "inventory"
    return "explain"


def extract_subjects(goal: str, *, limit: int = 4) -> list[str]:
    from app.corpus.relevance import strip_intent

    text = strip_intent(goal or "")
    parts = [piece.strip(" 。？?！!的并") for piece in _SPLIT.split(text) if piece.strip()]
    subjects: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        cleaned = re.sub(r"\s+", "", raw)
        if len(cleaned) < 2 or cleaned in seen:
            return
        seen.add(cleaned)
        subjects.append(cleaned)

    for part in parts:
        add(part)
        if len(subjects) >= limit:
            break
    if len(subjects) < limit:
        for token in _LATIN.findall(goal or ""):
            add(token)
            if len(subjects) >= limit:
                break
    kept: list[str] = []
    for item in subjects:
        if any(item != other and item.casefold() in other.casefold() for other in subjects):
            continue
        kept.append(item)
    return (kept or subjects)[:limit]


def domain_subjects(goal: str, subjects: list[str] | None) -> list[str]:
    """Drop generic fillers. Inventory of the whole corpus has an empty list."""
    out: list[str] = []
    for item in subjects or []:
        name = re.sub(r"\s+", "", str(item or ""))
        if len(name) < 2 or name in _GENERIC_SUBJECTS:
            continue
        out.append(name)
    return out


def is_catalog_inventory(intent: str, subjects: list[str] | None, goal: str = "") -> bool:
    """True when the task is a corpus listing, not a themed retrieve."""
    if is_document_outline(goal):
        return True
    if (intent or "") != "inventory":
        return bool(_CORPUS_ASK_RE.search(goal or "")) and not domain_subjects(goal, subjects)
    return not domain_subjects(goal, subjects)


def is_document_outline(goal: str) -> bool:
    return bool(
        re.search(
            r"章节.*(?:目录|概览|大纲)|(?:目录|大纲).*(?:章节|层级)|(?:chapter|section).*(?:overview|outline|contents)",
            goal or "",
            re.I,
        )
    )


def split_factoid_queries(goal: str, *, limit: int = 4) -> list[str]:
    """Split 「A 和 B 分别是什么」into per-subject retrieval queries.

    Only 分别/各自 triggers a split. Ordinary 「A 和 B」 stays one query.
    """
    from app.corpus.relevance import strip_chat

    raw = (goal or "").strip()
    cleaned = strip_chat(raw) or raw
    if not _RESPECTIVE_RE.search(raw):
        return [(cleaned or raw)[:160]]
    body = _RESPECTIVE_RE.sub(" ", cleaned)
    body = _ASK_TAIL_RE.sub("", body).strip(" 。？?！!，,、的")
    ask = "有哪些" if "有哪些" in raw else "是什么"
    parts = [piece.strip(" 。？?！!的并") for piece in _SPLIT.split(body) if piece.strip()]
    parts = [item for item in parts if len(item) >= 1]
    if len(parts) < 2:
        return [(cleaned or raw)[:160]]
    return [f"{part} {ask}"[:160] for part in parts[: max(2, int(limit))]]


_CONTRAST_ONLY = ("异同", "区别", "差异", "对照", "相比")
_DEFINITION_MARK = ("是什么", "定义", "要点", "如何", "哪些", "步骤", "约束")
_MAX_ACCEPTED = 6


def accept_questions(
    intent: str,
    goal: str,
    subjects: list[str] | None,
    raw: Any,
) -> list[str] | None:
    """Keep model questions only when they still name the resolved subjects.

    ``None`` means unusable: the caller falls back to ``build_questions``.
    An empty list is valid for a whole-corpus inventory.
    """
    cleaned = _clean_questions(raw)
    scoped = domain_subjects(goal, subjects)
    if intent == "inventory" and not scoped:
        return [] if not cleaned else None
    if not cleaned:
        return None
    if intent == "compare":
        return _accept_compare(cleaned, scoped)
    if scoped:
        if any(not any(name in item for name in scoped) for item in cleaned):
            return None
        cap = _MAX_ACCEPTED if intent == "plan" else 4
        return cleaned[:cap]
    return cleaned[:2]


def _clean_questions(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        cleaned.append(text[:160])
        if len(cleaned) >= _MAX_ACCEPTED:
            break
    return cleaned


def _accept_compare(cleaned: list[str], scoped: list[str]) -> list[str] | None:
    if len(scoped) < 2:
        return None
    kept: list[str] = []
    for item in cleaned:
        contrast = any(word in item for word in _CONTRAST_ONLY)
        defined = any(word in item for word in _DEFINITION_MARK)
        if contrast and not defined:
            continue
        if not any(name in item for name in scoped[:2]):
            continue
        kept.append(item)
    if len(kept) < 2:
        return None
    blob = " ".join(kept[:2])
    if scoped[0] not in blob or scoped[1] not in blob:
        return None
    return kept[:2]


def build_questions(intent: str, goal: str, subjects: list[str]) -> list[str]:
    goal = (goal or "").strip()
    if intent == "compare" and len(subjects) >= 2:
        left, right = subjects[0], subjects[1]
        return [
            f"{left}在知识库中如何定义、有哪些要点",
            f"{right}在知识库中如何定义、有哪些要点",
        ]
    if intent == "plan":
        topic = "、".join(subjects) if subjects else goal
        return [
            f"{topic}包含哪些概念与步骤",
            f"{topic}应如何分步掌握或实施",
            f"{topic}有哪些约束或知识库未写明的部分",
        ]
    if intent == "inventory":
        scoped = domain_subjects(goal, subjects)
        if not scoped:
            return []
        topic = "、".join(scoped)
        return [
            f"知识库已经覆盖{topic}的哪些部分",
            f"知识库尚未覆盖{topic}的哪些诉求",
        ]
    rewritten = [item for item in subjects if item and item not in goal]
    if rewritten:
        return [f"说明{item}" for item in subjects[:2]]
    if HOW_WHY_RE.search(goal) and not has_strong_synthesis(goal):
        return [(goal or "").strip()[:160] or "说明当前问题"]
    questions = [f"说明{item}" for item in subjects[:2]]
    return questions[:2] or [goal or "整理当前知识库中的相关要点"]


def _deliverable(intent: str) -> str:
    return {
        "explain": "紧扣用户问题的说明；只列实际未覆盖的用户要求，不虚构相邻章节",
        "compare": "按维度对比的长回答",
        "plan": "有步骤的实施说明",
        "inventory": "已覆盖 / 未覆盖盘点",
    }[intent]


def analyze_task(
    goal: str,
    requested_intent: str | None = None,
    *,
    settings: Any | None = None,
) -> TaskSpec:
    if is_document_outline(goal):
        return TaskSpec(
            intent="inventory",
            subjects=domain_subjects(goal, extract_subjects(goal)),
            questions=[],
            deliverable="按层级列出文档章节目录",
            source="rules",
        )
    intent = detect_intent(goal, requested_intent)
    rule_subjects = domain_subjects(goal, extract_subjects(goal))
    subjects = list(rule_subjects)
    source = "rules"
    offered: list[str] | None = None
    if settings is not None and not (requested_intent or "").strip():
        from app.llm import router_llm_available

        if router_llm_available(settings):
            polished = _llm_analyze(goal, settings)
            if polished is not None:
                intent = polished.intent
                if polished.subjects:
                    subjects = domain_subjects(goal, polished.subjects)
                offered = list(polished.questions)
                source = "llm"
    if not rule_subjects and (
        detect_intent(goal, requested_intent) == "inventory" or _CORPUS_ASK_RE.search(goal or "")
    ):
        intent = "inventory"
        subjects = []
    questions = accept_questions(intent, goal, subjects, offered) if offered is not None else None
    if questions is None:
        questions = build_questions(intent, goal, subjects)
        if source == "llm" and offered is not None:
            source = "rules"
    return TaskSpec(
        intent=intent,
        subjects=subjects,
        questions=questions,
        deliverable=_deliverable(intent),
        source=source,
        turn_kind="retrieve",
    )


def apply_analyze(
    state: AgentState,
    *,
    requested_intent: str | None = None,
    settings: Any | None = None,
) -> dict[str, Any]:
    from app.agent.state import AgentPhase

    seeded = state.outline if isinstance(state.outline, dict) else {}
    from app.agent.memory import resolve_answer_reference

    reference = resolve_answer_reference(
        state.goal,
        brief=state.memory_brief,
        answer=str(seeded.get("source_answer") or state.prior_answer or ""),
        paragraphs=list(seeded.get("paragraphs") or []) or None,
        pending_action=str(seeded.get("pending_action") or ""),
        steps=list(seeded.get("steps") or state.operation_steps),
    )
    state.operation_steps = list(seeded.get("steps") or state.operation_steps)
    operation_anchor = seeded.get("operation_anchor_id") or seeded.get("operation_anchor")
    if isinstance(operation_anchor, dict):
        operation_anchor = operation_anchor.get("run_id")
    state.operation_anchor = str(operation_anchor or state.operation_anchor)
    if reference:
        reference["anchor_run_id"] = str(seeded.get("anchor_run_id") or "")
    if reference and reference.get("action") != "ask" and reference.get("chunk_ids"):
        reference["retrieve"] = True
    if reference is not None and not reference.get("retrieve"):
        spec = TaskSpec(
            intent="explain",
            subjects=[],
            questions=[],
            deliverable=_deliverable("explain"),
            source="rules",
            turn_kind="clarify",
            update="clarify" if reference.get("action") == "ask" else "explain",
        )
        state.task_spec = spec
        state.turn_kind = "clarify"
        state.retrieve_slots = {}
        state.reference = reference
        state.outline = {
            "reference": reference,
            "anchor_run_id": str(seeded.get("anchor_run_id") or ""),
        }
        state.phase = AgentPhase.ANALYZING
        return spec.to_dict()

    if reference is not None and reference.get("retrieve"):
        from app.agent.memory import parse_subjects

        spec = TaskSpec(
            intent="explain", subjects=parse_subjects(state.memory_brief), source="rules", update="continue"
        )
    else:
        spec = analyze_task(
            state.goal,
            requested_intent or state.requested_intent,
            settings=settings,
        )
    from app.agent.memory import classify_turn_kind, merge_followup_spec

    turn_kind = classify_turn_kind(state.goal, state.memory_brief)
    if state.memory_brief:
        merged = merge_followup_spec(
            spec.to_dict(),
            user_goal=state.goal,
            brief=state.memory_brief,
        )
        spec = TaskSpec(
            intent=str(merged.get("intent") or spec.intent),
            subjects=list(merged.get("subjects") or spec.subjects),
            questions=list(merged["questions"]) if "questions" in merged else list(spec.questions),
            deliverable=spec.deliverable,
            source=spec.source,
            turn_kind=str(merged.get("turn_kind") or turn_kind),
            update=str(merged.get("update") or "new"),
        )
        turn_kind = spec.turn_kind
    if reference is not None and reference.get("retrieve"):
        state.reference = reference
        spec.update = "continue"
        turn_kind = "retrieve"
        spec.questions = [f"{state.goal}；指定操作步骤原文：{reference.get('text', '')}"[:600]]
    elif spec.update == "new":
        state.operation_steps = []
        state.operation_anchor = ""
    if is_document_outline(state.goal):
        spec.intent = "inventory"
    # New tasks must resurvey the live, owner/scoped corpus rather than reuse an old catalog.
    if spec.update == "new" and state.memory_brief:
        state.catalog = []
    state.nodes = []
    state.evidence = []
    state.evidence_by_node = {}
    state.digests = []
    spec.turn_kind = turn_kind
    state.task_spec = spec
    state.turn_kind = turn_kind
    state.phase = AgentPhase.ANALYZING
    if spec.subjects or spec.questions or state.retrieve_slots:
        from app.memory.query import retrieve_slots_from_user
        from app.memory.slots import apply_ltm_to_state

        slots = retrieve_slots_from_user(
            state.goal,
            subjects=list(spec.subjects),
            questions=list(spec.questions),
        )
        extra = apply_ltm_to_state(
            turn_kind=turn_kind,
            retrieve_slots=slots,
            items=list(state.ltm_items or []),
        )
        state.retrieve_slots = slots
        state.gap_hints = list(extra.get("gap_hints") or [])
    return spec.to_dict()


def _llm_analyze(goal: str, settings: Any) -> TaskSpec | None:
    from app.llm import LLMError, get_router_llm_client

    try:
        client = get_router_llm_client(settings)
        raw = client.generate(
            system=ANALYZE_SYSTEM,
            prompt=f"用户目标：{goal}\n请输出 JSON。",
            max_output_tokens=min(400, int(getattr(settings, "llm_max_output_tokens", 1024) or 1024)),
            temperature=0.0,
            purpose="analyze",
        ).text
    except (LLMError, ValueError, TypeError):
        return None
    data = _parse_analyze_json(raw)
    if data is None:
        return None
    intent = str(data.get("intent") or "").strip().lower()
    if intent not in INTENTS:
        return None
    subjects: list[str] = []
    raw_subjects = data.get("subjects")
    if isinstance(raw_subjects, list):
        for item in raw_subjects:
            name = re.sub(r"\s+", "", str(item or ""))
            if len(name) >= 2 and name not in subjects:
                subjects.append(name)
            if len(subjects) >= 4:
                break
    if intent == "compare" and len(subjects) < 2:
        subjects = extract_subjects(goal) or subjects
        if len(subjects) < 2:
            return None
    return TaskSpec(
        intent=intent,
        subjects=subjects,
        questions=_clean_questions(data.get("questions")),
        source="llm",
    )


def _parse_analyze_json(raw: str) -> dict[str, Any] | None:
    from app.llm import parse_json_object

    return parse_json_object(raw)
