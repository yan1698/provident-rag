"""Prior-run summary for follow-up turns. Summary only — never the evidence body."""

from __future__ import annotations

import re
from typing import Any

MARKER = "【上一轮摘要】"
_MAX = 1200
_SUBJECT_LINE = re.compile(r"主题：(.+)")
_QUESTION_LINE = re.compile(r"问题：(.+)")
_SUBQUESTION_LINE = re.compile(r"子问题：(.+)")

TURN_RETRIEVE = "retrieve"
TURN_SUMMARIZE = "summarize"
TURN_CLARIFY = "clarify"
TURN_OFFTOPIC = "offtopic"
MEMORY_TURNS = frozenset({TURN_SUMMARIZE, TURN_CLARIFY, TURN_OFFTOPIC})

_EXPAND_RE = re.compile(r"展开|补充|详细说|再讲|继续查|还有哪些|那.+呢")
_SUMMARIZE_RE = re.compile(r"之前问的是什么|刚才问|上一[问轮题]|用更短|更简短|简短的句子|概括一下|复述|缩短|精简")
_SCOPE_META_RE = re.compile(
    r"为什么只|为啥只|只(是)?搜|只搜查|其他知识库|其余(的)?(知识库|文档)|"
    r"这部分知识|没搜|其他(文档|库)的呢"
)
_CORPUS_ASK_RE = re.compile(r"覆盖面|整个知识库|全部文档|所有文档|知识库(里)?有哪些|盘点知识库|有哪些文档")
_CLARIFY_RE = re.compile(r"是什么意思|指的是什么|你说的")
_SECTION_RE = re.compile(r"第\s*([一二三四五六七八九十\d]+)\s*节")
_CHAT_RE = re.compile(r"^(你好|谢谢|在吗|哈哈|嗯+|ok|好的)[。！!？?\s]*$", re.IGNORECASE)
_PARA_RE = re.compile(r"第\s*([零一二三四五六七八九十\d]+)\s*(段|部分)")
_PICK_RE = re.compile(r"^(?:第\s*)?([零一二三四五六七八九十\d]+)\s*段?$")
_FUZZY_REF_RE = re.compile(r"展开一下|那段|刚才说的|上面那段|后面说的|不是这个")
_STEP_RE = re.compile(
    r"(?:第\s*([零一二三四五六七八九十\d]+)\s*(?:个\s*)?(?:操作\s*)?步骤?|步骤\s*([零一二三四五六七八九十\d]+))"
)
_CONTINUE_RE = re.compile(
    r"这个(?:备份)?操作|该操作|这一步|这个步骤|上述|刚才的|接着|继续|再(?:说明|解释|补充)|(?:它|其)的"
)
_NEW_TASK_RE = re.compile(r"换个(?:问题|任务)|改问|另一个任务|改为(?:讨论|回答)|现在(?:问|讨论)")
_WHY_RE = re.compile(r"为什么|为何|为啥")
_RECITE_RE = re.compile(r"复述|内容是啥|内容是什么")
_PENDING_ACTION = re.compile(r"待确认动作：(recite|expand|explain)")
_PENDING_ACTIONS = frozenset({"recite", "expand", "explain"})
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# Closed-world quote match. Not a topic similarity.
QUOTE_MIN_CHARS = 24
QUOTE_COVER = 0.75


def compose_goal(goal: str, brief: str) -> str:
    user = (goal or "").strip()
    memory = (brief or "").strip()
    if not memory:
        return user
    return f"{user}\n\n{MARKER}\n{memory}"


def split_goal(text: str) -> tuple[str, str]:
    raw = text or ""
    if MARKER not in raw:
        return raw.strip(), ""
    user, brief = raw.split(MARKER, 1)
    return user.strip(), brief.strip()


def parse_subjects(brief: str) -> list[str]:
    matches = _SUBJECT_LINE.findall(brief or "")
    if not matches:
        return []
    return [part.strip() for part in matches[-1].split("、") if part.strip()][:6]


def is_answer_pointer(goal: str) -> bool:
    text = (goal or "").strip()
    if _NEW_TASK_RE.search(text):
        return False
    if re.search(r"(?:原文|手册|文档)(?:中|里|的)?\s*(?:第\s*[零一二三四五六七八九十\d]+|步骤)", text):
        return False
    return bool(_STEP_RE.search(text) or _PARA_RE.search(text))


def run_operation_steps(run) -> list[dict]:
    """Rebuild legacy combined-paragraph anchors in memory; never rewrite history."""
    from app.generate.prompt import EvidenceItem

    outline = run.outline or {}
    allowed = EvidenceItem.__dataclass_fields__
    evidence = []
    for item in outline.get("evidence", []):
        if not isinstance(item, dict):
            continue
        try:
            evidence.append(EvidenceItem(**{k: v for k, v in item.items() if k in allowed}))
        except TypeError:
            continue
    rebuilt = index_operation_steps(
        outline.get("modules", []), evidence, operation_plan=(outline.get("task_spec") or {}).get("intent") == "plan"
    )
    return rebuilt if evidence and outline.get("modules") else list(outline.get("steps") or [])


def task_update_kind(goal: str, brief: str = "") -> str:
    """Explicit goals replace slots; deictic follow-ups inherit context, never queries."""
    text = (goal or "").strip()
    if not brief:
        return "new"
    if is_scope_meta(text):
        return "clarify"
    if _NEW_TASK_RE.search(text):
        return "new"
    if _CHAT_RE.match(text):
        return "offtopic"
    if _SUMMARIZE_RE.search(text):
        return "restate"
    if re.search(r"(?:这个|那个|该)条件", text) and not re.search(r"[:：\"“]", text):
        return "clarify"
    if _PARA_RE.search(text):
        return "explain"
    if (
        _STEP_RE.search(text)
        or _CONTINUE_RE.search(text)
        or re.match(r"^(?:展开|解释|查看)?第\s*[零一二三四五六七八九十\d]+\s*节", text)
    ):
        return "continue"
    return "new"


def is_scope_meta(goal: str) -> bool:
    return bool(_SCOPE_META_RE.search((goal or "").strip()))


def is_corpus_ask(goal: str) -> bool:
    return bool(_CORPUS_ASK_RE.search((goal or "").strip()))


def classify_turn_kind(goal: str, brief: str = "") -> str:
    """Session-turn router. Not a product lane. Rules only, no subject lexicon."""
    text = (goal or "").strip()
    if not (brief or "").strip():
        return TURN_RETRIEVE
    if is_scope_meta(text):
        return TURN_CLARIFY
    if is_corpus_ask(text):
        return TURN_RETRIEVE
    if _EXPAND_RE.search(text):
        return TURN_RETRIEVE
    if _SUMMARIZE_RE.search(text):
        return TURN_SUMMARIZE
    if _SECTION_RE.search(text) and "展开" not in text:
        return TURN_CLARIFY
    if _CLARIFY_RE.search(text):
        return TURN_CLARIFY
    if _CHAT_RE.match(text):
        return TURN_OFFTOPIC
    return TURN_RETRIEVE


def _cn_index(token: str) -> int | None:
    text = (token or "").strip()
    if not text:
        return None
    if text.isdigit():
        value = int(text)
        return value if value > 0 else None
    if any(ch not in "零一二三四五六七八九十" for ch in text):
        return None
    if text == "十":
        return 10
    if "十" not in text:
        if len(text) == 1:
            return _CN_DIGITS.get(text) or None
        return None
    left, right = text.split("十", 1)
    if left and left not in _CN_DIGITS:
        return None
    if right and right not in _CN_DIGITS:
        return None
    tens = _CN_DIGITS[left] if left else 1
    ones = _CN_DIGITS[right] if right else 0
    value = tens * 10 + ones
    return value if value > 0 else None


def _norm_chars(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def index_paragraphs(answer: str) -> list[dict[str, Any]]:
    """Paragraphs of the delivered answer. start is inclusive, end is exclusive."""
    text = answer or ""
    blocks = text.split("\n\n")
    found: list[dict[str, Any]] = []
    pos = 0
    for block_index, block in enumerate(blocks):
        start = pos
        end = pos + len(block)
        if block_index < len(blocks) - 1:
            pos = end + 2
        else:
            pos = end
        body_start = start + (len(block) - len(block.lstrip()))
        body_end = end - (len(block) - len(block.rstrip()))
        if body_end <= body_start:
            continue
        body = text[body_start:body_end]
        found.append(
            {
                "index": len(found) + 1,
                "head": body[:40],
                "start": body_start,
                "end": body_end,
            }
        )
    return found


def index_operation_steps(modules: list[dict], evidence: list[Any], *, operation_plan: bool = False) -> list[dict]:
    """Only explicit operation numbering counts; paragraph/section numbers do not."""
    import hashlib

    by_index = {item.index: item for item in evidence}
    steps = []
    seen = set()
    for module in modules:
        for paragraph in module.get("paragraphs") or []:
            text = str(paragraph.get("text") or "")
            matches = list(re.finditer(r"第\s*([零一二三四五六七八九十\d]+)\s*步", text))
            pieces = [
                (
                    text[m.start() : matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip(),
                    _cn_index(m.group(1)),
                )
                for i, m in enumerate(matches)
            ]
            if not pieces:
                match = re.match(r"^\s*([1-9]\d?)[.)、]\s+", text) if operation_plan else None
                number = paragraph.get("step_number")
                if not isinstance(number, int) or isinstance(number, bool):
                    number = _cn_index(match.group(1)) if match else None
                pieces = [(text, number)]
            sources = [by_index[n] for n in paragraph.get("sources", []) if n in by_index]
            for body, number in pieces:
                if not number or not sources:
                    continue
                if number in seen:
                    # Conflicting numbering is ambiguous, never silently pick the first.
                    prior = next((step for step in steps if step["index"] == number), None)
                    if prior and (prior["text"] != body or prior["chunk_ids"] != [s.chunk_id for s in sources]):
                        prior["ambiguous"] = True
                    continue
                seen.add(number)
                steps.append(
                    {
                        "id": f"step-{number}",
                        "kind": "step",
                        "index": number,
                        "text": body,
                        "head": body[:80],
                        "sources": [s.index for s in sources],
                        "chunk_ids": [s.chunk_id for s in sources],
                        "source_proofs": [
                            {
                                "chunk_id": s.chunk_id,
                                "doc_id": s.doc_id,
                                "chunk_hash": s.chunk_hash,
                                "text_sha256": hashlib.sha256(s.text.encode()).hexdigest(),
                                "section_id": getattr(s, "section_id", ""),
                                "section_path": getattr(s, "section_path", ""),
                            }
                            for s in sources
                        ],
                    }
                )
    return sorted([step for step in steps if not step.get("ambiguous")], key=lambda s: s["index"])


def _paragraph_text(answer: str, item: dict[str, Any]) -> str:
    start = int(item.get("start") or 0)
    end = int(item.get("end") or 0)
    if 0 <= start < end <= len(answer or ""):
        return (answer or "")[start:end]
    return str(item.get("text") or "")


def _load_paragraphs(answer: str, paragraphs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if paragraphs:
        loaded: list[dict[str, Any]] = []
        for item in paragraphs:
            if not isinstance(item, dict):
                continue
            body = _paragraph_text(answer, item)
            if not body:
                continue
            loaded.append(
                {
                    "index": int(item.get("index") or len(loaded) + 1),
                    "head": str(item.get("head") or body[:40]),
                    "start": int(item.get("start") or 0),
                    "end": int(item.get("end") or 0),
                    "text": body,
                    "chunk_ids": list(item.get("chunk_ids") or []),
                    "source_proofs": list(item.get("source_proofs") or []),
                }
            )
        if loaded:
            return loaded
    return [{**item, "text": _paragraph_text(answer, item)} for item in index_paragraphs(answer)]


def _quote_coverage(paragraph: str, user: str) -> float:
    left = _norm_chars(paragraph)
    right = _norm_chars(user)
    if not left or not right:
        return 0.0
    if len(left) < QUOTE_MIN_CHARS:
        return 1.0 if left in right else 0.0
    if left in right:
        return 1.0
    covered = 0
    cursor = 0
    while cursor < len(left):
        lo, hi = 1, len(left) - cursor
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if left[cursor : cursor + mid] in right:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best == 0:
            cursor += 1
            continue
        covered += best
        cursor += best
    return covered / len(left)


def _matching_quotes(goal: str, paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in paragraphs:
        if _quote_coverage(str(item.get("text") or ""), goal) >= QUOTE_COVER:
            hits.append(item)
    return hits


def _ask_message(paragraphs: list[dict[str, Any]]) -> str:
    lines = ["你想看哪一段？请选择："]
    for item in paragraphs:
        lines.append(f"{item['index']}. {item.get('head') or ''}")
    return "\n".join(lines)


def _reference(
    *,
    action: str,
    paragraphs: list[dict[str, Any]],
    source_answer: str,
    item: dict[str, Any] | None = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "hit": True,
        "action": action,
        "index": None if item is None else item.get("index"),
        "text": "" if item is None else str(item.get("text") or ""),
        "message": message,
        "candidates": [
            {"index": row["index"], "head": row.get("head") or "", "start": row.get("start"), "end": row.get("end")}
            for row in paragraphs
        ],
        "source_answer": source_answer,
        "retrieve": False,
        "kind": "paragraph",
        "chunk_ids": list((item or {}).get("chunk_ids") or []),
        "source_proofs": list((item or {}).get("source_proofs") or []),
    }


def _last_turn_text(brief: str) -> str:
    parts = re.split(r"\n\n(?=第\d+轮\n)", brief or "")
    return parts[-1] if parts else ""


def _pending_action(brief: str, explicit: str = "") -> str:
    if explicit in {"recite", "expand", "explain"}:
        return explicit
    match = _PENDING_ACTION.search(_last_turn_text(brief))
    if match:
        return match.group(1)
    return "recite"


def _action_outside(goal: str, quoted: str = "") -> str:
    outside = goal or ""
    removed = _norm_chars(quoted)
    if removed and removed in _norm_chars(outside):
        outside = _norm_chars(outside).replace(removed, "", 1)
    if _RECITE_RE.search(outside) or _SUMMARIZE_RE.search(outside):
        return "recite"
    if "展开" in outside:
        return "expand"
    if _WHY_RE.search(outside) or "什么意思" in outside:
        return "explain"
    return "recite"


def _run_outline(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        outline = row.get("outline")
    else:
        outline = getattr(row, "outline", None)
    return outline if isinstance(outline, dict) else {}


def _run_answer(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("answer") or "")
    return str(getattr(row, "answer", "") or "")


def _run_id(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("id") or row.get("run_id") or "")
    return str(getattr(row, "id", "") or "")


def is_substantive_answer(outline: dict[str, Any] | None, answer: str = "") -> bool:
    """A new answer the user can point at. Clarifications do not qualify."""
    mode = str((outline or {}).get("compose_mode") or "")
    if mode in {"memory", "reference"}:
        return False
    if mode in {"written", "catalog"}:
        return bool((answer or "").strip())
    return False


def select_answer_anchor(runs: list[Any]) -> dict[str, Any]:
    """The answer a paragraph pointer uses. Not always the latest reply."""
    rows = [row for row in (runs or []) if _run_outline(row).get("answer_source") != "general_explanation"]
    empty = {"run_id": "", "answer": "", "paragraphs": []}
    if not rows:
        return empty
    latest = rows[-1]
    outline = _run_outline(latest)
    anchor = outline.get("answer_anchor")
    if isinstance(anchor, dict) and anchor.get("run_id") and not anchor.get("fresh"):
        target_id = str(anchor["run_id"])
        for row in rows:
            if _run_id(row) == target_id:
                return _pack_anchor(row)
    if is_substantive_answer(outline, _run_answer(latest)):
        return _pack_anchor(latest)
    for row in reversed(rows[:-1]):
        if is_substantive_answer(_run_outline(row), _run_answer(row)):
            return _pack_anchor(row)
    return _pack_anchor(latest)


def _pack_anchor(row: Any) -> dict[str, Any]:
    outline = _run_outline(row)
    answer = _run_answer(row)
    stored = outline.get("paragraphs")
    paragraphs = list(stored) if isinstance(stored, list) and stored else index_paragraphs(answer)
    return {"run_id": _run_id(row), "answer": answer, "paragraphs": paragraphs}


def resolve_answer_reference(
    goal: str,
    *,
    brief: str = "",
    answer: str = "",
    paragraphs: list[dict[str, Any]] | None = None,
    pending_action: str = "",
    steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Closed-world pointer to the previous answer. None means a new topic.

    Paragraph numbers use 段/部分. 节 stays a document section and is not handled here.
    A bare 「2」 is a selection only when this turn was seeded with pending_action.
    The summary line 「待确认：」 is display text and does not open that gate.
    """
    text = (goal or "").strip()
    if not text:
        return None
    if _NEW_TASK_RE.search(text):
        return None
    source = answer or ""
    rows = _load_paragraphs(source, paragraphs)
    step_match = _STEP_RE.search(text)
    if step_match:
        if re.search(r"(?:原文|手册|文档)(?:中|里|的)?\s*(?:第\s*[零一二三四五六七八九十\d]+|步骤)", text):
            return None  # original procedure numbering is not answer-step numbering
        numbers = {_cn_index(m.group(1) or m.group(2)) for m in _STEP_RE.finditer(text)}
        ordered = list(dict.fromkeys(_cn_index(m.group(1) or m.group(2)) for m in _STEP_RE.finditer(text)))
        comparing = re.search(r"比较|对比|区别|差异|异同|有什么不同|有何不同|分别|各自", text)
        if len(numbers) > 1 and None not in numbers and comparing and not re.search(r"还是|或者|或是", text):
            groups = [[s for s in steps or [] if s.get("index") == number] for number in ordered]
            if all(len(group) == 1 and group[0].get("chunk_ids") and group[0].get("source_proofs") for group in groups):
                anchors = [
                    {k: group[0].get(k) for k in ("index", "text", "chunk_ids", "source_proofs")} for group in groups
                ]
                return {
                    **_reference(action="explain", paragraphs=list(steps or []), source_answer=source),
                    "kind": "step",
                    "indices": ordered,
                    "anchors": anchors,
                    "retrieve": True,
                    "text": "\n".join(a["text"] for a in anchors),
                    "chunk_ids": list(dict.fromkeys(cid for a in anchors for cid in a["chunk_ids"])),
                    "source_proofs": [proof for a in anchors for proof in a["source_proofs"]],
                }
        number = _cn_index(step_match.group(1) or step_match.group(2)) if len(numbers) == 1 else None
        matches = [item for item in steps or [] if item.get("index") == number]
        chosen = matches[0] if len(matches) == 1 else None
        if chosen:
            return {
                **_reference(action="explain", paragraphs=list(steps or []), source_answer=source, item=chosen),
                "kind": "step",
                "retrieve": True,
                "chunk_ids": list(chosen.get("chunk_ids") or []),
                "source_proofs": list(chosen.get("source_proofs") or []),
            }
        return {
            **_reference(
                action="ask",
                paragraphs=list(steps or []),
                source_answer=source,
                message="无法定位该操作步骤：上一回答没有这个带来源的步骤，请指定操作或原文。",
            ),
            "kind": "step",
        }
    if re.search(r"(?:这个|那个|该)条件", text) and not re.search(r"[:：\"“]", text):
        return {
            **_reference(
                action="ask",
                paragraphs=[],
                source_answer=source,
                message="请明确是哪一个条件，例如引用该条件原句；当前不能唯一确定，尚未读取新增事实。",
            ),
            "kind": "condition",
        }
    pending = pending_action in _PENDING_ACTIONS
    if not rows and not pending:
        return None
    if pending and _PICK_RE.match(text):
        picked = _cn_index(_PICK_RE.match(text).group(1))
        chosen = next((item for item in rows if item["index"] == picked), None)
        if chosen is None:
            return _reference(
                action="ask",
                paragraphs=rows,
                source_answer=source,
                message=_ask_message(rows),
            )
        return _reference(
            action=_pending_action(brief, pending_action),
            paragraphs=rows,
            source_answer=source,
            item=chosen,
        )
    quotes = _matching_quotes(text, rows)
    if len(quotes) > 1:
        return _reference(action="ask", paragraphs=rows, source_answer=source, message=_ask_message(rows))
    if len(quotes) == 1:
        return _reference(
            action=_action_outside(text, str(quotes[0].get("text") or "")),
            paragraphs=rows,
            source_answer=source,
            item=quotes[0],
        )
    if _SECTION_RE.search(text) and not _PARA_RE.search(text):
        return None
    numbers = [_cn_index(match.group(1)) for match in _PARA_RE.finditer(text)]
    numbers = [item for item in numbers if item]
    if numbers:
        target = numbers[0]
        chosen = next((item for item in rows if item["index"] == target), None)
        if chosen is None:
            return _reference(action="ask", paragraphs=rows, source_answer=source, message=_ask_message(rows))
        return _reference(
            action=_action_outside(text),
            paragraphs=rows,
            source_answer=source,
            item=chosen,
        )
    if rows and _FUZZY_REF_RE.search(text):
        if len(rows) == 1 and "展开" in text:
            return _reference(action="expand", paragraphs=rows, source_answer=source, item=rows[0])
        return _reference(action="ask", paragraphs=rows, source_answer=source, message=_ask_message(rows))
    return None


def parse_questions(brief: str) -> list[str]:
    matches = _SUBQUESTION_LINE.findall(brief or "")
    if not matches:
        return []
    return [part.strip() for part in re.split(r"[；;]", matches[-1]) if part.strip()][:6]


def resolve_followup_questions(
    intent: str,
    goal: str,
    subjects: list[str],
    spec_questions: list[Any],
    brief: str,
) -> list[str]:
    """Prefer questions that already name the subjects. Template is the fallback."""
    from app.agent.analyze import accept_questions, build_questions

    for raw in (spec_questions,):
        accepted = accept_questions(intent, goal, subjects, raw)
        if accepted is None:
            continue
        if subjects and accepted and all(item.strip() == (goal or "").strip() for item in accepted):
            continue
        return accepted
    return build_questions(intent, goal, subjects)


def summarize_run(
    *,
    goal: str = "",
    answer: str = "",
    outline: dict[str, Any] | None = None,
    gaps: list[Any] | None = None,
) -> str:
    parts = [f"问题：{(goal or '').strip()[:160]}"]
    spec = (outline or {}).get("task_spec") if isinstance(outline, dict) else None
    if isinstance(spec, dict):
        intent = str(spec.get("intent") or "").strip()
        if intent:
            parts.append(f"意图：{intent}")
        subjects = [str(item).strip() for item in (spec.get("subjects") or []) if str(item).strip()]
        if subjects:
            parts.append("主题：" + "、".join(subjects[:6]))
        questions = [str(item).strip() for item in (spec.get("questions") or []) if str(item).strip()]
        if questions:
            parts.append("子问题：" + "；".join(questions[:4]))
    stored = (outline or {}).get("paragraphs") if isinstance(outline, dict) else None
    titles: list[str] = []
    if isinstance(stored, list) and stored:
        for item in stored:
            if not isinstance(item, dict):
                continue
            head = str(item.get("head") or "").strip()
            index = item.get("index") or (len(titles) + 1)
            if head:
                titles.append(f"第{index}段 {head}")
            if len(titles) >= 8:
                break
    modules = (outline or {}).get("modules") if isinstance(outline, dict) else None
    if not titles:
        for index, mod in enumerate(modules or [], start=1):
            if not isinstance(mod, dict):
                continue
            heading = str(mod.get("heading") or mod.get("title") or "").strip()
            paras = mod.get("paragraphs") or mod.get("bullets") or []
            snippet = ""
            if paras and isinstance(paras[0], dict):
                snippet = str(paras[0].get("text") or "").replace("\n", " ").strip()[:60]
            label = heading or snippet
            if label:
                titles.append(f"第{index}节 {label}")
            if len(titles) >= 8:
                break
    if titles:
        parts.append("结构：" + "；".join(titles))
    elif (answer or "").strip():
        parts.append("结论：" + " ".join((answer or "").split())[:360])
    pending = (outline or {}).get("pending_paragraphs") if isinstance(outline, dict) else None
    if isinstance(pending, list) and pending:
        heads = [f"{item.get('index')} {item.get('head') or ''}".strip() for item in pending if isinstance(item, dict)]
        if heads:
            parts.append("待确认：" + "；".join(heads[:8]))
        action = str((outline or {}).get("pending_action") or "")
        if action in {"recite", "expand", "explain"}:
            parts.append(f"待确认动作：{action}")
    topics = []
    for item in gaps or []:
        if isinstance(item, dict) and item.get("topic"):
            topics.append(str(item["topic"]).strip())
        if len(topics) >= 5:
            break
    if topics:
        parts.append("未覆盖：" + "、".join(topics))
    text = "\n".join(part for part in parts if part)
    return text[:_MAX]


def compose_session_memory(runs: list[Any], *, limit: int = 3) -> str:
    """Join the last few turn summaries. Still no evidence body."""
    items = [row for row in (runs or []) if _run_outline(row).get("answer_source") != "general_explanation"][
        -max(int(limit), 1) :
    ]
    parts: list[str] = []
    for index, row in enumerate(items, start=1):
        if isinstance(row, dict):
            goal = str(row.get("goal") or "")
            answer = str(row.get("answer") or "")
            outline = row.get("outline") if isinstance(row.get("outline"), dict) else None
            gaps = list(row.get("gaps") or [])
        else:
            goal = str(getattr(row, "goal", "") or "")
            answer = str(getattr(row, "answer", "") or "")
            raw_outline = getattr(row, "outline", None)
            outline = raw_outline if isinstance(raw_outline, dict) else None
            gaps = list(getattr(row, "gaps", None) or [])
        brief = summarize_run(goal=goal, answer=answer, outline=outline, gaps=gaps)
        if brief:
            parts.append(f"第{index}轮\n{brief}")
    # Keep the newest turn, not a prefix dominated by the first task.
    kept: list[str] = []
    for part in reversed(parts):
        if kept and len("\n\n".join([part] + kept)) > _MAX:
            break
        kept.insert(0, part[:_MAX])
    return "\n\n".join(kept)


def merge_followup_spec(
    spec: dict[str, Any],
    *,
    user_goal: str,
    brief: str,
) -> dict[str, Any]:
    """Reuse prior subjects only when the follow-up stays on that domain retrieve."""
    from app.agent.analyze import detect_intent, domain_subjects, extract_subjects

    out = dict(spec)
    turn_kind = classify_turn_kind(user_goal, brief)
    out["turn_kind"] = turn_kind
    out["update"] = task_update_kind(user_goal, brief)
    goal = (user_goal or "").strip()
    if is_scope_meta(goal):
        out["questions"] = []
        return out
    if is_corpus_ask(goal):
        current = domain_subjects(goal, extract_subjects(goal))
        out["intent"] = "inventory"
        out["subjects"] = current
        out["questions"] = []
        return out
    update = task_update_kind(goal, brief)
    out["update"] = update
    prior = parse_subjects(brief)
    if update in {"continue", "restate", "explain"} and prior:
        out["subjects"] = prior
    subjects = [str(item).strip() for item in (out.get("subjects") or []) if str(item).strip()]
    intent = str(out.get("intent") or detect_intent(goal) or "explain")
    if turn_kind in MEMORY_TURNS:
        out["questions"] = []
        return out
    if update == "continue":
        context = "、".join(subjects)
        if re.fullmatch(r"(?:展开|解释|查看)?第\s*[零一二三四五六七八九十\d]+\s*节", goal):
            context += "；" + "；".join(parse_questions(brief))
        out["questions"] = [f"{context}：{goal}"[:400]]
        return out
    out["questions"] = resolve_followup_questions(
        intent,
        goal,
        subjects,
        list(out.get("questions") or []),
        brief,
    )
    return out


def _shorten(text: str, *, limit: int = 360) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    parts = re.split(r"(?<=[。！？!?])", compact)
    out: list[str] = []
    for piece in parts:
        chunk = piece.strip()
        if not chunk:
            continue
        if out and sum(len(item) for item in out) + len(chunk) > limit:
            break
        out.append(chunk)
    return "".join(out) or compact[:limit]


def compose_catalog_listing(catalog: list[Any] | None) -> str:
    docs = list(catalog or [])
    if not docs:
        return "知识库目录为空，请先上传文档。"
    lines = [
        f"当前知识库共 {len(docs)} 份文档。下面按目录列出，不是按某个领域主题检索的结果。",
        "",
    ]
    for doc in docs:
        filename = str(getattr(doc, "filename", "") or "")
        sections = [str(item).strip() for item in (getattr(doc, "sections", None) or []) if str(item).strip()]
        if not filename:
            continue
        if not sections:
            lines.append(filename)
            continue
        entries = getattr(doc, "section_entries", [])
        if entries:
            lines.append(filename)
            lines.extend("  " * max(0, s["level"] - 1) + s.get("label", s["title"]) for s in entries)
        else:
            lines.append(f"{filename}：" + "、".join(sections))
    return "\n".join(lines).strip()


def compose_followup_answer(
    goal: str,
    brief: str,
    prior_answer: str = "",
    turn_kind: str = TURN_SUMMARIZE,
    catalog: list[Any] | None = None,
) -> str:
    """Rewrite from last answer + summary. Never pastes evidence chunks."""
    text = (goal or "").strip()
    memory = brief or ""
    prior = (prior_answer or "").strip()
    if turn_kind == TURN_OFFTOPIC:
        return "这不是当前对话里的知识库任务。换主题请点「新开对话」。"
    asked = _QUESTION_LINE.search(memory)
    if turn_kind == TURN_SUMMARIZE and re.search(r"之前问|刚才问|上一[问轮题]", text):
        topic = asked.group(1).strip() if asked else ""
        if topic:
            return f"你上一轮问的是：{topic}"
        return "上一轮的问题我这边没有记下，请再描述一次主题。"
    if turn_kind == TURN_CLARIFY:
        if is_scope_meta(text):
            return _compose_scope_clarify(memory, catalog)
        match = re.search(r"第\s*([0-9]+)\s*节", text)
        if match:
            number = match.group(1)
            found = re.search(rf"第{number}节\s+([^；;]+)", memory)
            title = found.group(1).strip() if found else ""
            if title:
                return f"上一轮第{number}节讲的是「{title}」。若要按知识库展开这一节，请追问「展开第{number}节」。"
        if prior:
            return _shorten(prior)
        return "上一轮没有可澄清的提纲。"
    if prior:
        return _shorten(prior)
    if asked:
        return f"上一轮的问题是：{asked.group(1).strip()}"
    return "上一轮还没有可复述的回答。"


def _compose_scope_clarify(brief: str, catalog: list[Any] | None) -> str:
    asked = _QUESTION_LINE.search(brief or "")
    subjects = parse_subjects(brief or "")
    subq = re.search(r"子问题：(.+)", brief or "")
    parts: list[str] = []
    if asked:
        parts.append(f"上一轮问的是「{asked.group(1).strip()}」。")
    if subjects:
        parts.append(f"检索按主题「{'、'.join(subjects)}」进行，没有把整个知识库当作盘点范围。")
    if subq:
        parts.append(f"当时的子问题是：{subq.group(1).strip()}。")
    names = [str(getattr(doc, "filename", "") or "") for doc in (catalog or [])]
    names = [name for name in names if name]
    if names:
        shown = "、".join(names[:16])
        extra = f" 等 {len(names)} 份" if len(names) > 16 else ""
        parts.append(f"当前目录里的文档：{shown}{extra}。")
    parts.append("若要按文档清单看覆盖面，请直接问「知识库覆盖了哪些文档」。")
    return "".join(parts)


def selfcheck() -> None:
    brief = summarize_run(
        goal="梳理发票报销要点",
        outline={
            "task_spec": {"intent": "explain", "subjects": ["发票报销"]},
            "modules": [{"heading": "时限", "paragraphs": [{"text": "90 日内提交。"}]}],
        },
        gaps=[{"topic": "差旅标准", "reason": "库中未见"}],
    )
    if "主题：发票报销" not in brief or "第1节 时限" not in brief:
        raise SystemExit(f"summary missing structure: {brief}")
    if "差旅标准" not in brief:
        raise SystemExit(f"summary missing gap: {brief}")
    composed = compose_goal("展开第三节", brief)
    user, memory = split_goal(composed)
    if user != "展开第三节" or MARKER not in composed:
        raise SystemExit(f"compose/split failed: {composed!r}")
    merged = merge_followup_spec(
        {"intent": "explain", "subjects": ["展开第三节"], "questions": ["展开第三节"]},
        user_goal=user,
        brief=memory,
    )
    if merged["subjects"] != ["发票报销"]:
        raise SystemExit(f"should reuse prior subjects: {merged}")
    if merged.get("turn_kind") != TURN_RETRIEVE:
        raise SystemExit(f"expand should still retrieve: {merged}")
    if any(item == "展开第三节" for item in merged.get("questions") or []):
        raise SystemExit(f"follow-up text must not be the retrieve query: {merged}")
    if not any("发票报销" in str(item) for item in merged.get("questions") or []):
        raise SystemExit(f"retrieve follow-up should search prior subjects: {merged}")
    restated = merge_followup_spec(
        {"intent": "explain", "subjects": ["展开第三节"], "questions": ["展开第三节"]},
        user_goal="我之前问的是什么，用更简短的句子回答",
        brief=memory,
    )
    if restated.get("turn_kind") != TURN_SUMMARIZE:
        raise SystemExit(f"restatement should skip retrieve: {restated}")
    if restated.get("questions"):
        raise SystemExit(f"summarize must not enqueue retrieve questions: {restated}")
    if restated.get("subjects") != ["发票报销"]:
        raise SystemExit(f"summarize should keep prior subjects: {restated}")
    meta = merge_followup_spec(
        {"intent": "explain", "subjects": ["Agent"], "questions": ["说明Agent"]},
        user_goal="为什么只是搜查了这部分知识库的东西，其他知识库的呢",
        brief=memory,
    )
    if meta.get("turn_kind") != TURN_CLARIFY:
        raise SystemExit(f"scope meta should clarify: {meta}")
    if meta.get("questions"):
        raise SystemExit(f"scope meta must not retrieve: {meta}")
    cover = merge_followup_spec(
        {"intent": "explain", "subjects": ["Agent"], "questions": ["说明Agent"]},
        user_goal="知识库覆盖面",
        brief=memory,
    )
    if cover.get("intent") != "inventory" or cover.get("subjects"):
        raise SystemExit(f"corpus coverage must not reuse Agent: {cover}")
    if cover.get("questions"):
        raise SystemExit(f"corpus coverage has no retrieve questions: {cover}")
    from types import SimpleNamespace

    scoped = compose_followup_answer(
        "为什么只是搜查了这部分知识库的东西，其他知识库的呢",
        memory,
        turn_kind=TURN_CLARIFY,
        catalog=[SimpleNamespace(filename="invoice-policy.md", sections=["发票报销"])],
    )
    if "梳理发票报销要点" not in scoped or "invoice-policy.md" not in scoped:
        raise SystemExit(f"scope clarify should name prior question and catalog: {scoped}")
    listing = compose_catalog_listing(
        [
            SimpleNamespace(filename="bird_guide.pdf", sections=["迁徙"]),
            SimpleNamespace(filename="invoice-policy.md", sections=["时限"]),
        ]
    )
    if "bird_guide.pdf" not in listing or "invoice-policy.md" not in listing:
        raise SystemExit(f"catalog listing should name files: {listing}")
    spoken = compose_followup_answer(
        "我之前问的是什么，用更简短的句子回答",
        memory,
        prior_answer="发票报销须在 90 日内提交。",
        turn_kind=TURN_SUMMARIZE,
    )
    if "梳理发票报销要点" not in spoken:
        raise SystemExit(f"restatement should name the prior question: {spoken}")
    if "证据" in brief and "chunk" in brief.lower():
        raise SystemExit("summary must not carry raw evidence ids")
    bundled = compose_session_memory(
        [
            {
                "goal": "梳理发票报销要点",
                "outline": {"task_spec": {"subjects": ["发票报销"]}},
                "gaps": [{"topic": "差旅标准"}],
            },
            {"goal": "展开第三节", "answer": "时限是 90 日。"},
        ]
    )
    if "第1轮" not in bundled or "第2轮" not in bundled or "展开第三节" not in bundled:
        raise SystemExit(f"session memory should keep recent turns: {bundled}")
    print("agent memory unit ok")


if __name__ == "__main__":
    selfcheck()
