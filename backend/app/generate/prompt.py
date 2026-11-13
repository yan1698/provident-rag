from __future__ import annotations

from dataclasses import asdict, dataclass

from app.generate.fence import EVIDENCE_RULE, format_evidence_block

PROMPT_VERSION = "answer-v2-evidence-budget"

SYSTEM_PROMPT = (
    "你是 Provident RAG 系统的问答助手。只能依据下面提供的证据回答用户问题，禁止编造证据之外的信息。"
    + EVIDENCE_RULE
    + "用连贯的中文句子作答，不要 Markdown：不要标题、列表、表格、加粗、斜体、代码围栏或链接语法。"
    "每一句陈述结尾必须标注引用来源，格式为 [S<编号>]，编号对应证据列表中的顺序；"
    "如果一句话由多条证据共同支持，可以连续标注多个来源，例如 [S1][S3]。"
    "证据正文里若出现单独的脚注数字（如句末的 1、56），不要抄进答案；引用只使用 [S<编号>]。"
    "只回答用户问的那一层，不要把证据里未问到的相邻论点写成必答项。"
    "如果现有证据不足以回答问题，必须直接说明“现有证据不足以回答该问题”，不要猜测或使用背景知识。"
)


@dataclass
class EvidenceItem:
    index: int
    chunk_id: str
    doc_id: str
    page: int
    text: str
    chunk_hash: str = ""
    section_path: str = ""
    role: str = "hit"  # hit | neighbor
    score: float = 0.0
    start_offset: int = 0
    parent_chunk_id: str = ""
    section_id: str = ""


@dataclass
class EvidencePrompt:
    text: str
    evidence: list[EvidenceItem]
    selection: dict


def prepare_prompt(query: str, evidence: list[EvidenceItem], *, max_context_chars: int) -> EvidencePrompt:
    """Select once: these exact blocks are also the sole citation whitelist."""
    lines = [f"用户问题：{query}", "", "证据列表："]
    used = 0
    selected: list[EvidenceItem] = []
    for item in evidence:
        block = format_evidence_block(item.text, index=item.index, source=item.doc_id, page=item.page) + "\n"
        if used and used + len(block) > max_context_chars:
            break
        lines.append(block)
        selected.append(item)
        used += len(block)
    lines.append("请基于上面的证据用连贯中文句子回答，不要使用 Markdown，并按系统提示的格式标注引用。")
    sent = {item.index for item in selected}
    return EvidencePrompt(
        text="\n".join(lines),
        evidence=selected,
        selection={
            "candidate_evidence": [asdict(item) for item in evidence],
            "sent_indices": [item.index for item in selected],
            "omitted_indices": [item.index for item in evidence if item.index not in sent],
            "max_context_chars": max_context_chars,
            "evidence_chars": used,
            # Preserve the existing first-block budget policy, but make it observable.
            "first_block_exceeds_budget": bool(selected and used > max_context_chars),
        },
    )


def build_prompt(query: str, evidence: list[EvidenceItem], *, max_context_chars: int) -> str:
    """Compatibility entry point for prompt-only callers (e.g. offline evaluation)."""
    return prepare_prompt(query, evidence, max_context_chars=max_context_chars).text
