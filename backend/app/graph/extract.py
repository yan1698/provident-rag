"""G1: LLM extraction of concepts and grounded relations. Never writes chunk text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.corpus.catalog import label_chunk_sections
from app.generate.fence import EVIDENCE_RULE, format_evidence_block
from app.graph.schema import (
    KINDS_EXTRACT,
    RELATIONS_ALL,
    DraftEdge,
    DraftEntity,
    GraphDrafts,
    HeadingHit,
    quote_in_chunk,
)

ALREADY_EXTRACTED_NOTE = "已抽取过，跳过"
NO_KEY_NOTE = "未配置任务模型密钥，仅保留章节结构"
SKIP_SHORT_NOTE = "正文过短，仅保留章节结构"
DEFAULT_EXTRACT_TOKENS = 4096
DEFAULT_MIN_CHARS = 800
DEFAULT_WINDOW_CHARS = 6000
DEFAULT_MAX_WINDOWS = 10
MAX_ENTITIES_PER_WINDOW = 8
MAX_RELATED_TO_PER_WINDOW = 2
_SUBSECTION_HEADING_RE = re.compile(r"^\d+\.\d+(?:\.\d+)*\b")
EXTRACT_SYSTEM = (
    EVIDENCE_RULE + "你从用户给出的文档片段抽取概念、关系和正文里已有的标题行。禁止常识补全，正文没写的不要抽。"
    "只输出一个 JSON 对象，不要 Markdown。"
    '形状：{"entities":[{"name":"...","kind":"concept|method|metric|tool|role","quote":"...","aliases":[]}],'
    '"edges":[{"src":"...","src_kind":"concept","dst":"...","dst_kind":"method",'
    '"relation":"defines|requires|causes|contrasts_with|measured_by|instance_of|related_to|part_of","quote":"..."}],'
    '"headings":[{"title":"...","level":1,"quote":"..."}]}。'
    "kind 只能是 concept/method/metric/tool/role。relation 必须是闭集。"
    "【实体准入】每窗最多 8 个实体，优先保留：正文有明确定义或形式化描述、可独立检索、"
    "不是品牌/App 名、附录速查项、版本号、文件名或配置枚举。同一概念只保留最短规范名，"
    "其余记入 aliases，禁止为别名重复建节点。"
    "【关系准入】仅正文明确写出 defines/requires/causes/contrasts_with/measured_by/instance_of/part_of 时才建边。"
    "related_to 仅当正文出现「与…相关」「与…有关」等字样且其它关系不适用时可用，每窗最多 2 条。"
    "禁止跨段推断关系，禁止同义词替换后建边。"
    "【标题抽取】只标正文里已有的标题行，且仅保留 level=1 的章级标题"
    "（如「第一章」「第1章」「Chapter 1」），不要 1.1、2.3 等小节标题；没有则输出 []。"
    "每条 quote 与 heading.title 必须是正文连续子串，不能改写、概括或拼接；对不上就丢弃。"
    "不确定的一律丢弃。宁可少抽，不要编造。"
)


@dataclass
class ExtractWindow:
    section: str
    chunks: list[Any] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join((item.text or "") for item in self.chunks)


def body_char_count(chunks: list[Any]) -> int:
    return sum(len((item.text or "").strip()) for item in chunks)


def should_extract(chunks: list[Any], settings: Any | None = None, *, force: bool = False) -> tuple[bool, str]:
    """Skip LLM when the body is too short. Heading count is not a signal."""
    if force:
        return True, ""
    floor = DEFAULT_MIN_CHARS
    raw = getattr(settings, "graph_extract_min_chars", floor) if settings is not None else floor
    try:
        floor = max(200, int(raw or DEFAULT_MIN_CHARS))
    except (TypeError, ValueError):
        floor = DEFAULT_MIN_CHARS
    if body_char_count(chunks) < floor:
        return False, SKIP_SHORT_NOTE
    return True, ""


def pack_extract_windows(
    chunks: list[Any],
    *,
    filename: str = "",
    cap: int = DEFAULT_MAX_WINDOWS,
    max_chars: int = DEFAULT_WINDOW_CHARS,
) -> tuple[list[ExtractWindow], int]:
    """Merge consecutive chunks that share a section. Cap windows, not raw chunks."""
    ordered = list(chunks)
    if not ordered:
        return [], 0
    texts = [item.text or "" for item in ordered]
    labels = label_chunk_sections(texts)
    fallback = filename or "正文"
    groups: list[tuple[str, list[Any]]] = []
    current_key = ""
    bucket: list[Any] = []
    for chunk, labs in zip(ordered, labels):
        key = (labs[0] if labs else "") or fallback
        if bucket and key != current_key:
            groups.append((current_key, bucket))
            bucket = []
        current_key = key
        bucket.append(chunk)
    if bucket:
        groups.append((current_key, bucket))

    windows: list[ExtractWindow] = []
    limit = max(800, int(max_chars or DEFAULT_WINDOW_CHARS))
    for section, group in groups:
        acc: list[Any] = []
        size = 0
        for chunk in group:
            piece = len(chunk.text or "")
            if acc and size + piece > limit:
                windows.append(ExtractWindow(section=section, chunks=acc))
                acc = []
                size = 0
            acc.append(chunk)
            size += piece
        if acc:
            windows.append(ExtractWindow(section=section, chunks=acc))

    keep = max(1, int(cap or DEFAULT_MAX_WINDOWS))
    skipped = max(0, len(windows) - keep)
    return windows[:keep], skipped


def drafts_from_extraction(
    payload: dict[str, Any] | None,
    *,
    chunk_id: str,
    chunk_hash: str,
    text: str,
) -> GraphDrafts:
    """Turn a model JSON object into drafts. Unaligned quotes and unknown labels are dropped."""
    chunk = _FakeChunk(chunk_id, chunk_hash, text)
    drafts = drafts_from_window(payload, [chunk])
    if drafts.headings:
        drafts.extend(heading_tree_from_hits(drafts.headings, "文档"))
    return drafts


class _FakeChunk:
    def __init__(self, chunk_id: str, chunk_hash: str, text: str) -> None:
        self.id = chunk_id
        self.chunk_hash = chunk_hash
        self.text = text


def drafts_from_window(payload: dict[str, Any] | None, chunks: list[Any]) -> GraphDrafts:
    """Align quotes to whichever chunk in the window contains them."""
    drafts = GraphDrafts(source="llm")
    if not isinstance(payload, dict) or not chunks:
        return drafts
    kinds_by_name: dict[str, str] = {}
    for raw in payload.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        kind = str(raw.get("kind") or "concept").strip().lower()
        quote = str(raw.get("quote") or "").strip()
        if kind not in KINDS_EXTRACT or not name:
            continue
        hit = _align_quote(quote, chunks)
        if hit is None:
            drafts.dropped += 1
            continue
        chunk, span, quote_text = hit
        aliases = [
            str(item).strip() for item in (raw.get("aliases") or []) if str(item).strip() and str(item).strip() != name
        ]
        drafts.entities.append(
            DraftEntity(
                name=name,
                kind=kind,
                quote=quote_text,
                chunk_id=str(chunk.id),
                chunk_hash=getattr(chunk, "chunk_hash", None) or "",
                start_char=span[0],
                end_char=span[1],
                aliases=aliases,
            )
        )
        kinds_by_name[name] = kind
    for raw in payload.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        src = str(raw.get("src") or raw.get("src_name") or "").strip()
        dst = str(raw.get("dst") or raw.get("dst_name") or "").strip()
        relation = str(raw.get("relation") or "").strip().lower()
        quote = str(raw.get("quote") or "").strip()
        if relation not in RELATIONS_ALL or not src or not dst or src == dst:
            continue
        hit = _align_quote(quote, chunks)
        if hit is None:
            drafts.dropped += 1
            continue
        chunk, span, quote_text = hit
        src_kind = str(raw.get("src_kind") or kinds_by_name.get(src) or "concept").strip().lower()
        dst_kind = str(raw.get("dst_kind") or kinds_by_name.get(dst) or "concept").strip().lower()
        if src_kind not in KINDS_EXTRACT:
            src_kind = "concept"
        if dst_kind not in KINDS_EXTRACT:
            dst_kind = "concept"
        drafts.edges.append(
            DraftEdge(
                src_name=src,
                src_kind=src_kind,
                dst_name=dst,
                dst_kind=dst_kind,
                relation=relation,
                quote=quote_text,
                chunk_id=str(chunk.id),
                chunk_hash=getattr(chunk, "chunk_hash", None) or "",
                source="llm",
                start_char=span[0],
                end_char=span[1],
            )
        )
        if src not in kinds_by_name:
            drafts.entities.append(
                DraftEntity(
                    name=src,
                    kind=src_kind,
                    quote=quote_text,
                    chunk_id=str(chunk.id),
                    chunk_hash=getattr(chunk, "chunk_hash", None) or "",
                    start_char=span[0],
                    end_char=span[1],
                )
            )
            kinds_by_name[src] = src_kind
        if dst not in kinds_by_name:
            drafts.entities.append(
                DraftEntity(
                    name=dst,
                    kind=dst_kind,
                    quote=quote_text,
                    chunk_id=str(chunk.id),
                    chunk_hash=getattr(chunk, "chunk_hash", None) or "",
                    start_char=span[0],
                    end_char=span[1],
                )
            )
            kinds_by_name[dst] = dst_kind
    _collect_headings(payload, chunks, drafts)
    _trim_window_drafts(drafts)
    return drafts


def _chapter_heading_ok(title: str, level: int) -> bool:
    """Keep chapter-level headings only; drop numbered subsections like 2.1."""
    if level > 1:
        return False
    blob = (title or "").strip()
    if not blob:
        return False
    if _SUBSECTION_HEADING_RE.match(blob):
        return False
    return True


def _trim_window_drafts(drafts: GraphDrafts) -> None:
    """Hard caps mirroring EXTRACT_SYSTEM — models still over-generate."""
    if len(drafts.entities) > MAX_ENTITIES_PER_WINDOW:
        drafts.dropped += len(drafts.entities) - MAX_ENTITIES_PER_WINDOW
        drafts.entities = drafts.entities[:MAX_ENTITIES_PER_WINDOW]
    kept: list[DraftEdge] = []
    related_seen = 0
    for edge in drafts.edges:
        if edge.relation == "related_to":
            if related_seen >= MAX_RELATED_TO_PER_WINDOW:
                drafts.dropped += 1
                continue
            related_seen += 1
        kept.append(edge)
    drafts.edges = kept


def _collect_headings(payload: dict[str, Any], chunks: list[Any], drafts: GraphDrafts) -> None:
    seen: set[str] = set()
    for raw in payload.get("headings") or []:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or raw.get("name") or "").strip()
        quote = str(raw.get("quote") or title).strip()
        if not title:
            continue
        hit = _align_quote(title, chunks) or _align_quote(quote, chunks)
        if hit is None:
            drafts.dropped += 1
            continue
        chunk, span, quote_text = hit
        name = title if quote_in_chunk(title, chunk.text or "") else quote_text.strip()
        if not name:
            drafts.dropped += 1
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            level = int(raw.get("level") or 1)
        except (TypeError, ValueError):
            level = 1
        level = max(1, min(6, level))
        if not _chapter_heading_ok(name, level):
            drafts.dropped += 1
            continue
        drafts.headings.append(
            HeadingHit(
                name=name[:80],
                level=1,
                quote=quote_text,
                chunk_id=str(chunk.id),
                chunk_hash=getattr(chunk, "chunk_hash", None) or "",
                start_char=span[0],
                end_char=span[1],
            )
        )


def heading_tree_from_hits(
    headings: list[HeadingHit],
    document_name: str,
    *,
    chunk_rank: dict[str, int] | None = None,
) -> GraphDrafts:
    """Build doc_section + part_of from grounded heading rows. Invented titles never reach here."""
    ordered = list(headings)
    if chunk_rank:
        ordered.sort(key=lambda hit: (chunk_rank.get(hit.chunk_id, 10_000), hit.start_char))
    else:
        ordered.sort(key=lambda hit: hit.start_char)
    drafts = GraphDrafts(source="llm")
    root = (document_name or "文档").strip() or "文档"
    stack: list[tuple[int, str]] = []
    for hit in ordered:
        while stack and stack[-1][0] >= hit.level:
            stack.pop()
        parent_name = stack[-1][1] if stack else root
        parent_kind = "doc_section" if stack else "document"
        drafts.entities.append(
            DraftEntity(
                name=hit.name,
                kind="doc_section",
                quote=hit.quote,
                chunk_id=hit.chunk_id,
                chunk_hash=hit.chunk_hash,
                start_char=hit.start_char,
                end_char=hit.end_char,
            )
        )
        if hit.name != parent_name:
            drafts.edges.append(
                DraftEdge(
                    src_name=hit.name,
                    src_kind="doc_section",
                    dst_name=parent_name,
                    dst_kind=parent_kind,
                    relation="part_of",
                    quote=hit.quote,
                    chunk_id=hit.chunk_id,
                    chunk_hash=hit.chunk_hash,
                    source="llm",
                    start_char=hit.start_char,
                    end_char=hit.end_char,
                )
            )
        stack.append((hit.level, hit.name))
    return drafts


def _align_quote(quote: str, chunks: list[Any]) -> tuple[Any, tuple[int, int], str] | None:
    for chunk in chunks:
        text = chunk.text or ""
        span = quote_in_chunk(quote, text)
        if span is None:
            continue
        return chunk, span, text[span[0] : span[1]]
    return None


def _extract_model(settings: Any) -> str:
    """Qwen uses the task model. DeepSeek keeps its separate non-reasoning model."""
    from app.llm import task_model_name, task_provider_name

    if task_provider_name(settings) == "deepseek":
        return extract_deepseek_model(settings)
    return task_model_name(settings)


def extract_deepseek_model(settings: Any) -> str:
    """Non-reasoning chat model by default — v4-flash spends the budget on reasoning_content."""
    raw = (
        getattr(settings, "graph_extract_deepseek_model", "deepseek-chat") if settings is not None else "deepseek-chat"
    )
    model = str(raw or "deepseek-chat").strip()
    if model:
        return model
    fallback = getattr(settings, "llm_deepseek_model", "deepseek-chat") if settings is not None else "deepseek-chat"
    return str(fallback or "deepseek-chat").strip() or "deepseek-chat"


def llm_wrote_entities(drafts: GraphDrafts) -> bool:
    """True when the model returned at least one aligned entity, edge, or heading."""
    return bool(drafts.entities or drafts.edges or drafts.headings)


def extract_token_budget(settings: Any) -> int:
    """Independent of LLM_MAX_OUTPUT_TOKENS — compose budgets starve reasoning models."""
    raw = getattr(settings, "graph_extract_max_tokens", DEFAULT_EXTRACT_TOKENS)
    try:
        n = int(raw or DEFAULT_EXTRACT_TOKENS)
    except (TypeError, ValueError):
        n = DEFAULT_EXTRACT_TOKENS
    return max(800, min(n, 4096))


PROMPT_OVERHEAD_TOKENS = 400
CHARS_PER_INPUT_TOKEN = 2


def estimate_extract(document: Any, chunks: list[Any], settings: Any | None = None) -> dict[str, Any]:
    """Upper-bound token estimate. Already-extracted docs cost zero."""
    ordered = sorted(chunks, key=lambda row: (getattr(row, "page", 0), getattr(row, "start_offset", 0)))
    chars = body_char_count(ordered)
    out = {
        "extracted": getattr(document, "graph_extracted_at", None) is not None,
        "chars": chars,
        "calls": 0,
        "skipped_windows": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "skip": "",
    }
    if out["extracted"]:
        out["skip"] = "already"
        return out
    ok, reason = should_extract(ordered, settings)
    if not ok:
        out["skip"] = "short"
        return out
    cap = (
        max(1, min(12, int(getattr(settings, "graph_extract_max_chunks", DEFAULT_MAX_WINDOWS) or DEFAULT_MAX_WINDOWS)))
        if settings
        else DEFAULT_MAX_WINDOWS
    )
    window_chars = (
        int(getattr(settings, "graph_extract_window_chars", DEFAULT_WINDOW_CHARS) or DEFAULT_WINDOW_CHARS)
        if settings
        else DEFAULT_WINDOW_CHARS
    )
    windows, skipped = pack_extract_windows(
        ordered,
        filename=getattr(document, "filename", "") or "",
        cap=cap,
        max_chars=window_chars,
    )
    budget = extract_token_budget(settings) if settings is not None else DEFAULT_EXTRACT_TOKENS
    input_tokens = 0
    for window in windows:
        input_tokens += PROMPT_OVERHEAD_TOKENS + max(
            1, (len(window.text) + CHARS_PER_INPUT_TOKEN - 1) // CHARS_PER_INPUT_TOKEN
        )
    output_tokens = len(windows) * budget
    out["calls"] = len(windows)
    out["skipped_windows"] = skipped
    out["input_tokens"] = input_tokens
    out["output_tokens"] = output_tokens
    out["total_tokens"] = input_tokens + output_tokens
    if reason:
        out["skip"] = reason
    return out


def extract_document(
    document: Any,
    chunks: list[Any],
    settings: Any,
    *,
    force: bool = False,
) -> GraphDrafts:
    """Extract concepts for a document. Stamped docs are never sent to the model again."""
    drafts = GraphDrafts(source="llm")
    if getattr(document, "graph_extracted_at", None) is not None:
        drafts.note = ALREADY_EXTRACTED_NOTE
        return drafts
    from app.llm import LLMError, router_llm_available

    ordered = sorted(chunks, key=lambda row: (row.page, row.start_offset))
    if not router_llm_available(settings):
        drafts.note = NO_KEY_NOTE
        return drafts
    ok, reason = should_extract(ordered, settings, force=force)
    if not ok:
        drafts.note = reason
        return drafts
    cap = max(
        1, min(12, int(getattr(settings, "graph_extract_max_chunks", DEFAULT_MAX_WINDOWS) or DEFAULT_MAX_WINDOWS))
    )
    window_chars = int(getattr(settings, "graph_extract_window_chars", DEFAULT_WINDOW_CHARS) or DEFAULT_WINDOW_CHARS)
    windows, skipped = pack_extract_windows(
        ordered,
        filename=getattr(document, "filename", "") or "",
        cap=cap,
        max_chars=window_chars,
    )
    drafts.skipped = skipped
    if skipped:
        drafts.note = f"还有 {skipped} 个章节窗口未抽取"
    fail_notes: list[str] = []
    filename = getattr(document, "filename", "") or "文档"
    for window in windows:
        drafts.calls += 1
        try:
            payload = _extract_window(window, settings, filename=filename)
        except (LLMError, TypeError) as exc:
            drafts.failed += 1
            fail_notes.append(str(exc))
            continue
        if not isinstance(payload, dict):
            drafts.empty += 1
            continue
        drafts.extend(drafts_from_window(payload, window.chunks))
    if drafts.headings:
        chunk_rank = {str(chunk.id): index for index, chunk in enumerate(ordered)}
        drafts.headings.sort(key=lambda hit: (chunk_rank.get(hit.chunk_id, 10_000), hit.start_char))
        drafts.extend(heading_tree_from_hits(drafts.headings, filename, chunk_rank=chunk_rank))
    drafts.note = _extract_note(drafts, fail_notes)
    return drafts


def _extract_note(drafts: GraphDrafts, fail_notes: list[str]) -> str:
    parts = [piece for piece in (drafts.note,) if piece]
    if drafts.failed:
        sample = fail_notes[-1] if fail_notes else "未知错误"
        parts.append(f"抽取失败 {drafts.failed} 窗：{sample}")
    if drafts.empty:
        parts.append(f"空响应 {drafts.empty} 窗")
    if drafts.dropped and not drafts.entities and not drafts.edges:
        parts.append(f"有 {drafts.dropped} 条对不上原文已丢弃")
    if not drafts.entities and not drafts.edges:
        if drafts.failed or drafts.empty:
            parts.append("抽取未写入概念")
        elif SKIP_SHORT_NOTE not in "；".join(parts) and "抽取未写入" not in "；".join(parts):
            parts.append("模型未抽出可对齐原文的概念")
    return "；".join(dict.fromkeys(parts))


def _retryable(exc: Exception) -> bool:
    kind = str(getattr(exc, "kind", "") or "")
    if kind in {"unavailable", "timeout"}:
        return True
    if kind in {"auth", "rate_limit"}:
        return False
    text = str(exc).lower()
    return "empty response" in text or "request failed" in text


def _extract_window(window: ExtractWindow, settings: Any, *, filename: str) -> dict[str, Any] | None:
    from app.llm import LLMError, get_router_llm_client, parse_json_object

    client = get_router_llm_client(settings)
    body = window.text[
        : max(800, int(getattr(settings, "graph_extract_window_chars", DEFAULT_WINDOW_CHARS) or DEFAULT_WINDOW_CHARS))
    ]
    prompt = f"文档：{filename}\n章节：{window.section}\n正文：\n{format_evidence_block(body, source=filename)}"
    budget = extract_token_budget(settings)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = client.generate(
                system=EXTRACT_SYSTEM,
                prompt=prompt,
                max_output_tokens=budget,
                temperature=0.0,
                json_object=True,
                model=_extract_model(settings),
                purpose="extract",
            ).text
        except TypeError:
            raw = client.generate(
                system=EXTRACT_SYSTEM,
                prompt=prompt,
                max_output_tokens=budget,
                temperature=0.0,
                model=_extract_model(settings),
                purpose="extract",
            ).text
        except LLMError as exc:
            last_error = exc
            if _retryable(exc) and attempt == 0:
                continue
            raise
        data = parse_json_object(raw)
        if data is not None:
            return data
        # JSON is locally repairable or not; do not spend a second model call.
        return None
    if last_error is not None:
        raise last_error
    return None
