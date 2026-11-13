"""Derived section identities over existing chunk text; no database migration.

IDs identify a heading occurrence within an ingested document, never a title alone.
Chunks are returned intact and in source order. No offset-based text slicing.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.corpus.catalog import iter_headings


def section_records(doc_id: str, rows: list[Any]) -> list[dict]:
    ordered = sorted(rows, key=lambda r: (getattr(r, "start_offset", 0), getattr(r, "page", 1)))
    records: list[dict] = []
    stack: list[dict] = []
    for row in ordered:
        chunk_id = str(getattr(row, "id", getattr(row, "chunk_id", "")))
        matches = iter_headings(row.text or "")
        # Text before a heading can still belong to the previous section.
        if stack and (not matches or matches[0][2] > 0):
            if chunk_id not in stack[-1]["chunk_ids"]:
                stack[-1]["chunk_ids"].append(chunk_id)
        for level, title, local_start in matches:
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            identity = hashlib.sha256(f"{doc_id}:{chunk_id}:{local_start}:{title}".encode()).hexdigest()[:24]
            record = {
                "id": identity,
                "doc_id": str(doc_id),
                "title": title,
                "level": level,
                "path": [item["title"] for item in stack] + [title],
                "parent_id": stack[-1]["id"] if stack else None,
                "ordinal": len(records) + 1,
                "chunk_ids": [chunk_id],
            }
            siblings = [s for s in records if s["parent_id"] == record["parent_id"] and s["title"] == title]
            record["occurrence"] = len(siblings) + 1
            record["label"] = f"{title} ({record['occurrence']})" if siblings else title
            for sibling in siblings:
                sibling["label"] = f"{title} ({sibling['occurrence']})"
            records.append(record)
            stack.append(record)
        if not matches and not stack:
            # Legacy/plain text has a genuine unsectioned range rather than a fake title.
            if not records:
                records.append(
                    {
                        "id": hashlib.sha256(f"{doc_id}:unsectioned".encode()).hexdigest()[:24],
                        "doc_id": str(doc_id),
                        "title": "未分节正文",
                        "level": 1,
                        "path": ["未分节正文"],
                        "parent_id": None,
                        "ordinal": 1,
                        "chunk_ids": [],
                    }
                )
            records[0]["chunk_ids"].append(chunk_id)
    return records


def section_chunk_ids(sections: list[dict], section_id: str, *, descendants: bool = True) -> list[str]:
    found = next((s for s in sections if s["id"] == section_id), None)
    if found is None:
        return []
    chosen = {section_id}
    if descendants:
        for section in sections:
            if section.get("parent_id") in chosen:
                chosen.add(section["id"])
    return list(dict.fromkeys(cid for s in sections if s["id"] in chosen for cid in s["chunk_ids"]))


def norm_title(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]", "", (text or "").casefold())


def required_sections(goal: str, sections: list[dict], *, target_id: str | None = None) -> dict:
    """Explicit title/number first. Related conditions/risks stay in the same branch.

    This identifies reading obligations, not an automatic semantic proof that the
    document contains every prerequisite. Missing categories remain visible.
    """
    normalized = norm_title(goal)
    named = [
        s
        for s in sections
        if (len(norm_title(s["title"])) >= 8 or re.search(r"[\u4e00-\u9fff]{2}", s["title"]))
        and norm_title(re.sub(r"^\d+(?:\.\d+)*\.\s*", "", s["title"])) in normalized
    ]
    if target_id is None and len({norm_title(s["title"]) for s in named}) > 1:
        titles = [norm_title(s["title"]) for s in named]
        if len(set(titles)) != len(titles):
            return {"target_id": None, "required": [], "missing_categories": ["目标章节标题重复，请指定章节身份"]}
        plans = [required_sections(goal, sections, target_id=s["id"]) for s in named]
        required = {s["id"]: s for plan in plans for s in plan["required"]}
        missing = [name for name in ("conditions", "risks") if not any(s["reason"] == name for s in required.values())]
        return {
            "target_id": max(named, key=lambda s: len(s["title"]))["id"],
            "required": list(required.values()),
            "missing_categories": missing,
        }
    numbers = re.findall(r"(?<!\d)(\d+(?:\.\d+){1,4})(?!\d)", goal)
    numbered = [s for s in sections if any(re.match(re.escape(n) + r"\.?(?:\s|$)", s["title"]) for n in numbers)]
    target = (
        next((s for s in sections if s["id"] == target_id), None)
        if target_id is not None
        else max(named or numbered, key=lambda s: len(s["title"]), default=None)
    )
    if target is None:
        return {"target_id": None, "required": [], "missing_categories": ["未唯一识别目标章节，请指定章节标题或编号"]}
    target_number = re.match(r"^(\d+(?:\.\d+)*)", target["title"])
    branch = ".".join(target_number.group(1).split(".")[:2]) if target_number else ""
    required = []
    for section in sections:
        is_child = section["id"] == target["id"] or section["path"][: len(target["path"])] == target["path"]
        is_parent = target["path"][: len(section["path"])] == section["path"]
        same_branch = not branch or any(re.match(re.escape(branch) + r"(?:\.|\s)", title) for title in section["path"])
        risk = bool(re.search(r"caveat|risk|warning|caution|注意|风险|警告", " / ".join(section["path"]), re.I))
        condition = bool(
            re.search(
                r"prerequi|requirement|setting up|before|前置|条件|要求|准备|配置", " / ".join(section["path"]), re.I
            )
        )
        admonition = section["title"] in {"Note", "Important", "Tip", "Warning", "Caution"}
        if is_child or is_parent or (same_branch and (risk or condition or admonition)):
            required.append(
                {
                    **section,
                    "reason": "risks" if risk else "conditions" if condition else "steps" if is_child else "context",
                }
            )
    missing = [name for name in ["conditions", "risks"] if not any(s["reason"] == name for s in required)]
    return {"target_id": target["id"], "required": required, "missing_categories": missing}
