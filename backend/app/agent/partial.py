"""Partial delivery: keep supported claims and list the rest as gaps."""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState, GapItem
from app.verify.service import STATUS_SUPPORTED

_HALLUCINATION = {"unsupported", "uncited", "invalid_citation"}


def only_unverified_gaps(verification: dict[str, Any] | None) -> bool:
    """Unknown support is not a generation error or a source violation."""
    if not isinstance(verification, dict) or verification.get("hallucination_count", 0):
        return False
    claims = verification.get("claims") or []
    statuses = {c.get("status") for c in claims if isinstance(c, dict)}
    return "unverified" in statuses and statuses <= {"supported", "unverified", "insufficient", "meta"}


def supported_count(observation: dict[str, Any] | None) -> int:
    if not isinstance(observation, dict):
        return 0
    raw = observation.get("supported_count")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    claims = observation.get("claims") or []
    return sum(1 for claim in claims if isinstance(claim, dict) and claim.get("status") == STATUS_SUPPORTED)


def should_partial(verification: dict[str, Any] | None, *, coverage_floor: float) -> bool:
    if not isinstance(verification, dict):
        return False
    if only_unverified_gaps(verification):
        return False
    try:
        coverage = float(verification.get("coverage_rate") or 0.0)
    except (TypeError, ValueError):
        coverage = 0.0
    return coverage < float(coverage_floor) and supported_count(verification) > 0


def apply_partial_answer(state: AgentState) -> dict[str, Any]:
    """Rewrite the factoid answer to supported claims only. Gaps stay on state."""
    from app.generate.synthesis import attach_citations, format_cite_markers, plain_answer, plain_prose

    verification = state.verification if isinstance(state.verification, dict) else {}
    claims = verification.get("claims") or []
    kept: list[str] = []
    gaps: list[GapItem] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = plain_prose(str(claim.get("text") or "").strip())
        if not text:
            continue
        status = str(claim.get("status") or "")
        if status in {STATUS_SUPPORTED, "unverified"}:
            body, cites = attach_citations(text, claim.get("citation_indices") or [])
            marker = format_cite_markers(cites)
            kept.append(f"{body} {marker}".strip() if marker else body)
        elif status in _HALLUCINATION:
            gaps.append(GapItem(topic=text[:48], reason="现有证据未覆盖该部分"))
    if not kept:
        return {"partial": False}
    body = "\n".join(kept)
    if gaps:
        missing = "；".join(item.topic for item in gaps)
        body = f"{body}\n\n知识库未写明：{missing}。"
    from app.generate.synthesis import plain_answer

    state.answer = plain_answer(body)
    seen = {(item.topic, item.reason) for item in state.gaps}
    for item in gaps:
        if (item.topic, item.reason) not in seen:
            state.gaps.append(item)
            seen.add((item.topic, item.reason))
    state.partial = True
    state.fallback_reason = "覆盖率不足，已交付已覆盖部分"
    return {
        "partial": True,
        "note": state.fallback_reason,
        "gap_count": len(gaps),
        "kept_claims": len(kept),
    }
