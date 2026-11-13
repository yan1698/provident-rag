"""Apply a corpus catalog to agent state (survey step)."""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentPhase, AgentState
from app.corpus.catalog import CatalogBuild, CatalogDocument, catalog_observation

EMPTY_CORPUS_ANSWER = "知识库为空，请先上传文档后再使用 Agent。"


def apply_survey(state: AgentState, built: CatalogBuild) -> tuple[dict[str, Any], bool]:
    """Write catalog onto state. Empty corpus is a terminal refuse."""
    state.phase = AgentPhase.SURVEYING
    state.catalog = [
        CatalogDocument(
            doc_id=item.doc_id,
            filename=item.filename,
            sections=list(item.sections),
            preview=getattr(item, "preview", "") or "",
            section_entries=list(item.section_entries),
        )
        for item in built.documents
    ]
    observation = catalog_observation(built)
    if built.section_count == 0:
        state.phase = AgentPhase.DONE
        state.answer = EMPTY_CORPUS_ANSWER
        state.citations = []
        observation["status"] = "refused"
        return observation, True
    return observation, False
