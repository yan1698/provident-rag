"""Agent runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.corpus.catalog import CatalogDocument
from app.generate.prompt import EvidenceItem


class AgentPhase(str, Enum):
    INIT = "init"
    ANALYZING = "analyzing"
    SURVEYING = "surveying"
    DIGESTING = "digesting"
    RETRIEVING = "retrieving"
    ANSWERING = "answering"
    COMPOSING = "composing"
    VERIFYING = "verifying"
    DONE = "done"


@dataclass
class TaskSpec:
    intent: str = "explain"  # explain | compare | plan | inventory
    subjects: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    deliverable: str = ""
    source: str = "rules"
    turn_kind: str = "retrieve"  # retrieve | summarize | clarify | offtopic
    update: str = "new"  # new | continue | explain | clarify | restate | offtopic

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "subjects": list(self.subjects),
            "questions": list(self.questions),
            "deliverable": self.deliverable,
            "source": self.source,
            "turn_kind": self.turn_kind,
            "update": self.update,
        }


@dataclass
class SynthesisNode:
    id: str
    title: str
    query: str
    status: str = "pending"  # pending | retrieved | empty | gap
    section: str | None = None
    doc_id: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    question: str = ""
    source_sections: list[dict[str, str]] = field(default_factory=list)
    score: float = 0.0
    retried: bool = False


@dataclass
class GapItem:
    topic: str
    reason: str


@dataclass
class AgentState:
    goal: str
    task_kind: str = "factoid"
    requested_intent: str | None = None
    phase: AgentPhase = AgentPhase.INIT
    catalog: list[CatalogDocument] = field(default_factory=list)
    task_spec: TaskSpec | None = None
    nodes: list[SynthesisNode] = field(default_factory=list)
    outline: dict[str, Any] | None = None
    gaps: list[GapItem] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_by_node: dict[str, list[int]] = field(default_factory=dict)
    digests: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] | None = None
    compose_mode: str = ""
    fallback_reason: str = ""
    partial: bool = False
    task_coverage: float | None = None
    hit_count: int = 0
    retrieve_attempts: int = 0
    subtopic_retrieves: int = 0
    total_steps: int = 0
    max_steps: int = 8
    max_retrieves: int = 2
    revised: bool = False
    owner_id: Any = None
    graph_dimensions: list[str] = field(default_factory=list)
    graph_used_edges: list[dict[str, Any]] = field(default_factory=list)
    usage: list[Any] = field(default_factory=list)
    memory_brief: str = ""
    turn_kind: str = "retrieve"
    prior_answer: str = ""
    retrieve_slots: dict[str, Any] = field(default_factory=dict)
    last_retrieve_obs: dict[str, Any] = field(default_factory=dict)
    gap_hints: list[dict[str, Any]] = field(default_factory=list)
    ltm_items: list[dict[str, Any]] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)
    evidence_selection: dict[str, Any] = field(default_factory=dict)
    reading: dict[str, Any] = field(default_factory=dict)
    run_budget: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] | None = None
    operation_steps: list[dict[str, Any]] = field(default_factory=list)
    operation_anchor: str = ""

    def working_goal(self) -> str:
        from app.agent.memory import compose_goal

        return compose_goal(self.goal, self.memory_brief)

    def skips_retrieve(self) -> bool:
        from app.agent.memory import MEMORY_TURNS

        return (self.turn_kind or "retrieve") in MEMORY_TURNS

    def catalog_inventory(self) -> bool:
        from app.agent.analyze import is_catalog_inventory

        spec = self.task_spec
        if spec is None:
            return False
        return is_catalog_inventory(spec.intent, spec.subjects, self.goal)
