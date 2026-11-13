"""Retrieval scope. Unclassified documents stay in ``all`` and ``soft``."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class Scope:
    mode: str = "all"
    collection_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    widened: bool = False

    def key(self) -> str:
        ids = ",".join(self.collection_ids)
        return f"{self.mode}:{ids}"

    def allows(self, collection_id: UUID | str | None) -> bool:
        if self.mode == "all" or not self.collection_ids:
            return True
        if collection_id is None:
            return self.mode == "soft"
        token = str(collection_id)
        return token in self.collection_ids

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "collection_ids": list(self.collection_ids),
            "labels": list(self.labels),
            "widened": self.widened,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "Scope":
        data = raw or {}
        mode = str(data.get("mode") or "all")
        if mode not in {"all", "soft", "hard"}:
            mode = "all"
        ids = tuple(dict.fromkeys(str(item) for item in (data.get("collection_ids") or []) if str(item).strip()))
        labels = tuple(str(item) for item in (data.get("labels") or []) if str(item).strip())
        if mode == "all":
            ids = ()
        if mode == "hard" and len(ids) != 1:
            mode = "all"
            ids = ()
        if mode == "soft" and not ids:
            mode = "all"
        return cls(mode=mode, collection_ids=ids, labels=labels, widened=bool(data.get("widened")))


def scope_sql(scope: Scope | None) -> tuple[str, dict[str, str]]:
    """Extra ``chunks`` predicate. Empty when the scope is the owner's full corpus."""
    if scope is None or scope.mode == "all" or not scope.collection_ids:
        return "", {}
    payload = "{" + ",".join(scope.collection_ids) + "}"
    if scope.mode == "hard":
        sql = "AND doc_id IN (SELECT id FROM documents WHERE collection_id = ANY(CAST(:scope_ids AS uuid[])))"
    else:
        sql = (
            "AND doc_id IN (SELECT id FROM documents "
            "WHERE collection_id IS NULL OR collection_id = ANY(CAST(:scope_ids AS uuid[])))"
        )
    return sql, {"scope_ids": payload}
