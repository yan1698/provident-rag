"""Session checkpoint lifecycle. No second orphan detector.

Lock: checkpoint_json is not None ⇔ an in-flight turn (orphan if worker died).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

IDLE_REVISION = 0


def is_orphan(checkpoint_json: Any) -> bool:
    return checkpoint_json is not None


def idle_state(*, revision: int = IDLE_REVISION, slots: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "revision": int(revision),
        "slots": dict(slots or {}),
        "catalog_digest": [],
        "gaps": [],
        "checkpoint_json": None,
    }


def begin_turn(state: dict[str, Any]) -> dict[str, Any]:
    """Start a turn: freeze an idle snapshot as checkpoint."""
    if state.get("checkpoint_json") is not None:
        raise AssertionError("cannot begin_turn while checkpoint exists")
    snapshot = deepcopy(state)
    snapshot["checkpoint_json"] = None
    out = deepcopy(state)
    out["checkpoint_json"] = snapshot
    return out


def finish_turn(state: dict[str, Any]) -> dict[str, Any]:
    """Normal terminal: checkpoint must exist, then it is cleared."""
    if state.get("checkpoint_json") is None:
        raise AssertionError("finish_turn requires checkpoint_json")
    out = deepcopy(state)
    out["checkpoint_json"] = None
    out["revision"] = int(out.get("revision") or 0) + 1
    return out


def recover_orphan(state: dict[str, Any]) -> dict[str, Any]:
    """Restore the frozen idle snapshot. Resulting checkpoint is None."""
    checkpoint = state.get("checkpoint_json")
    if checkpoint is None:
        raise AssertionError("recover_orphan requires checkpoint_json")
    restored = deepcopy(checkpoint)
    if restored.get("checkpoint_json") is not None:
        raise AssertionError("checkpoint snapshot itself must be idle")
    return restored


def sse_snapshot_frame(
    *,
    current_revision: int,
    snapshot: dict[str, Any],
    client_since: int | None = None,
) -> dict[str, Any]:
    """Reconnect always gets a full snapshot. Never resume from client_since."""
    del client_since  # protocol: ignore; first frame is always current state
    return {
        "type": "snapshot",
        "revision": int(current_revision),
        "state": snapshot,
        "resumed": False,
    }


def accept_snapshot(local_revision: int | None, frame: dict[str, Any]) -> dict[str, Any]:
    """Client must apply snapshot even when revision is not monotonic."""
    if str(frame.get("type") or "") != "snapshot":
        raise AssertionError("expected snapshot frame")
    return frame
