"""Contract tests for the frozen Agent baseline (B/C/E/F/G). No Postgres."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.agent.runtime import (  # noqa: E402
    accept_snapshot,
    begin_turn,
    finish_turn,
    idle_state,
    is_orphan,
    recover_orphan,
    sse_snapshot_frame,
)
from app.memory.extractor import assert_field_origins, extractor_allows_run, project  # noqa: E402
from app.memory.healthy import can_promote_gap, healthy  # noqa: E402
from app.memory.promote import (  # noqa: E402
    mark_stale_on_ingest,
    promote_gap,
    retract_stale_on_hit,
    should_skip_retrieve_for_gap,
)
from app.memory.query import build_retrieve_query, retrieve_slots_from_user  # noqa: E402
from app.memory.slots import TURN_NEW_TOPIC, apply_ltm_to_state  # noqa: E402
from app.verify.inductive import (  # noqa: E402
    STATUS_SUPPORTED_INDUCTIVE,
    STATUS_UNSUPPORTED_INDUCTIVE,
    judge_inductive,
    split_subclaims,
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _check_checkpoint() -> None:
    idle = idle_state()
    _assert(not is_orphan(idle["checkpoint_json"]), _TEST_MATERIALS["text_005"])
    started = begin_turn(idle)
    _assert(is_orphan(started["checkpoint_json"]), _TEST_MATERIALS["text_006"])
    finished = finish_turn(started)
    _assert(finished["checkpoint_json"] is None, _TEST_MATERIALS["text_007"])
    started = begin_turn(idle)
    restored = recover_orphan(started)
    _assert(restored == idle, f"{_TEST_MATERIALS['text_001']}{restored}")
    _assert(restored["checkpoint_json"] is None, _TEST_MATERIALS["text_008"])
    try:
        begin_turn(started)
        raise SystemExit(_TEST_MATERIALS["text_027"])
    except AssertionError:
        pass
    print("checkpoint lifecycle ok")


def _check_sse_nonmonotonic() -> None:
    snap = idle_state(revision=3)
    frame = sse_snapshot_frame(current_revision=0, snapshot=snap, client_since=9)
    _assert(frame["resumed"] is False, _TEST_MATERIALS["text_009"])
    _assert(frame["revision"] == 0, frame)
    applied = accept_snapshot(9, frame)
    _assert(applied["revision"] == 0, _TEST_MATERIALS["text_010"])
    print("sse snapshot ok")


def _check_query_isolation() -> None:
    goal = _TEST_MATERIALS["text_004"]
    slots = retrieve_slots_from_user(goal, subjects=[_TEST_MATERIALS["text_035"]])
    base = build_retrieve_query(slots)
    adversarial = [
        {
            "kind": "gap_confirmed",
            "key": _TEST_MATERIALS["text_028"],
            "payload": {"query": _TEST_MATERIALS["text_036"]},
            "stale": False,
        },
        {
            "kind": "session_distill",
            "key": _TEST_MATERIALS["text_029"],
            "payload": {"text": _TEST_MATERIALS["text_037"]},
        },
    ]
    extra = apply_ltm_to_state(turn_kind=TURN_NEW_TOPIC, retrieve_slots=slots, items=adversarial)
    again = build_retrieve_query(slots)
    _assert(again == base, f"{_TEST_MATERIALS['text_002']}{again!r} vs {base!r}")
    _assert(_TEST_MATERIALS["text_030"] not in again, again)
    _assert("gap" not in slots and "gap_hint" not in slots and "memory" not in slots, slots)
    _assert("gap_hints" in extra, extra)
    print("retrieve isolation ok")


def _check_promotion() -> None:
    retrieve = {
        "index_version": 1,
        "committed_index_version": 1,
        "search_executed": True,
        "downstream_ok": True,
        "query": _TEST_MATERIALS["text_011"],
        "count": 0,
    }
    run = {
        "id": "r1",
        "status": "refused",
        "steps": [{"action": "retrieve", "observation": retrieve}],
    }
    draft = {"kind": "empty_query", "key": "OAuth", "payload": {"query": _TEST_MATERIALS["text_031"]}}
    item = promote_gap(run, retrieve, draft)
    _assert(item is not None, _TEST_MATERIALS["text_012"])

    lag = dict(retrieve, index_version=1, committed_index_version=2)
    _assert(promote_gap(run, lag, draft) is None, _TEST_MATERIALS["text_013"])

    circuit = {
        "id": "r2",
        "status": "refused",
        "steps": [
            {"action": "error", "observation": {"circuit_open": True}},
            {"action": "retrieve", "observation": retrieve},
        ],
    }
    _assert(not healthy(circuit), _TEST_MATERIALS["text_014"])
    _assert(promote_gap(circuit, retrieve, draft) is None, _TEST_MATERIALS["text_015"])

    cancelled = {"id": "r3", "status": "cancelled", "rolled_back": True, "steps": []}
    _assert(not extractor_allows_run(cancelled), "cancelled skipped")
    crashed = {"id": "r4", "status": "crashed", "steps": []}
    _assert(not extractor_allows_run(crashed), "crashed skipped")
    _assert(promote_gap(cancelled, retrieve, draft) is None, "cancelled no promote")
    _assert(promote_gap(crashed, retrieve, draft) is None, "crashed no promote")

    unready = dict(retrieve, search_executed=True, downstream_ok=False)
    _assert(not can_promote_gap(run, unready), _TEST_MATERIALS["text_016"])
    print("promotion / health ok")


def _check_stale_lifecycle() -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    items = [
        {
            "id": "m1",
            "kind": "gap_confirmed",
            "key": "OAuth",
            "source_run_id": "r1",
            "stale": False,
            "retracted": False,
            "version": 1,
        }
    ]
    stale = mark_stale_on_ingest(
        items,
        ingest_id="ing1",
        ingest_time=now,
        run_time_by_id={"r1": now},
    )
    _assert(stale[0]["stale"] is True, stale)
    _assert(not should_skip_retrieve_for_gap(stale[0], digest_unchanged=True), "stale cannot skip")
    live = dict(items[0])
    _assert(should_skip_retrieve_for_gap(live, digest_unchanged=True), _TEST_MATERIALS["text_017"])
    hit = retract_stale_on_hit(stale[0], answered=True)
    _assert(hit["retracted"] is True and hit["stale"] is False, hit)
    _assert(hit["version"] == 2, hit)
    print("stale lifecycle ok")


def _check_extractor() -> None:
    events = [
        {
            "id": "e1",
            "type": "empty_retrieve",
            "input": _TEST_MATERIALS["text_032"],
            "observation": {"query": _TEST_MATERIALS["text_038"], "coverage_rate": 0, "reason_code": "refuse"},
        }
    ]
    drafts = project(
        events, model_extra={"why": _TEST_MATERIALS["text_039"], "kind": "empty_query", "source_event_ids": ["e1"]}
    )
    _assert(drafts, "should project")
    assert_field_origins(drafts[0], events)
    _assert("why" not in (drafts[0].get("payload") or {}), drafts[0])
    _assert(drafts[0]["extractor_overflow"].get("why"), _TEST_MATERIALS["text_018"])
    print("extractor projection ok")


def _check_claim_split() -> None:
    parts = split_subclaims(_TEST_MATERIALS["text_019"])
    _assert(len(parts) >= 2, parts)
    inserted = split_subclaims(_TEST_MATERIALS["text_020"])
    _assert(len(inserted) >= 2, inserted)
    print(_TEST_MATERIALS["text_021"])


def _check_inductive() -> None:
    evidence = {1: {"text": _TEST_MATERIALS["text_033"]}, 2: {"text": _TEST_MATERIALS["text_034"]}}
    ok = judge_inductive(
        _TEST_MATERIALS["text_022"],
        citation_indices=[1, 2],
        evidence_by_index=evidence,
        subset_entailed={
            (_TEST_MATERIALS["text_044"], (1, 2)): True,
            (_TEST_MATERIALS["text_045"], (1, 2)): True,
        },
    )
    _assert(ok == STATUS_SUPPORTED_INDUCTIVE, ok)
    bad = judge_inductive(
        _TEST_MATERIALS["text_023"],
        citation_indices=[1, 2],
        evidence_by_index=evidence,
        subset_entailed={(_TEST_MATERIALS["text_046"], (1, 2)): False},
    )
    _assert(bad == STATUS_UNSUPPORTED_INDUCTIVE, bad)
    stealth = judge_inductive(
        _TEST_MATERIALS["text_024"],
        citation_indices=[1],
        evidence_by_index={1: {"text": _TEST_MATERIALS["text_047"]}},
        subset_entailed={(_TEST_MATERIALS["text_048"], (1,)): False},
    )
    _assert(stealth == STATUS_UNSUPPORTED_INDUCTIVE, stealth)
    print("inductive adversarial ok")


def _check_classification() -> None:
    from app.agent.classify import classify_task_kind
    from app.memory.slots import TURN_INVENTORY, resolve_turn_kind

    _assert(classify_task_kind(_TEST_MATERIALS["text_040"]) == "factoid", _TEST_MATERIALS["text_025"])
    _assert(classify_task_kind(_TEST_MATERIALS["text_041"]) == "synthesis", "deliverable stays synthesis")
    _assert(
        resolve_turn_kind(_TEST_MATERIALS["text_042"], _TEST_MATERIALS["text_043"]) == TURN_INVENTORY,
        _TEST_MATERIALS["text_026"],
    )
    print("classification gold ok")


def _check_eval_isolation() -> None:
    golden = json.loads((ROOT / "data" / "eval" / "golden" / _TEST_MATERIALS["text_049"]).read_text(encoding="utf-8"))
    agent_path = ROOT / "data" / "eval" / "agent_quality" / "agent-quality-v1.json"
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
    g_files = {Path(name).name for name in (golden.get("corpus") or [])}
    a_files = {Path(name).name for name in (agent.get("corpus") or [])}
    overlap = g_files & a_files
    _assert(not overlap, f"{_TEST_MATERIALS['text_003']}{overlap}")
    print("eval isolation ok")


def main() -> None:
    _check_checkpoint()
    _check_sse_nonmonotonic()
    _check_query_isolation()
    _check_promotion()
    _check_stale_lifecycle()
    _check_extractor()
    _check_claim_split()
    _check_inductive()
    _check_classification()
    _check_eval_isolation()
    print("agent contract ok")


if __name__ == "__main__":
    main()
