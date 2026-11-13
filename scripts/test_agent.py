"""Day 13 acceptance: agent planner/audit log + feedback."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def _request(url: str, data: bytes | None = None, *, timeout: int = 240, method: str | None = None) -> dict:
    method = method or ("GET" if data is None else "POST")
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} {url}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc


def _post(url: str, payload: dict, *, timeout: int = 240) -> dict:
    return _request(url, data=json.dumps(payload).encode("utf-8"), timeout=timeout)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _wait_agent(run_id: str, *, timeout_s: int = 240) -> dict:
    """POST /agent/run returns immediately; poll until terminal status."""
    import time

    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = _request(f"{BASE}/agent/{run_id}", timeout=30)
        if last.get("status") in {"finished", "refused", "error", "cancelled", "crashed"}:
            return last
        time.sleep(1.0)
    raise SystemExit(f"FAIL: agent {run_id} still {last.get('status')!r} after {timeout_s}s")


def _version_tuple(raw: str) -> tuple[int, ...]:
    parts = []
    for piece in str(raw).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _check_rules_planner() -> None:
    from app.agent.planner import rule_plan

    first = rule_plan("chunk metadata", [])
    _assert(first.action == "retrieve", f"{_TEST_MATERIALS['text_001']}{first}")
    after_empty = rule_plan(
        "chunk metadata",
        [{"action": "retrieve", "observation": {"count": 0, "hits": []}}],
    )
    _assert(after_empty.action == "retrieve", f"{_TEST_MATERIALS['text_002']}{after_empty}")
    after_empty_again = rule_plan(
        "chunk metadata",
        [
            {"action": "retrieve", "input": _TEST_MATERIALS["text_027"], "observation": {"count": 0, "hits": []}},
            {"action": "retrieve", "input": _TEST_MATERIALS["text_028"], "observation": {"count": 0, "hits": []}},
        ],
        max_retrieves=2,
    )
    _assert(after_empty_again.action == "refuse", f"{_TEST_MATERIALS['text_003']}{after_empty_again}")
    after_hits = rule_plan(
        "chunk metadata",
        [{"action": "retrieve", "observation": {"count": 2, "hits": [{}, {}]}}],
    )
    _assert(after_hits.action == "answer", f"hits should answer: {after_hits}")
    after_answer = rule_plan(
        "chunk metadata",
        [
            {"action": "retrieve", "observation": {"count": 2}},
            {"action": "answer", "observation": {"answer": _TEST_MATERIALS["text_034"]}},
        ],
    )
    _assert(after_answer.action == "verify", f"answered should verify: {after_answer}")

    after_good_verify = rule_plan(
        "chunk metadata",
        [
            {"action": "retrieve", "observation": {"count": 2}},
            {"action": "answer", "observation": {"answer": _TEST_MATERIALS["text_035"]}},
            {"action": "verify", "observation": {"coverage_rate": 0.9}},
        ],
        min_coverage=0.5,
        max_retrieves=2,
    )
    _assert(after_good_verify.action == "finish", f"{_TEST_MATERIALS['text_004']}{after_good_verify}")

    after_weak_verify = rule_plan(
        "chunk metadata",
        [
            {"action": "retrieve", "observation": {"count": 2}},
            {"action": "answer", "observation": {"answer": _TEST_MATERIALS["text_036"]}},
            {
                "action": "verify",
                "observation": {
                    "coverage_rate": 0.2,
                    "claims": [{"text": _TEST_MATERIALS["text_043"], "hallucination": True}],
                },
            },
        ],
        min_coverage=0.5,
        max_retrieves=2,
    )
    _assert(after_weak_verify.action == "retrieve", f"{_TEST_MATERIALS['text_005']}{after_weak_verify}")
    _assert(
        "missing definition" not in after_weak_verify.input,
        f"{_TEST_MATERIALS['text_006']}{after_weak_verify.input}",
    )
    _assert("chunk metadata" in after_weak_verify.input, f"retry query: {after_weak_verify.input}")

    chatty = rule_plan(_TEST_MATERIALS["text_012"], [])
    _assert(chatty.action == "retrieve", chatty)
    _assert(
        _TEST_MATERIALS["text_029"] not in chatty.input and _TEST_MATERIALS["text_030"] not in chatty.input,
        f"chat filler: {chatty.input}",
    )
    _assert("agent" in chatty.input.casefold(), f"topic dropped: {chatty.input}")

    after_retry_exhausted = rule_plan(
        "chunk metadata",
        [
            {"action": "retrieve", "observation": {"count": 2}},
            {"action": "answer", "observation": {"answer": _TEST_MATERIALS["text_037"]}},
            {"action": "verify", "observation": {"coverage_rate": 0.1}},
            {"action": "retrieve", "observation": {"count": 3}},
            {"action": "answer", "observation": {"answer": _TEST_MATERIALS["text_038"]}},
            {"action": "verify", "observation": {"coverage_rate": 0.2}},
        ],
        min_coverage=0.5,
        max_retrieves=2,
    )
    _assert(
        after_retry_exhausted.action == "refuse",
        f"{_TEST_MATERIALS['text_007']}{after_retry_exhausted}",
    )

    from app.agent.analyze import split_factoid_queries
    from app.agent.partial import apply_partial_answer
    from app.agent.state import AgentState

    queries = split_factoid_queries(_TEST_MATERIALS["text_013"])
    _assert(len(queries) == 2, queries)
    _assert("Workflow" in queries[0] and "Agent" not in queries[0], queries[0])
    _assert("Agent" in queries[1] and "Workflow" not in queries[1], queries[1])
    _assert(len(split_factoid_queries(_TEST_MATERIALS["text_039"])) == 1, _TEST_MATERIALS["text_014"])

    first_part = rule_plan(_TEST_MATERIALS["text_015"], [])
    _assert(first_part.action == "retrieve" and "Workflow" in first_part.input, first_part)
    second_part = rule_plan(
        _TEST_MATERIALS["text_016"],
        [{"action": "retrieve", "input": first_part.input, "observation": {"count": 2, "hits": [{}, {}]}}],
    )
    _assert(second_part.action == "retrieve" and "Agent" in second_part.input, second_part)
    after_both = rule_plan(
        _TEST_MATERIALS["text_017"],
        [
            {"action": "retrieve", "observation": {"count": 2}},
            {"action": "retrieve", "observation": {"count": 4}},
        ],
    )
    _assert(after_both.action == "answer", after_both)

    after_partial = rule_plan(
        "chunk metadata",
        [
            {"action": "retrieve", "observation": {"count": 2}},
            {"action": "answer", "observation": {"answer": _TEST_MATERIALS["text_040"]}},
            {"action": "retrieve", "observation": {"count": 3}},
            {"action": "answer", "observation": {"answer": _TEST_MATERIALS["text_041"]}},
            {
                "action": "verify",
                "observation": {
                    "coverage_rate": 0.2,
                    "supported_count": 2,
                    "claims": [
                        {"text": _TEST_MATERIALS["text_044"], "status": "supported", "citation_indices": [1]},
                        {"text": _TEST_MATERIALS["text_045"], "status": "unsupported", "citation_indices": [1]},
                    ],
                },
            },
        ],
        min_coverage=0.5,
        max_retrieves=2,
    )
    _assert(after_partial.action == "finish", after_partial)
    _assert(_TEST_MATERIALS["text_020"] in after_partial.thought, after_partial.thought)

    partial_state = AgentState(goal=_TEST_MATERIALS["text_021"], answer=_TEST_MATERIALS["text_022"])
    partial_state.verification = {
        "coverage_rate": 0.2,
        "supported_count": 1,
        "claims": [
            {"text": _TEST_MATERIALS["text_031"], "status": "supported", "citation_indices": [1]},
            {"text": _TEST_MATERIALS["text_032"], "status": "unsupported", "citation_indices": []},
        ],
    }
    extra = apply_partial_answer(partial_state)
    _assert(extra.get("partial") is True, extra)
    _assert("file_hash" in partial_state.answer, partial_state.answer)
    _assert(
        _TEST_MATERIALS["text_023"] not in partial_state.answer.split(_TEST_MATERIALS["text_042"])[0],
        partial_state.answer,
    )
    _assert(partial_state.gaps and _TEST_MATERIALS["text_033"] in partial_state.gaps[0].topic, partial_state.gaps)
    print("rules planner ok")


def main() -> None:
    _check_rules_planner()

    root = _request(f"{BASE}/")
    _assert(_version_tuple(root.get("version", "0")) >= (0, 13), f"need API 0.13+: {root.get('version')}")
    _assert(root.get("agent") == "/agent/run", f"root missing agent: {root}")
    _assert(root.get("feedback") == "/feedback", f"root missing feedback: {root}")

    health = _request(f"{BASE}/health")
    _assert(health.get("status") == "ok", f"health not ok: {health}")

    started = _post(
        f"{BASE}/agent/run",
        {
            "goal": _TEST_MATERIALS["text_024"],
            "planner": "rules",
        },
        timeout=30,
    )
    _assert(started.get("id"), f"{_TEST_MATERIALS['text_008']}{started}")
    _assert(started.get("status") == "queued", f"expected queued immediately: {started.get('status')}")
    run = _wait_agent(started["id"])
    actions = [item.get("action") for item in run.get("steps") or []]
    _assert(run.get("status") in {"finished", "refused"}, f"unexpected status: {run.get('status')}")
    _assert("retrieve" in actions, f"{_TEST_MATERIALS['text_009']}{actions}")
    _assert(run.get("step_count", 0) >= 1, _TEST_MATERIALS["text_018"])
    print(f"agent {run['id']} status={run['status']} steps={actions}")

    fetched = _request(f"{BASE}/agent/{run['id']}")
    _assert(fetched.get("goal") == run.get("goal"), _TEST_MATERIALS["text_019"])
    listing = _request(f"{BASE}/agent?limit=5")
    _assert(any(item.get("id") == run["id"] for item in listing.get("results") or []), "run not listed")

    missing = "00000000-0000-0000-0000-000000000000"
    try:
        _post(f"{BASE}/feedback", {"rating": "up", "trace_id": missing})
        raise SystemExit(_TEST_MATERIALS["text_025"])
    except SystemExit as exc:
        _assert("HTTP 404" in str(exc), f"expected 404, got {exc}")

    fb = _post(f"{BASE}/feedback", {"rating": "up", "agent_run_id": run["id"], "comment": "day13"})
    _assert(fb.get("rating") == "up", f"feedback: {fb}")
    listed = _request(f"{BASE}/feedback?agent_run_id={run['id']}")
    _assert(listed.get("up") >= 1, f"feedback list: {listed}")
    print("feedback ok")

    deleted = _request(f"{BASE}/agent/{run['id']}", method="DELETE")
    _assert(deleted.get("deleted") is True, f"delete: {deleted}")
    _assert(deleted.get("id") == run["id"], f"delete id mismatch: {deleted}")
    listing_after = _request(f"{BASE}/agent?limit=20")
    _assert(
        all(item.get("id") != run["id"] for item in listing_after.get("results") or []),
        "deleted run still listed",
    )
    try:
        _request(f"{BASE}/agent/{run['id']}")
        raise SystemExit(_TEST_MATERIALS["text_026"])
    except SystemExit as exc:
        _assert("HTTP 404" in str(exc), f"{_TEST_MATERIALS['text_011']}{exc}")
    fb_after = _request(f"{BASE}/feedback?agent_run_id={run['id']}")
    _assert(fb_after.get("count", 0) == 0, f"{_TEST_MATERIALS['text_010']}{fb_after}")
    print("delete ok")


if __name__ == "__main__":
    main()
