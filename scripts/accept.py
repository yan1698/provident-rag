"""Fixed acceptance matrix for Provident.

offline: CI. No model calls and no torch.
live: lifecycle, including lease ownership. Needs Postgres, Redis, and the api container.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OFFLINE = (
    "test_benchmark",
    "test_benchmark_candidates",
    "test_tags",
    "test_eval_gate",
    "test_queue",
    "test_agent_tools",
    "test_llm_usage",
    "test_agent_memory",
    "test_evidence_fence",
    "test_graph_extract",
    "test_graph_expand",
    "test_graph_compare",
    "test_verify_claims",
    "test_agent_contract",
    "test_agent_synthesis",
    "test_agent_tasks",
    "test_agent_reference_session",
    "test_long_reading_repair",
    "test_bounded_reading",
    "test_compose",
    "test_eval_report",
    "test_citation_integrity",
    "test_prompt_evidence",
    "test_rerank_limit",
    "test_stage_diagnostics",
    "test_retrieval_diagnostics",
    "test_sparse_cache",
    "test_lease_rules",
    "test_delivery_contract",
    "test_step_followup_repair",
    "test_general_explanation",
    "test_unified_routing",
)

LIVE = ("test_agent_lifecycle",)


def _run(name: str) -> bool:
    script = ROOT / "scripts" / f"{name}.py"
    if not script.is_file():
        raise SystemExit(f"FAIL: missing {script}")
    print(f"== {name}", flush=True)
    env = os.environ.copy()
    env["SKIP_EVAL_LLM"] = "1"
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=False)
    if result.returncode == 77:
        return False
    result.check_returncode()
    return True


def _run_offline() -> bool:
    completed = [_run(name) for name in OFFLINE]
    passed = sum(completed)
    skipped = len(completed) - passed
    if skipped:
        print(f"offline checks finished: {passed} scripts passed, {skipped} skipped (local materials unavailable)")
    else:
        print("offline acceptance passed")
    return skipped == 0


def _run_live() -> bool:
    if (ROOT / "app" / "main.py").is_file() and not (ROOT / "backend").is_dir():
        return all([_run(name) for name in LIVE])
    script = "/app/scripts/test_agent_lifecycle.py"
    print("== test_agent_lifecycle", flush=True)
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-e",
            "PYTHONPATH=/app",
            "-T",
            "api",
            "python",
            script,
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode == 77:
        return False
    result.check_returncode()
    return True


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "offline"
    if mode == "offline":
        _run_offline()
        return
    if mode == "live":
        passed = _run_live()
        print("live acceptance passed" if passed else "live checks skipped: local materials unavailable")
        return
    if mode == "all":
        offline_passed = _run_offline()
        live_passed = _run_live()
        print("acceptance passed" if offline_passed and live_passed else "acceptance finished with skips")
        return
    raise SystemExit("usage: python scripts/accept.py [offline|live|all]")


if __name__ == "__main__":
    main()
