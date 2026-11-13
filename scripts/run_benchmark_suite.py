"""Execute a reviewed plan with one shared request/token/cost budget across all runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.eval.benchmarks.data import load_snapshot
from app.eval.benchmarks.runner import run, select_cases, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--authorize-models", action="store_true")
    args = parser.parse_args()
    if not args.authorize_models:
        parser.error("--authorize-models is required; no model was called")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    config = json.loads((ROOT / plan["config"]).read_text(encoding="utf-8"))
    validate_config(config)
    # Validate the complete reviewable selection before the first request.
    jobs = []
    for job in plan["runs"]:
        path = ROOT / job["snapshot"]
        _, cases = load_snapshot(path)
        jobs.append(
            (
                path,
                job["mode"],
                select_cases(cases, limit=job["limit"], offset=job.get("offset", 0), ids=job.get("ids")),
            )
        )
    used = plan.get("initial_reserved", {"requests": 0, "tokens": 0, "cost": 0.0})
    failed = False
    for path, mode, cases in jobs:
        folder = run(
            path, config, mode=mode, selected=cases, execute=True, authorize_models=True, initial_reserved=used
        )
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        used = manifest["budget_reserved"]
        print(folder, flush=True)
        if manifest["status"] != "completed":
            failed = True
        if manifest["cost_status"] == "unknown_failed_attempt_usage" or manifest["status"] != "completed":
            break  # A provider may have accepted a failed/timed-out request; stop the suite.
    print(json.dumps({"shared_reserved": used, "currency": config["currency"], "execution_failures": failed}))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
