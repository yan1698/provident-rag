"""Reproducible V1 Chinese adapter/preflight pass. No downloads, model calls or database access."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.eval.benchmarks.data import DATA_ROOT, convert, load_snapshot, registry, write_json
from app.eval.benchmarks.runner import run


def main():
    config = json.loads((ROOT / "benchmarks/controlled.example.json").read_text())
    jobs = [
        ("rgb_noise_0", "rgb", "data/zh_refine.json", {"noise_rate": 0}),
        ("rgb_noise_06", "rgb", "data/zh_refine.json", {"noise_rate": 0.6}),
        ("rgb_refusal", "rgb", "data/zh_refine.json", {"noise_rate": 1}),
        ("rgb_integration", "rgb", "data/zh_int.json", {"noise_rate": 0}),
        ("rgb_counterfactual", "rgb", "data/zh_fact.json", {"noise_rate": 0, "correct_rate": 0}),
    ]
    for task in (
        "questanswer_1doc",
        "questanswer_2docs",
        "questanswer_3docs",
        "event_summary",
        "continuing_writing",
        "hallu_modified",
    ):
        jobs.append((f"crud_{task}", "crud", "data/crud_split/split_merged.json", {"task": task}))
    jobs += [
        ("DuRetrieval", "cmteb_du", registry("cmteb_du")["files"][1], {}),
        ("T2Reranking", "cmteb_t2", registry("cmteb_t2")["files"][1], {}),
    ]
    report = {
        "date": datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
        "scope": "Provident V1 Chinese",
        "real_model_calls": 0,
        "real_model_samples": 0,
        "model_cost": 0,
        "results": [],
        "semantic_accuracy": "not_evaluable: no real model execution or judge",
        "official_score_comparable": False,
    }
    for label, name, source, options in jobs:
        entry = {"task": label, "benchmark": name, "revision": registry(name)["revision"]}
        try:
            pointer = DATA_ROOT / "t2-controlled-sample.json"
            if name == "cmteb_t2" and pointer.exists():
                ref = json.loads(pointer.read_text(encoding="utf-8"))
                path = DATA_ROOT / ref["snapshot"]
                candidate, _ = load_snapshot(path)
                if (
                    candidate["snapshot_sha256"] != ref["snapshot_sha256"]
                    or candidate["dataset_revision"] != registry(name)["revision"]
                ):
                    raise ValueError("controlled range sample pointer mismatch")
            else:
                path = convert(name, source, limit=20, **options)
            snapshot, cases = load_snapshot(path)
            mode = cases[0]["evaluation_mode"]
            folder = run(path, config, mode=mode, selected=cases, execute=False)
            entry.update(
                status="validated_and_prepared",
                samples=len(cases),
                mode=mode,
                snapshot=str(path.relative_to(ROOT)).replace("\\", "/"),
                snapshot_sha256=snapshot["snapshot_sha256"],
                preparation_run=str(folder.relative_to(ROOT)).replace("\\", "/"),
                summary=json.loads((folder / "summary.json").read_text()),
            )
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            entry.update(status="blocked", error=f"{type(exc).__name__}: {exc}", samples=0)
        report["results"].append(entry)
        print(label, entry["status"], entry["samples"], flush=True)
    write_json(ROOT / "benchmarks/reports/preflight.json", report)
    pilot, pending = [], []
    for label, limit in (("rgb_noise_06", 10), ("crud_questanswer_1doc", 10), ("T2Reranking", 20)):
        entry = next(x for x in report["results"] if x["task"] == label)
        if entry["status"] == "validated_and_prepared":
            job = {"snapshot": entry["snapshot"], "mode": entry["mode"], "limit": limit}
            if label == "T2Reranking":
                pending.append(
                    {**job, "prerequisite": "compatible Cross-Encoder runtime and explicit model authorization"}
                )
            else:
                pilot.append(job)
    write_json(
        ROOT / "benchmarks/controlled_plan.json",
        {
            "config": "benchmarks/controlled.example.json",
            "runs": pilot,
            "pending_runs": pending,
            "note": "Explicit authorization required. This file selects official samples, not generated questions.",
        },
    )


if __name__ == "__main__":
    main()
