"""Bounded execution and comparison in data/benchmarks; no business database writes."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from app.eval.benchmarks.adapters import MODES
from app.eval.benchmarks.data import DATA_ROOT, ROOT, canonical, digest, load_snapshot, write_json
from app.eval.benchmarks.metrics import aggregate, score


def validate_config(config: dict) -> None:
    for key in ("max_requests", "max_tokens", "max_output_tokens", "max_total_output_tokens", "max_pairs_per_case"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_cost", "timeout_seconds", "input_price_per_million", "output_price_per_million"):
        value = config.get(key)
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if type(config.get("retries")) is not int or config["retries"] != 0:
        raise ValueError("initial controlled runs require retries=0, including SDK retries")
    if config.get("provider") != "qwen" or not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError(
            "explicit qwen model required; stub forbidden; other providers not yet audited for hidden retries"
        )
    if not config.get("currency") or not config.get("price_source"):
        raise ValueError("currency and price_source required; prices may be explicit conservative ceilings")


def select_cases(cases: list[dict], *, limit: int, offset: int = 0, ids: list[str] | None = None) -> list[dict]:
    if limit < 1 or offset < 0:
        raise ValueError("positive explicit limit and nonnegative offset required")
    if ids:
        if len(set(ids)) != len(ids) or set(ids) - {x["sample_id"] for x in cases}:
            raise ValueError("duplicate or unknown selected IDs")
        cases = [x for x in cases if x["sample_id"] in ids]
    selected = cases[offset : offset + limit]
    if not selected:
        raise ValueError("empty selection")
    return selected


def reservation(case: dict, config: dict) -> dict:
    if case["evaluation_mode"] == "generation_with_context":
        # Conservative byte-based reservation, not a claimed exact tokenizer count.
        input_tokens = (
            len(
                canonical(
                    {
                        "question": case["question"],
                        "contexts": case["contexts"],
                        "prompt_fields": case.get("prompt_fields"),
                    }
                )
            )
            + 4096
        )
        return {
            "requests": 1,
            "tokens": input_tokens + config["max_total_output_tokens"],
            "pairs": 0,
            "cost": (
                input_tokens * config["input_price_per_million"]
                + config["max_total_output_tokens"] * config["output_price_per_million"]
            )
            / 1_000_000,
        }
    return {"requests": 1, "tokens": 0, "cost": 0.0, "pairs": len(case.get("contexts") or [])}


def code_snapshot() -> dict:
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    hashes = {
        str(p.relative_to(ROOT)).replace("\\", "/"): digest(p.read_bytes())
        for folder in (ROOT / "backend" / "app", ROOT / "scripts")
        for p in sorted(folder.rglob("*.py"))
    }
    hashes["benchmarks/registry.json"] = digest((ROOT / "benchmarks" / "registry.json").read_bytes())
    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "source_sha256": digest(canonical(hashes)),
        "files": hashes,
    }


def supported(case: dict, mode: str) -> tuple[bool, str | None]:
    if case["evaluation_mode"] != mode or case["mode_support"].get(mode) != "supported":
        return False, "adapter/runtime does not implement requested mode"
    if case["task_type"] == "retrieval":
        return False, "DuRetrieval requires complete official corpus and isolated pgvector index; not provisioned"
    return True, None


def run(
    snapshot_path: Path,
    config: dict,
    *,
    mode: str,
    selected: list[dict],
    execute: bool,
    authorize_models: bool = False,
    root: Path = DATA_ROOT,
    worker_python: str = sys.executable,
    initial_reserved: dict | None = None,
) -> Path:
    validate_config(config)
    if mode not in MODES:
        raise ValueError("unknown mode")
    if execute and not authorize_models:
        raise ValueError("real execution requires --authorize-models; no model was called")
    snapshot, _ = load_snapshot(snapshot_path)
    originals = {canonical(x) for x in snapshot["cases"]}
    if not selected or any(canonical(x) not in originals for x in selected):
        raise ValueError("selected cases not in verified snapshot")
    folder = root / "runs" / str(uuid.uuid4())
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "run_id": folder.name,
        "benchmark": snapshot["benchmark"],
        "dataset_revision": snapshot["dataset_revision"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "selected_ids": [x["sample_id"] for x in selected],
        "mode": mode,
        "config": config,
        "code": code_snapshot(),
        "corpus_snapshot": {"type": "official_contexts_or_candidates", "sha256": snapshot["snapshot_sha256"]},
        "index_snapshot": None,
        "storage_namespace": "data/benchmarks; no application DB writes",
        "authorization": authorize_models,
        "execution_requested": execute,
        "initial_shared_reservation": initial_reserved,
        "status": "running" if execute else "prepared_not_executed",
        "official_score_comparable": False,
        "protocol_caveat": "controlled selection and Provident prompt; see benchmark protocol documentation",
    }
    from app.eval.benchmarks.local import source_archive

    archive = source_archive(folder)
    manifest["code"]["archive_sha256"] = archive["archive_sha256"]
    manifest["code"]["archive"] = archive["archive"]
    manifest["budget_policy"] = (
        "reserve worst case before request; reconcile confirmed successful usage; retain unknown failed reservation"
    )
    write_json(folder / "manifest.json", manifest)
    write_json(folder / "snapshot.json", snapshot)
    used = dict(initial_reserved or {"requests": 0, "tokens": 0, "cost": 0.0})
    if set(used) != {"requests", "tokens", "cost"} or any(
        not math.isfinite(value) or value < 0 for value in used.values()
    ):
        raise ValueError("invalid initial shared budget")
    projected = {"requests": 0, "tokens": 0, "cost": 0.0}
    observed_tokens, observed_cost, unknown_usage = 0, 0.0, False
    halt_reason = None
    results = []
    for case in selected:
        trace_id = str(uuid.uuid4())
        trace = {
            "sample_id": case["sample_id"],
            "mode": mode,
            "input": case,
            "trace_id": trace_id,
            "trace_namespace": "benchmark_local",
            "application_trace_id": None,
            "output": None,
            "status": "prepared_not_executed",
            "metrics": {},
            "badcase_types": [],
            "latency_ms": None,
            "token_usage": None,
            "cost": None,
        }
        is_supported, reason = supported(case, mode)
        if not is_supported:
            trace.update(status="not_supported", reason=reason)
        elif execute and halt_reason:
            trace.update(status="not_executed", reason=halt_reason)
        else:
            budget = reservation(case, config)
            for key in projected:
                projected[key] += budget[key]
            trace["reservation"] = budget
            reason = next(
                (
                    f"{key} budget exhausted"
                    for key, maximum in (
                        ("requests", config["max_requests"]),
                        ("tokens", config["max_tokens"]),
                        ("cost", config["max_cost"]),
                    )
                    if used[key] + budget[key] > maximum
                ),
                None,
            )
            if budget["pairs"] > config["max_pairs_per_case"]:
                reason = "Cross-Encoder pair cap exceeded"
            if not execute:
                trace["reason"] = "offline preparation only; no answer produced"
                trace["budget_preflight"] = reason or "within_limits_for_this_case"
            elif reason:
                trace.update(status="budget_stopped", reason=reason)
            else:
                for key in used:
                    used[key] += budget[key]
                write_json(folder / "manifest.json", {**manifest, "budget_reserved": used})
                input_path, output_path = folder / f"{trace_id}.request.json", folder / f"{trace_id}.response.json"
                write_json(input_path, {"case": case, "config": config, "authorize_models": True})
                started = time.perf_counter()
                try:
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(ROOT / "backend") + os.pathsep + env.get("PYTHONPATH", "")
                    child = subprocess.run(
                        [worker_python, "-m", "app.eval.benchmarks.worker", str(input_path), str(output_path)],
                        cwd=ROOT,
                        env=env,
                        capture_output=True,
                        timeout=config["timeout_seconds"],
                    )
                    if child.returncode or not output_path.exists():
                        raise RuntimeError("worker process failed; interpreter/dependency preflight required")
                    response = json.loads(output_path.read_text(encoding="utf-8"))
                    if "worker_error" in response:
                        transport_usage = response.get("transport_usage") or []
                        if transport_usage:
                            trace["transport_usage"] = transport_usage
                            trace["token_usage"] = transport_usage
                            trace["cost"] = sum(
                                (
                                    x["prompt_tokens"] * config["input_price_per_million"]
                                    + x["completion_tokens"] * config["output_price_per_million"]
                                )
                                / 1_000_000
                                for x in transport_usage
                            )
                            observed_tokens += sum(x["prompt_tokens"] + x["completion_tokens"] for x in transport_usage)
                            observed_cost += trace["cost"]
                        raise RuntimeError(
                            f"{response['worker_error']}:{response['kind']}:{response.get('missing_module')}"
                        )
                    usage = response["usage"]
                    tokens = sum(x["prompt_tokens"] + x["completion_tokens"] for x in usage)
                    cost = sum(
                        (
                            x["prompt_tokens"] * config["input_price_per_million"]
                            + x["completion_tokens"] * config["output_price_per_million"]
                        )
                        / 1_000_000
                        for x in usage
                    )
                    observed_tokens += tokens
                    observed_cost += cost
                    trace.update(
                        output=response["output"],
                        token_usage=usage,
                        cost=cost,
                        transport_usage=response.get("transport_usage"),
                        effective_config=response["effective_config"],
                        stages_ms=response["stages_ms"],
                        timing_meta=response["timing_meta"],
                    )
                    if (
                        tokens > budget["tokens"]
                        or cost > budget["cost"]
                        or len(usage) > budget["requests"]
                        or any(x["completion_tokens"] > config["max_total_output_tokens"] for x in usage)
                    ):
                        raise RuntimeError("provider usage exceeded conservative reservation; stop for review")
                    # Release only usage proven unused by a successful response. Unknown failures retain the full hold.
                    used["tokens"] -= budget["tokens"] - tokens
                    used["cost"] -= budget["cost"] - cost
                    trace["inference_status"] = "completed"
                    try:
                        trace["metrics"], trace["badcase_types"] = score(case, trace["output"])
                        trace["scoring_status"] = "scored"
                    except (ValueError, KeyError, ArithmeticError) as exc:
                        trace.update(
                            scoring_status="failed",
                            scoring_error=f"{type(exc).__name__}: {exc}",
                            badcase_types=["scoring_failure"],
                        )
                        halt_reason = "stopped_after_scoring_failure; actual model output preserved"
                    trace["status"] = "executed"
                except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                    unknown_usage = unknown_usage or trace["token_usage"] is None
                    halt_reason = "stopped_after_failed_request; inspect actual response/usage before continuing"
                    # Reserve worst-case usage for failures; never retry potentially accepted requests.
                    trace.update(
                        status="error",
                        reason="hard timeout; child terminated"
                        if isinstance(exc, subprocess.TimeoutExpired)
                        else str(exc),
                        badcase_types=["execution_failure"],
                    )
                finally:
                    trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if trace["badcase_types"]:
            trace["badcase_status"] = "candidate_badcase"
        results.append(trace)
        write_json(folder / "traces" / f"{trace_id}.json", trace)
        write_json(folder / "results.json", results)
        if unknown_usage:
            # Stop all subsequent provider requests when a failed attempt's usage is unknown.
            # Remaining cases are explicitly budget-stopped and never disappear from denominators.
            halt_reason = "stopped_after_failed_request; usage may be unknown; budget is not necessarily exhausted"
    summary = aggregate(results)
    scoring_errors = sum(x.get("scoring_status") == "failed" for x in results)
    summary["scoring_errors"] = scoring_errors
    status = (
        ("completed_with_failures" if summary["executed"] != len(results) or scoring_errors else "completed")
        if execute
        else "prepared_not_executed"
    )
    manifest.update(
        status=status,
        budget_reserved=used,
        observed_tokens=observed_tokens,
        priced_observed_cost=observed_cost,
        projected_reservation=projected,
        cost_status="unknown_failed_attempt_usage" if unknown_usage else "configured_price_estimate",
        actual_invoice_cost=None,
    )
    write_json(folder / "manifest.json", manifest)
    write_json(folder / "summary.json", {**summary, "run_status": manifest["status"]})
    write_json(folder / "badcases.json", [x for x in results if x.get("badcase_status") == "candidate_badcase"])
    report = [
        f"# {snapshot['benchmark']} / {folder.name}",
        f"Mode: {mode}",
        f"Status: {manifest['status']}; selected {len(selected)}; executed {summary['executed']}; errors {summary['errors']}.",
        "Official comparable score: false. Process completion is not benchmark success.",
        "Fixed contexts do not measure end-to-end retrieval. Candidate reranking does not measure corpus retrieval.",
        f"Observed LLM tokens: {observed_tokens}; price estimate: {observed_cost} {config['currency']}.",
        f"Cost status: {manifest['cost_status']}; invoice cost unknown.",
        f"Candidate badcases: {summary['candidate_badcases']}. See badcases.json and traces/.",
        "Per-metric eligibility and denominators: summary.json; raw input/output: results.json.",
    ]
    (folder / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    return folder


def compare(left: Path, right: Path) -> dict:
    a, b = (json.loads((p / "manifest.json").read_text(encoding="utf-8")) for p in (left, right))
    for key in ("snapshot_sha256", "dataset_revision", "selected_ids", "mode", "corpus_snapshot", "index_snapshot"):
        if a[key] != b[key]:
            raise ValueError(f"incomparable runs: {key} differs")
    if not a["execution_requested"] or not b["execution_requested"]:
        raise ValueError("prepared runs contain no model results")
    ra, rb = (json.loads((p / "results.json").read_text(encoding="utf-8")) for p in (left, right))
    good_a = [x["sample_id"] for x in ra if x["status"] == "executed"]
    good_b = [x["sample_id"] for x in rb if x["status"] == "executed"]
    if not good_a or good_a != good_b:
        raise ValueError("empty or different executed populations")
    sa, sb = (json.loads((p / "summary.json").read_text(encoding="utf-8")) for p in (left, right))
    deltas = {
        key: sb["metrics"][key]["value"] - sa["metrics"][key]["value"]
        for key in sa["metrics"].keys() & sb["metrics"].keys()
        if sa["metrics"][key]["status"] == sb["metrics"][key]["status"] == "evaluated"
    }
    return {
        "left": a["run_id"],
        "right": b["run_id"],
        "metric_deltas": deltas,
        "left_summary": sa,
        "right_summary": sb,
        "official_score_comparable": False,
    }
