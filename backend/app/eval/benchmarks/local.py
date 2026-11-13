"""Provenance and scoring helpers; models load only in an authorized remote runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

from app.eval.benchmarks.data import ROOT, canonical, digest, write_json
from app.eval.benchmarks.metrics import ranking


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_archive(folder: Path) -> dict:
    files = sorted(
        p
        for base in (ROOT / "backend/app", ROOT / "scripts", ROOT / "benchmarks")
        for p in base.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".txt"}
        and "reports" not in p.parts
        and "fixtures" not in p.parts
    )
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files}
    archive = folder / "source.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            stream.write(path, path.relative_to(ROOT).as_posix())
    try:
        revision = (
            subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "git_commit": revision,
        "files": hashes,
        "source_sha256": digest(canonical(hashes)),
        "archive_sha256": sha256_file(archive),
        "archive": "source.zip",
    }


def model_inventory(path: Path) -> dict:
    if not path.is_dir() or not (path / "model.safetensors").is_file():
        raise ValueError("existing local safetensors model required; downloads forbidden")
    files = {
        p.relative_to(path).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "README.md"
    }
    return {
        "path": str(path),
        "local_snapshot_name": path.name,
        "upstream_revision": None,
        "version_kind": "content-addressed local weights; mutable master directory name is not a revision",
        "files": files,
        "model_snapshot_sha256": digest(canonical(files)),
    }


def runtime_info():
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "numpy",
                "scikit-learn",
                "sqlalchemy",
                "psycopg",
            )
        },
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "image_id": os.environ.get("BENCHMARK_IMAGE_ID"),
        "cpu_limit": os.environ.get("BENCHMARK_CPU_LIMIT"),
        "memory_limit": os.environ.get("BENCHMARK_MEMORY_LIMIT"),
    }


def resources():
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_user_seconds": value.ru_utime,
        "cpu_system_seconds": value.ru_stime,
        "peak_rss_bytes": value.ru_maxrss * 1024,
        "resource_scope": "Linux process; excludes PostgreSQL/container overhead",
    }


def new_run(kind: str, config: dict):
    folder = ROOT / "data/benchmarks/runs" / str(uuid.uuid4())
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": folder.name,
        "benchmark": kind,
        "mode": "retrieval_only",
        "status": "running",
        "execution_requested": True,
        "config": config,
        "code": source_archive(folder),
        "started_unix": time.time(),
        "generation_calls": 0,
        "remote_embedding_calls": 0,
        "paid_api_cost": 0,
        "remote_compute_cost": None,
        "storage_transfer_cost": None,
        "official_score_comparable": False,
    }
    write_json(folder / "manifest.json", manifest)
    return folder, manifest


def validate_candidate_scores(case, scores):
    contexts = case["contexts"]
    if (
        not isinstance(scores, list)
        or len(scores) != len(contexts)
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in scores
        )
    ):
        raise ValueError("incomplete or non-finite model scores")
    if set(case["relevance_labels"]) != {item["id"] for item in contexts} or len(
        {item["id"] for item in contexts}
    ) != len(contexts):
        raise ValueError("candidate IDs/labels incomplete or duplicated")


def score_candidates(case, scores):
    validate_candidate_scores(case, scores)
    contexts = case["contexts"]
    scored = [{"id": item["id"], "score": float(value)} for item, value in zip(contexts, scores)]
    ranked = sorted(scored, key=lambda x: (-x["score"], x["id"]))
    metrics = ranking([x["id"] for x in ranked], case["relevance_labels"], complete_ranking=True, scored=scored)
    from sklearn.metrics import average_precision_score, ndcg_score

    labels = [case["relevance_labels"][x["id"]] for x in ranked]
    ap = float(average_precision_score(labels, [x["score"] for x in ranked]))
    if abs(ap - metrics["map"]["value"]) > 1e-10:
        raise ValueError("MAP differs from independent sklearn implementation")
    metrics["ndcg@10_tie_averaged"] = {
        "status": "evaluated",
        "value": float(ndcg_score([labels], [[x["score"] for x in ranked]], k=10, ignore_ties=False)),
    }
    return ranked, metrics
