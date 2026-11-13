"""Read-only runtime preflight. Creates a client with a dummy key; never sends model requests."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
    os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
    from app.core.config import get_settings
    from app.eval.benchmarks.data import write_json
    from app.generate.service import answer_from_evidence
    from app.llm import get_llm_client

    s = get_settings()
    report = {
        "model_requests": 0,
        "database": "in-memory SQLite only; no production connection",
        "official_dashscope_endpoint": urlparse(s.qwen_base_url).hostname == "dashscope.aliyuncs.com",
        "provider": s.answer_llm_provider,
        "model": s.answer_llm_model,
        "credential_present": bool(s.dashscope_api_key.strip()),
        "cross_encoder_runtime_installed": importlib.util.find_spec("sentence_transformers") is not None,
        "local_reranker_directory_exists": Path(s.rerank_model_ref).is_dir(),
        "packages": {},
    }
    for name in ("openai", "pydantic-settings", "sqlalchemy", "pyarrow", "jieba"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        client = get_llm_client(s.model_copy(update={"dashscope_api_key": "preflight-dummy-non-secret"}))
        report["client_constructor"] = type(client).__name__
        report["generation_runtime_status"] = "ready_for_authorized_pilot"
    except Exception as exc:
        report["generation_runtime_status"] = "blocked"
        # Dummy key only; message can diagnose SDK/runtime compatibility without exposing a real key.
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_json(ROOT / "benchmarks/reports/runtime-preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
