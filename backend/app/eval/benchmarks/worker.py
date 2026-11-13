"""Per-case subprocess using production functions; no application DB or trace writes."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


def execute(request: dict, *, transport_calls: list[dict] | None = None) -> dict:
    if request.get("authorize_models") is not True:
        raise ValueError("explicit model authorization required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # app.models imports app.db.session eagerly. Fence even that lazily-created engine away from production.
    os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
    os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
    from app.core.config import get_settings
    from app.generate.prompt import PROMPT_VERSION, SYSTEM_PROMPT, EvidenceItem, build_prompt
    from app.llm.ledger import capture_calls
    from app.search.stages import reset_stages, snapshot_meta, snapshot_stages

    config, case = request["config"], request["case"]
    transport_calls = transport_calls if transport_calls is not None else []
    from app.eval.benchmarks.transport import bounded_client

    settings = get_settings()
    settings.llm_max_retries = 0
    settings.llm_max_output_tokens = config["max_output_tokens"]
    settings.llm_connect_timeout_seconds = min(5, config["timeout_seconds"])
    settings.llm_read_timeout_seconds = config["timeout_seconds"]
    settings.llm_write_timeout_seconds = min(10, config["timeout_seconds"])
    settings.answer_llm_provider, settings.answer_llm_model = config["provider"], config["model"]
    public = {
        key: getattr(settings, key, None)
        for key in (
            "embedding_model_name",
            "embedding_model_ref",
            "rerank_model_name",
            "rerank_model_ref",
            "rerank_device",
            "rerank_batch_size",
            "llm_temperature",
            "verify_entail_threshold",
            "verify_lexical_gate",
            "verify_lexical_threshold",
        )
    }
    calls = []
    reset_stages()
    with capture_calls(calls):
        if case["benchmark"] == "crud":
            from app.eval.benchmarks.data import digest, verified_raw
            from app.llm import get_llm_client

            prompt_path, source = verified_raw("crud", case["prompt_file"])
            template = prompt_path.read_text(encoding="utf-8")
            prompt = template.format(
                **case["prompt_fields"], search_documents="\n\n".join(x["text"] for x in case["contexts"])
            )
            response = bounded_client(
                settings, total_output_tokens=config["max_total_output_tokens"], calls=transport_calls
            ).generate(
                system="",
                prompt=prompt,
                max_output_tokens=config["max_output_tokens"],
                temperature=settings.llm_temperature,
                purpose="benchmark_crud",
            )
            output = {
                "answer": response.text.split("<response>")[-1].split("</response>")[0].strip(),
                "raw_answer": response.text,
                "usage": [response.usage.to_dict()],
                "prompt": {"system": "", "user": prompt, "version": source["sha256"], "source": source["url"]},
                "pipeline": "official CRUD task prompt via Provident model adapter; not Agent digest/synthesis",
            }
        elif case["evaluation_mode"] == "generation_with_context":
            from app.generate.service import answer_from_evidence

            evidence = [
                EvidenceItem(index=i, chunk_id=item["id"], doc_id=item["id"], page=1, text=item["text"])
                for i, item in enumerate(case["contexts"], 1)
            ]
            # Supply every selected official context; no silent application context truncation.
            settings.qa_max_context_chars = sum(len(x.text) + 512 for x in evidence) + 4096
            prompt = build_prompt(case["question"], evidence, max_context_chars=settings.qa_max_context_chars)
            with patch(
                "app.generate.service.get_llm_client",
                lambda s: bounded_client(
                    s, total_output_tokens=config["max_total_output_tokens"], calls=transport_calls
                ),
            ):
                output = asdict(answer_from_evidence(case["question"], evidence, settings=settings))
            if not output["usage"] or any(x["provider"] == "stub" for x in output["usage"]):
                raise ValueError("real provider usage required; stub forbidden")
            output["prompt"] = {"version": PROMPT_VERSION, "system": SYSTEM_PROMPT, "user": prompt}
        elif case["task_type"] == "reranking":
            if not Path(settings.rerank_model_ref).is_dir():
                raise FileNotFoundError("local reranker snapshot required; automatic downloads disabled")
            from app.rerank.model import rerank_scores

            contexts = case["contexts"]
            scores = rerank_scores(case["question"], [x["text"] for x in contexts], settings=settings)
            if len(scores) != len(contexts):
                raise ValueError("reranker score count mismatch")
            ranked = sorted(zip(contexts, scores), key=lambda pair: (-pair[1], pair[0]["id"]))
            output = {
                "ranked_ids": [x["id"] for x, _ in ranked],
                "scores": [{"id": x["id"], "score": float(value)} for x, value in ranked],
                "candidate_scope": case["candidate_scope"],
            }
        else:
            raise ValueError("worker mode not_supported")
    if case["evaluation_mode"] == "generation_with_context":
        output["transport_policy"] = {
            "max_tokens_replaced_with": "max_completion_tokens",
            "max_completion_tokens": config["max_total_output_tokens"],
            "thinking_mode": "unchanged provider default",
        }
    return {
        "output": output,
        "usage": calls,
        "transport_usage": transport_calls,
        "effective_config": public,
        "stages_ms": snapshot_stages(),
        "timing_meta": snapshot_meta(),
    }


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    transport_calls = []
    try:
        result = execute(request, transport_calls=transport_calls)
    except Exception as exc:
        # No provider bodies/credentials in reports. Known missing dependencies are named.
        result = {
            "worker_error": type(exc).__name__,
            "kind": getattr(exc, "kind", "worker_failure"),
            "transport_usage": transport_calls,
            "missing_module": exc.name if isinstance(exc, ModuleNotFoundError) else None,
        }
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
