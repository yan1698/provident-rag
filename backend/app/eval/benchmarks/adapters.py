"""Chinese benchmark adapters: preserve official rows, IDs, splits and label provenance."""

from __future__ import annotations

import math
import random
import re
from copy import deepcopy

MODES = {"retrieval_only", "generation_with_context", "verifier_only", "end_to_end_rag", "multi_turn_rag"}


def base(name: str, row: dict, index: int, spec: dict, relative: str) -> dict:
    original_id = row.get("id")
    return {
        "benchmark": name,
        "original_sample_id": original_id,
        "sample_id": str(original_id) if original_id is not None else f"{relative}#row={index}",
        "id_kind": "official" if original_id is not None else "source_row_locator",
        "source_row": index,
        "source_url": spec["source"],
        "source_file": relative,
        "dataset_revision": spec["revision"],
        "official_split": None,
        "question": None,
        "contexts": None,
        "reference_answers": None,
        "relevance_labels": None,
        "evidence_labels": None,
        "dialogue_context": None,
        "answerable": None,
        "mode_support": spec["modes"],
        "label_provenance": {"description": spec["label_provenance"], "human_confirmed_by_provident": False},
        "original": deepcopy(row),
    }


def rgb_contexts(row: dict, variant: str, options: dict) -> list[str]:
    """Align official processdata using a local per-case RNG; raw data stays immutable."""
    noise, correct = options.get("noise_rate", 0.0), options.get("correct_rate", 0.0)
    count = options.get("passage_num", 5)
    if not 0 <= noise <= 1 or not 0 <= correct <= 1 or noise + correct > 1 or count < 1:
        raise ValueError("invalid RGB noise/correct/passages")
    if correct and variant != "zh_fact":
        raise ValueError("correct_rate only applies to zh_fact")
    rng = random.Random(options.get("seed", 2333))
    positives, negatives = deepcopy(row["positive"]), row["negative"]
    neg_num = math.ceil(count * noise)
    pos_num = count - neg_num
    if variant == "zh_int":
        if noise == 1:
            raise ValueError("official integration selection does not implement noise-only refusal")
        for group in positives:
            if not isinstance(group, list) or not group:
                raise ValueError("empty positive evidence group")
            rng.shuffle(group)
        docs = [group[0] for group in positives]
        if len(docs) < pos_num:
            for i in range(1, max(map(len, positives))):
                for group in positives:
                    if len(group) > i:
                        docs.append(group[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        docs += negatives[: max(0, count - len(docs))]
    elif variant == "zh_fact":
        correct_num = math.ceil(count * correct)
        pos_num = count - neg_num - correct_num
        if pos_num < 0 or len(row["positive_wrong"]) != len(positives):
            raise ValueError("invalid counterfactual context alignment")
        indices = list(range(len(positives)))
        selected = rng.sample(indices, min(len(indices), pos_num))
        docs = [row["positive_wrong"][i] for i in selected]
        remaining = [i for i in indices if i not in selected]
        if correct_num and remaining:
            docs += [positives[i] for i in rng.sample(remaining, min(len(remaining), correct_num))]
        docs += negatives[:neg_num]
    else:
        if noise != 1:
            if neg_num > len(negatives):
                neg_num, pos_num = len(negatives), count - len(negatives)
            elif pos_num > len(positives):
                pos_num, neg_num = len(positives), count - len(positives)
        docs = positives[:pos_num] + negatives[:neg_num]
    rng.shuffle(docs)
    if not docs or any(not isinstance(x, str) or not x.strip() for x in docs):
        raise ValueError("empty RGB selected contexts")
    return docs


def adapt(name: str, rows: list[dict], *, relative: str, spec: dict, options: dict) -> list[dict]:
    if name == "crud":
        if not isinstance(rows, dict) or not options.get("task") or options["task"] not in rows:
            raise ValueError("CRUD conversion requires an official task name from the source object")
        task = options["task"]
        group = rows[task]
        if options.get("limit") is not None:
            if options["limit"] < 1:
                raise ValueError("positive limit required")
            group = group[: options["limit"]]
        sources = {r["ID"]: r for r in rows["event_summary"]}
        return [
            adapt_crud(base(name, row, index, spec, relative), row, task=task, sources=sources)
            for index, row in enumerate(group)
        ]
    cases = []
    for index, row in enumerate(rows):
        case = base(name, row, index, spec, relative)
        if name == "rgb":
            variant = relative.rsplit("/", 1)[-1].removesuffix(".json")
            if variant not in {"zh_refine", "zh_int", "zh_fact"}:
                raise ValueError("only official Chinese RGB variants supported")
            docs = rgb_contexts(row, variant, options)
            case.update(
                question=row["query"],
                reference_answers=row["answer"],
                contexts=[{"id": f"context:{i}", "text": text} for i, text in enumerate(docs, 1)],
                task_type="counterfactual"
                if variant == "zh_fact"
                else (
                    "integration"
                    if variant == "zh_int"
                    else ("refusal" if options.get("noise_rate") == 1 else "noise_robustness")
                ),
                evaluation_mode="generation_with_context",
                variant=variant,
                answerable=False
                if options.get("noise_rate") == 1 and variant == "zh_refine"
                else (None if variant == "zh_fact" else True),
                context_protocol={
                    "name": "rgb-processdata",
                    "seed": options.get("seed", 2333),
                    "noise_rate": options.get("noise_rate", 0.0),
                    "correct_rate": options.get("correct_rate", 0.0),
                    "passage_num": options.get("passage_num", 5),
                    "actual": len(docs),
                },
            )
        elif name == "cmteb_t2":
            positive, negative = row["positive"], row["negative"]
            if not positive or not negative:
                raise ValueError("T2 requires positive and negative candidates")
            # Sort by text hash: the scorer never receives positive/negative input order or labels.
            from app.eval.benchmarks.data import digest

            by_id = {}
            qrels = {}
            for label, texts in ((1, positive), (0, negative)):
                for text in texts:
                    # Keep every official candidate, including repeated text. Never collapse AP denominators.
                    key = digest(text.encode("utf-8")) + f":candidate={len(by_id)}"
                    by_id[key] = {"id": key, "text": text}
                    qrels[key] = label
            case.update(
                question=row["query"],
                official_split="dev",
                task_type="reranking",
                evaluation_mode="retrieval_only",
                contexts=[by_id[key] for key in sorted(by_id)],
                relevance_labels=qrels,
                candidate_scope="official_per_query_candidates",
            )
        elif name == "cmteb_du":
            case.update(
                question=row["text"],
                official_split="dev",
                source_split="queries",
                task_type="retrieval",
                evaluation_mode="retrieval_only",
                contexts=None,
                candidate_scope="official_full_corpus_required",
            )
        else:
            raise ValueError("unregistered benchmark")
        cases.append(case)
    return cases


def adapt_crud(case: dict, row: dict, *, task: str, sources: dict) -> dict:
    case.update(
        task_type=task,
        official_split="crud_split (paper experiment subset)",
        evaluation_mode="generation_with_context",
        original_sample_id=row["ID"],
        sample_id=f"{task}#row={case['source_row']}",
        id_kind="official_task_and_row_locator",
        source_document_ids=[row["ID"]],
        context_protocol="official_source_text (oracle context); no retrieval",
    )
    if task.startswith("questanswer_"):
        count = int(task.removeprefix("questanswer_").split("doc")[0])
        case.update(
            question=row["questions"],
            reference_answers=row["answers"],
            contexts=[{"id": f"{row['ID']}:news{i}", "text": row[f"news{i}"]} for i in range(1, count + 1)],
            prompt_file="src/prompts/quest_answer.txt",
            prompt_fields={"question": row["questions"]},
        )
    elif task == "event_summary":
        case.update(
            question=row["event"],
            reference_answers=row["summary"],
            contexts=[{"id": row["ID"], "text": row["text"]}],
            prompt_file="src/prompts/summary.txt",
            prompt_fields={"event": row["event"]},
        )
    elif task == "continuing_writing":
        if row["ID"] not in sources:
            raise ValueError("continuation has no official source article; reference must not become evidence")
        source = sources[row["ID"]]
        case.update(
            question=row["beginning"],
            reference_answers=row["continuing"],
            contexts=[{"id": row["ID"], "text": source["text"]}],
            context_source_join={"task": "event_summary", "ID": row["ID"]},
            prompt_file="src/prompts/continue_writing.txt",
            prompt_fields={"beginning_text": row["beginning"]},
        )
    elif task == "hallu_modified":
        case.update(
            question=row["newsBeginning"],
            reference_answers=row["hallucinatedMod"],
            contexts=[{"id": row["ID"], "text": row["newsBeginning"] + row["newsRemainder"]}],
            prompt_file="src/prompts/hallu_mod.txt",
            prompt_fields={"begin": row["newsBeginning"], "hallu_continue": row["hallucinatedContinuation"]},
        )
        case["label_provenance"]["generation_model"] = row.get("generatedBy")
    else:
        raise ValueError(f"CRUD task not_supported: {task}")
    return case


def validate_cases(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("empty case selection")
    seen = set()
    for case in cases:
        key = (case["benchmark"], case["source_file"], case["sample_id"])
        if key in seen:
            raise ValueError(f"duplicate id: {key}")
        seen.add(key)
        if not re.fullmatch(r"[a-f0-9]{40}", case["dataset_revision"]):
            raise ValueError("immutable dataset revision required")
        if not case["source_url"].startswith("https://") or not case["label_provenance"]:
            raise ValueError("missing provenance")
        if case["evaluation_mode"] not in MODES:
            raise ValueError("unknown mode")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"empty official question: {key}")
        contexts = case["contexts"]
        if case["task_type"] != "retrieval":
            if not contexts or any(not isinstance(x["text"], str) or not x["text"].strip() for x in contexts):
                raise ValueError(f"missing contexts: {key}")
            if len({x["id"] for x in contexts}) != len(contexts):
                raise ValueError("duplicate context ids")
        if case["benchmark"] == "rgb" and not case["reference_answers"]:
            raise ValueError("missing RGB reference")
        if case["task_type"] == "reranking" and not set(case["relevance_labels"]) == {x["id"] for x in contexts}:
            raise ValueError("incomplete candidate labels")
    return {"status": "valid", "samples": len(cases), "model_calls": 0}
