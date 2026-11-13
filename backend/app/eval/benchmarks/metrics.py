"""Public benchmark metrics. Missing labels and undefined ratios are not scores of zero."""

from __future__ import annotations

import math


def metric(value=None, reason: str | None = None, **meta) -> dict:
    return {
        "status": "not_evaluable" if value is None else "evaluated",
        "value": value,
        **({"reason": reason} if reason else {}),
        **meta,
    }


def rgb_match(answer: str, reference) -> bool:
    """Align RGB evalue.checkanswer: AND over outer entries, OR over nested aliases."""
    prediction = answer.replace(" ", "").lower()
    answers = reference if isinstance(reference, list) else [reference]
    if not answers:
        raise ValueError("empty reference")
    return all(
        any(alias.lower() in prediction for alias in item) if isinstance(item, list) else item.lower() in prediction
        for item in answers
    )


def ranking(
    ranked: list[str],
    qrels: dict[str, float] | None,
    *,
    complete_ranking: bool = False,
    scored: list[dict] | None = None,
) -> dict:
    names = ("recall@5", "recall@10", "mrr@10", "ndcg@10", "map")
    if not qrels or not any(x > 0 for x in qrels.values()):
        return {name: metric(reason="missing positive relevance labels") for name in names}
    if len(set(ranked)) != len(ranked):
        raise ValueError("duplicate ranked IDs")
    positive = {key for key, value in qrels.items() if value > 0}
    first = next((i for i, key in enumerate(ranked[:10], 1) if key in positive), None)

    def dcg(values):
        return sum(value / math.log2(i + 2) for i, value in enumerate(values))

    # BEIR/trec_eval uses linear gain. C-MTEB selected datasets have binary relevance.
    ideal = dcg(sorted(qrels.values(), reverse=True)[:10])
    hits = 0
    ap = 0.0
    for i, key in enumerate(ranked, 1):
        if key in positive:
            hits += 1
            ap += hits / i
    if scored is not None and complete_ranking:
        # Match sklearn average_precision_score used by MTEB: equal scores share one threshold.
        groups = {}
        for item in scored:
            value = item["score"]
            if not math.isfinite(value):
                raise ValueError("non-finite reranking score")
            groups.setdefault(value, []).append(item["id"])
        hits, seen, ap = 0, 0, 0.0
        for value in sorted(groups, reverse=True):
            group = groups[value]
            found = sum(key in positive for key in group)
            hits += found
            seen += len(group)
            ap += found * hits / seen
    return {
        **{f"recall@{k}": metric(len(set(ranked[:k]) & positive) / len(positive)) for k in (5, 10)},
        "mrr@10": metric(1 / first if first else 0.0),
        "ndcg@10": metric(dcg([qrels.get(key, 0) for key in ranked[:10]]) / ideal),
        "map": metric(
            ap / len(positive) if complete_ranking else None,
            "complete candidate ranking required" if not complete_ranking else None,
        ),
    }


def score(case: dict, output: dict) -> tuple[dict, list[str]]:
    scores = {
        "answer_correctness": metric(reason="semantic correctness judge not enabled"),
        "citation_precision": metric(reason="no gold citations aligned with actual output"),
        "citation_coverage": metric(reason="no gold claim/evidence alignment for actual output"),
        "unsupported_f1": metric(reason="selected benchmarks do not provide aligned sentence support labels"),
        "multi_turn_correctness": metric(reason="no official multi-turn tasks in selected V1 subsets"),
    }
    failures = []
    if case["task_type"] in {"reranking", "retrieval"}:
        scores.update(
            ranking(
                output["ranked_ids"],
                case["relevance_labels"],
                complete_ranking=case["task_type"] == "reranking",
                scored=output.get("scores"),
            )
        )
        if not any(case["relevance_labels"].get(key, 0) > 0 for key in output["ranked_ids"][:10]):
            failures.append("retrieval_miss_at_10")
    elif case["benchmark"] == "rgb":
        answer = output["answer"]
        matched = rgb_match(answer, case["reference_answers"])
        refusal = "现有证据不足以回答" in answer
        official_refusal = "信息不足" in answer.replace(" ", "") or "insufficient information" in answer
        scores["rgb_reference_match"] = metric(
            float(matched), protocol="official substring proxy, not semantic accuracy"
        )
        scores["rgb_official_all_rate_item"] = metric(
            float(official_refusal if case["answerable"] is False else matched and not official_refusal)
        )
        scores["refusal_accuracy"] = metric(
            float(refusal) if case["answerable"] is False else None,
            "noise-only cases required" if case["answerable"] is not False else None,
        )
        scores["false_refusal_rate"] = metric(
            float(refusal) if case["answerable"] is True else None,
            "answerability label unavailable" if case["answerable"] is not True else None,
        )
        if case["task_type"] == "counterfactual":
            scores["rgb_error_detection_phrase"] = metric(float("事实性错误" in answer or "factual errors" in answer))
        if case["answerable"] is False and not refusal:
            failures.append("refusal_not_detected")
        elif case["answerable"] is True and refusal:
            failures.append("false_refusal")
        elif case["answerable"] is not False and not matched:
            failures.append("reference_answer_mismatch")
    elif case["benchmark"] == "crud":
        scores.update(crud_overlap(output["answer"], case["reference_answers"]))
        if not output["answer"].strip():
            failures.append("empty_answer")
    return scores, failures


def crud_overlap(answer: str, reference: str) -> dict:
    """Align official common.py: Jieba tokens, BLEU without brevity penalty; ROUGE-L F1."""
    if not reference or "request openai failed" in reference:
        return {
            key: metric(reason="official reference is empty/provider-error artifact")
            for key in ("crud_bleu_no_bp", "crud_rougeL")
        }
    try:
        import jieba
    except ImportError:
        return {
            key: metric(reason="optional pinned Jieba tokenizer not installed")
            for key in ("crud_bleu_no_bp", "crud_rougeL")
        }
    from collections import Counter

    predicted, expected = list(jieba.cut(answer)), list(jieba.cut(reference))
    precisions = []
    for n in range(1, 5):
        left = Counter(tuple(predicted[i : i + n]) for i in range(max(0, len(predicted) - n + 1)))
        right = Counter(tuple(expected[i : i + n]) for i in range(max(0, len(expected) - n + 1)))
        denominator = sum(left.values())
        precisions.append(sum((left & right).values()) / denominator if denominator else 0.0)
    bleu = math.exp(sum(math.log(x) for x in precisions) / 4) if all(precisions) else 0.0
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for j, other in enumerate(expected, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    rouge = 2 * lcs / (len(predicted) + len(expected)) if predicted or expected else 0.0
    protocol = "CRUD common.py formula alignment; textual overlap only; not semantic correctness"
    return {"crud_bleu_no_bp": metric(bleu, protocol=protocol), "crud_rougeL": metric(rouge, protocol=protocol)}


def percentile(values: list[float], p: float):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def aggregate(results: list[dict]) -> dict:
    executed = [x for x in results if x["status"] == "executed"]
    names = sorted(
        {key for x in executed for key in x["metrics"]}
        | {
            "answer_correctness",
            "citation_precision",
            "citation_coverage",
            "unsupported_f1",
            "multi_turn_correctness",
            "recall@5",
            "recall@10",
            "mrr@10",
            "ndcg@10",
            "map",
        }
    )
    scores = {}
    for name in names:
        values = [
            x["metrics"][name]["value"]
            for x in executed
            if name in x["metrics"] and x["metrics"][name]["status"] == "evaluated"
        ]
        scores[name] = metric(
            sum(values) / len(values) if values else None,
            "no eligible labels/output; see traces" if not values else None,
            evaluated_samples=len(values),
            selected_samples=len(results),
        )
    latency = [x["latency_ms"] for x in executed]
    return {
        "selected": len(results),
        "executed": len(executed),
        "metrics": scores,
        "errors": sum(x["status"] == "error" for x in results),
        "not_supported": sum(x["status"] == "not_supported" for x in results),
        "budget_stopped": sum(x["status"] == "budget_stopped" for x in results),
        "not_executed": sum(x["status"] in {"not_executed", "prepared_not_executed"} for x in results),
        "candidate_badcases": sum(bool(x["badcase_types"]) for x in results),
        "p50_ms": percentile(latency, 0.5),
        "p95_ms": percentile(latency, 0.95),
        "latency_scope": "successful case wall time, including imports and model loading",
    }
