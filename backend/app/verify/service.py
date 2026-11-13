from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.generate.prompt import EvidenceItem
from app.verify.claims import Claim, split_claims
from app.verify.meta import STATUS_META, meta_claim_checks, module_coverage_rate, task_coverage_rate


def pair_scores(*args, **kwargs):
    # Loading API contracts and replaying saved answers must not load torch.
    from app.rerank.model import pair_scores as score

    return score(*args, **kwargs)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "be",
    "as",
    "at",
    "by",
    "with",
    "that",
    "this",
    "it",
    "的",
    "是",
    "和",
    "与",
    "在",
    "了",
    "也",
    "都",
    "等",
    "为",
    "对",
    "及",
    "其",
    "或",
    "把",
    "被",
    "从",
    "到",
    "中",
}
_INSUFFICIENT_RE = re.compile(r"现有证据不足以回答")

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_UNCITED = "uncited"
STATUS_INVALID_CITATION = "invalid_citation"
STATUS_INSUFFICIENT = "insufficient"
STATUS_UNVERIFIED = "unverified"

_HALLUCINATION_STATUSES = {STATUS_UNSUPPORTED, STATUS_UNCITED, STATUS_INVALID_CITATION}


@dataclass
class ClaimCheck:
    text: str
    citation_indices: list[int]
    lexical_coverage: float
    entailment_score: float | None
    status: str
    hallucination: bool
    source_gaps: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    claim_count: int
    supported_count: int
    hallucination_count: int
    coverage_rate: float
    hallucination_rate: float
    claims: list[ClaimCheck]
    task_coverage: float | None = None
    unverified_count: int = 0


_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|"
    r"[A-Za-z_][A-Za-z0-9_]*(?=\s*\())(?![A-Za-z0-9_])"
)
_NAMED_FUNCTION_RE = re.compile(
    r"(?:函数(?:指针)?|方法)\s*[`（(\"']?([A-Za-z_][A-Za-z0-9_]*)"
    r"|(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*(?:函数|方法)"
    r"|\bfunction\s+`([A-Za-z_][A-Za-z0-9_]*)`"
    r"|(?:更名为|改名为|renamed\s+to)\s*[`\"']?([A-Za-z_][A-Za-z0-9_]*)",
    re.I,
)
_VERSION_RE = re.compile(
    r"(?:\b(?:version|release)\s*|版本\s*|(?<![A-Za-z0-9_])v)(\d+(?:\.\d+){0,3})(?![A-Za-z0-9_.])"
    r"|(?<![A-Za-z0-9_])(?!(?:example|step|section|chapter|figure|table|part|appendix)\s*\d)"
    r"[A-Za-z][A-Za-z_-]*\s*(\d+(?:\.\d+){0,3})(?=\s*(?:中|版本|版))",
    re.I,
)
_HISTORY_RE = re.compile(
    r"早期|旧版|(?:以前|之前|此前)(?:的)?版本|原名|曾经|曾用|更名|改名|弃用|引入|"
    r"\b(?:earlier|previous|older|prior)\s+(?:to\s+)?(?:versions?|releases?)\b|"
    r"\b(?:formerly|renamed|introduced|deprecated|removed|replaced)\b|"
    r"\b(?:since|before)\s+(?:version|release|v?\d)",
    re.I,
)


def _specific_source_gaps(text: str, cited_texts: list[str]) -> list[str]:
    """Conservative specificity gate inside the existing claim Verifier.

    CE relevance cannot establish missing exact API names, explicit versions or
    historical comparisons. These are unverified, not proven false. Passing the
    gate still requires the existing CE/claim checks; it never grants support.
    """

    def versions(value):
        return {a or b for a, b in _VERSION_RE.findall(value)}

    def contains(value, identifier):
        return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])", value, re.I))

    identifiers = set(_IDENTIFIER_RE.findall(text))
    identifiers.update(name for match in _NAMED_FUNCTION_RE.findall(text) for name in match if name)
    missing = sorted(n for n in identifiers if not any(contains(s, n) for s in cited_texts))
    gaps = ["引用原文未出现标识符：" + "、".join(missing)] if missing else []
    stated_versions = versions(text)
    available_versions = {v for source in cited_texts for v in versions(source)}
    absent_versions = sorted(stated_versions - available_versions)
    if absent_versions:
        gaps.append("引用原文未核实版本：" + "、".join(absent_versions))
    if _HISTORY_RE.search(text) and (identifiers or stated_versions):
        # Require a local comparison/rationale statement, rather than names
        # scattered across a long chunk. Two adjacent sentences allow ordinary
        # source prose to state a version and its renamed function separately.
        units = []
        for source in cited_texts:
            sentences = re.split(r"(?<=[。！？!?])|(?<=[.])\s+|\n\s*\n", source)
            units.extend(" ".join(sentences[i : i + 2]) for i in range(len(sentences)))
        if not any(
            len(u) <= 1200
            and _HISTORY_RE.search(u)
            and all(contains(u, n) for n in identifiers)
            and stated_versions <= versions(u)
            for u in units
        ):
            gaps.append("引用原文未提供可核验的版本或历史比较关系。")
    # A high relevance score for one sentence must not certify a new temporal
    # or causal relationship in a compound claim. Require a local source clause.
    atoms = [a.strip() for a in re.split(r"[。！？；;\n]", text) if a.strip()]
    source_atoms = [a.strip() for source in cited_texts for a in re.split(r"[。！？；;\n]", source) if a.strip()]
    duration = r"\d+(?:\.\d+)?\s*(?:毫秒|秒钟?|分钟|小时|天|ms|seconds?|minutes?|hours?)"
    deadline = r"硬性?时间(?:限制|边界)|完成时限|(?:整个|整体).{0,20}操作.{0,5}时限|(?:在|于|此|该).{0,30}(?:时限|期限|时间|分钟|小时).{0,15}(?:内|之内).{0,12}(?:完成|结束|停止)|整个.{0,25}(?:时限|时间|分钟|小时).{0,12}内|直接约束.{0,15}完成"
    causal = r"(?:导致|造成|因而|从而|因此|意味着|保证)"
    negative = r"(?:不意味着|并不|不能|无法|未说明|未提及|不足以|没有证据|是否|不代表)"

    def key_tokens(value):
        return set(_tokens(re.sub(duration, "", value, flags=re.I)))

    def same_relation(claim, source):
        left = re.split(causal, claim, maxsplit=1)
        right = re.split(causal, source, maxsplit=1)
        if len(left) != 2 or len(right) != 2:
            return False
        for a, b in zip(left, right):
            aa, bb = key_tokens(a), key_tokens(b)
            if not aa or not bb or len(aa & bb) / min(len(aa), len(bb)) < 0.6:
                return False
        return True

    for atom in atoms:
        if re.search(negative, atom):
            continue
        if re.search(deadline, atom):
            relevant = [source for source in source_atoms if re.search(deadline, source)]
            if not any(
                key_tokens(atom) and len(key_tokens(atom) & key_tokens(source)) / len(key_tokens(atom)) >= 0.55
                for source in relevant
            ):
                gaps.append("引用原文未明确支持该完成时限或硬性时间约束；超时配置不能单独证明任务完成期限。")
        if re.search(causal, atom) and (len(atoms) > 1 or re.search(r"意味着|保证|因此|从而", atom)):
            if not any(same_relation(atom, source) for source in source_atoms):
                gaps.append("引用原文未明确支持该因果或保证关系。")
    return list(dict.fromkeys(gaps))


def verify_answer(
    answer: str,
    evidence: list[EvidenceItem],
    *,
    settings: Settings | None = None,
) -> VerificationResult:
    """Check each ``[S<n>]`` claim against the cited evidence (Day 7 Verifier).

    Primary signal: Day 5 Cross-Encoder score on ``(claim, cited evidence)``.
    Fallback: lexical token coverage, so near-verbatim quotes still pass if the
    reranker is conservative. Uncited / invalid-citation sentences are treated
    as hallucination candidates for Day 11's hallucination-rate metric.
    """
    settings = settings or get_settings()
    claims = split_claims(answer)
    by_index = {item.index: item for item in evidence}

    pairs: list[tuple[str, str]] = []
    pair_owners: list[int] = []
    valid_citations: list[list[int]] = []

    for claim in claims:
        valid = [idx for idx in claim.citation_indices if idx in by_index]
        valid_citations.append(valid)
        for idx in valid:
            pairs.append((claim.text, by_index[idx].text or ""))
            pair_owners.append(len(valid_citations) - 1)

    ce_scores = pair_scores(pairs, settings=settings, label="verify") if pairs else []
    max_ce: dict[int, float] = {}
    for owner, score in zip(pair_owners, ce_scores):
        prev = max_ce.get(owner)
        if prev is None or score > prev:
            max_ce[owner] = score

    checks = [
        _judge_claim(
            claim,
            evidence_by_index=by_index,
            valid_indices=valid_citations[i],
            entailment_score=max_ce.get(i),
            settings=settings,
        )
        for i, claim in enumerate(claims)
    ]
    return _aggregate(checks)


def lexical_coverage(claim_text: str, evidence_texts: list[str]) -> float:
    claim_tokens = _tokens(claim_text)
    if not claim_tokens:
        return 0.0
    evidence_tokens = {tok for text in evidence_texts for tok in _tokens(text)}
    if not evidence_tokens:
        return 0.0
    overlap = sum(1 for tok in claim_tokens if tok in evidence_tokens)
    return overlap / len(claim_tokens)


def _judge_claim(
    claim: Claim,
    *,
    evidence_by_index: dict[int, EvidenceItem],
    valid_indices: list[int],
    entailment_score: float | None,
    settings: Settings,
) -> ClaimCheck:
    cited_texts = [evidence_by_index[idx].text or "" for idx in valid_indices]
    coverage = round(lexical_coverage(claim.text, cited_texts), 6)
    score = None if entailment_score is None else round(float(entailment_score), 6)
    source_gaps = _specific_source_gaps(claim.text, cited_texts) if valid_indices else []

    if any(idx not in evidence_by_index for idx in claim.citation_indices):
        status = STATUS_INVALID_CITATION
    elif _INSUFFICIENT_RE.search(claim.text):
        status = STATUS_INSUFFICIENT
    elif not claim.citation_indices:
        status = STATUS_UNCITED
    elif not valid_indices:
        status = STATUS_INVALID_CITATION
    elif source_gaps:
        status = STATUS_UNVERIFIED
    elif score is not None and score >= settings.verify_entail_threshold:
        status = STATUS_SUPPORTED
    elif score is not None and score >= settings.verify_lexical_gate and coverage >= settings.verify_lexical_threshold:
        # Near-verbatim quotes the reranker under-scores; do not let lexical
        # overlap rescue a claim the Cross-Encoder already treats as unrelated.
        status = STATUS_SUPPORTED
    else:
        status = STATUS_UNSUPPORTED

    if valid_indices and status in {STATUS_SUPPORTED, STATUS_UNSUPPORTED}:
        from app.verify.inductive import (
            STATUS_UNSUPPORTED_INDUCTIVE,
            is_elevation,
            judge_inductive,
            split_subclaims,
        )

        if is_elevation(claim.text):
            support_t = tuple(valid_indices)
            subs = split_subclaims(claim.text)
            subs_ok = all(lexical_coverage(sub, cited_texts) >= settings.verify_lexical_threshold for sub in subs)
            evidence_map = {idx: {"text": evidence_by_index[idx].text or ""} for idx in valid_indices}
            judged = judge_inductive(
                claim.text,
                citation_indices=valid_indices,
                evidence_by_index=evidence_map,
                subset_entailed={(claim.text, support_t): status == STATUS_SUPPORTED and subs_ok},
            )
            if judged == STATUS_UNSUPPORTED_INDUCTIVE:
                status = STATUS_UNSUPPORTED

    return ClaimCheck(
        text=claim.text,
        citation_indices=list(claim.citation_indices),
        lexical_coverage=coverage,
        entailment_score=score,
        status=status,
        hallucination=status in _HALLUCINATION_STATUSES,
        source_gaps=source_gaps,
    )


def _tokens(value: str) -> list[str]:
    """English/numeric words plus CJK character bigrams (more precise than unigrams)."""
    raw = _TOKEN_RE.findall(value)
    tokens: list[str] = []
    cjk: list[str] = []

    def flush_cjk() -> None:
        if len(cjk) == 1:
            if cjk[0] not in _STOPWORDS:
                tokens.append(cjk[0])
        elif len(cjk) >= 2:
            tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
        cjk.clear()

    for piece in raw:
        if "\u4e00" <= piece <= "\u9fff":
            cjk.append(piece)
            continue
        flush_cjk()
        lowered = piece.lower()
        if lowered not in _STOPWORDS:
            tokens.append(lowered)
    flush_cjk()
    return tokens


def verify_synthesis(
    *,
    modules: list[dict],
    order_note: str = "",
    gaps: list[dict[str, str]] | None = None,
    evidence: list[EvidenceItem],
    settings: Settings | None = None,
    task_questions: list[str] | None = None,
) -> VerificationResult:
    """Verify module paragraphs. Gaps / order_note are attached as meta."""
    from app.generate.synthesis import synthesis_verify_body

    body = synthesis_verify_body(modules)
    if body:
        result = verify_answer(body, evidence, settings=settings)
    else:
        result = VerificationResult(
            claim_count=0,
            supported_count=0,
            hallucination_count=0,
            coverage_rate=1.0,
            hallucination_rate=0.0,
            claims=[],
        )
    coverage = module_coverage_rate(modules, result.claims)
    questions = [str(item) for item in (task_questions or []) if str(item).strip()]
    task_cov = task_coverage_rate(questions, modules) if questions else None
    meta = [
        ClaimCheck(
            text=item.text,
            citation_indices=list(item.citation_indices or []),
            lexical_coverage=0.0,
            entailment_score=None,
            status=item.status,
            hallucination=item.hallucination,
        )
        for item in meta_claim_checks(order_note, gaps)
    ]
    return VerificationResult(
        claim_count=len(result.claims) + len(meta),
        supported_count=result.supported_count,
        hallucination_count=result.hallucination_count,
        coverage_rate=coverage,
        hallucination_rate=result.hallucination_rate,
        claims=list(result.claims) + meta,
        task_coverage=task_cov,
        unverified_count=result.unverified_count,
    )


def _aggregate(checks: list[ClaimCheck]) -> VerificationResult:
    evaluable = [item for item in checks if item.status not in {STATUS_INSUFFICIENT, STATUS_META}]
    supported = sum(1 for item in evaluable if item.status == STATUS_SUPPORTED)
    hallucinations = sum(1 for item in evaluable if item.hallucination)
    total = len(evaluable)
    if total == 0:
        coverage_rate = 1.0
        hallucination_rate = 0.0
    else:
        coverage_rate = round(supported / total, 6)
        hallucination_rate = round(hallucinations / total, 6)
    return VerificationResult(
        claim_count=len(checks),
        supported_count=supported,
        hallucination_count=hallucinations,
        coverage_rate=coverage_rate,
        hallucination_rate=hallucination_rate,
        claims=checks,
        unverified_count=sum(item.status == STATUS_UNVERIFIED for item in checks),
    )


def reconcile_delivery(state, store, *, requirements=None, reviews=None, exhaustive=False) -> dict:
    """Reconcile existing receipts and Verifier results, without semantic inference.

    Requirements bind atomic task facts to original source spans. Optional reviews
    are trusted, source-backed acceptance annotations, NEVER model self-reports.
    Without an exhaustive reviewed fact inventory, completeness remains unknown.
    The fixed acceptance oracles belong in fixtures, not in this function.
    """
    import hashlib

    def normalized(text):
        return re.sub(r"\s+", "", re.sub(r"\[S\d+\]", "", str(text or "")))

    def paragraphs(modules):
        return [
            p
            for m in (modules if isinstance(modules, list) else [])
            if isinstance(m, dict)
            for p in (m.get("paragraphs", m.get("bullets", [])) or [])
            if isinstance(p, dict)
        ]

    explicit_requirements = requirements is not None
    requirements = (
        requirements
        if explicit_requirements
        else [
            {"id": str(i + 1), "text": q, "sources": []}
            for i, q in enumerate(state.task_spec.questions if state.task_spec else [state.goal])
        ]
    )
    reviews = reviews or {}
    receipts = {x["chunk_id"]: x for x in state.reading.get("receipts", [])}
    evidence = {e.chunk_id: e for e in state.evidence}
    source_proofs = [p for r in requirements for p in r.get("sources", [])]
    current = store.load_chunks(list({p.get("chunk_id", "") for p in source_proofs} | set(evidence)))
    # The store applies tenant and hard-scope filters. Reuse the existing source
    # assessor for hash identity; receipt SHA pins the entire original text.
    from app.verify.structure import assess_citations

    integrity = assess_citations(
        [
            EvidenceItem(
                index=i + 1,
                chunk_id=p.get("chunk_id", ""),
                doc_id=p.get("doc_id", ""),
                page=0,
                chunk_hash=p.get("hash", ""),
                text=getattr(current.get(p.get("chunk_id")), "text", ""),
            )
            for i, p in enumerate(source_proofs)
        ],
        {cid: {"doc_id": row.doc_id, "chunk_hash": row.chunk_hash, "text": row.text} for cid, row in current.items()},
        owner_id=None,
    )
    identity_ok = {
        (p.get("chunk_id"), p.get("hash")): check["status"] == "ok" for p, check in zip(source_proofs, integrity)
    }
    selection = state.evidence_selection.get("compose", {})
    delivery = state.evidence_selection.get("delivery", {})
    raw = paragraphs(delivery.get("raw_content", [])) if delivery.get("origin") == "model" else []
    visible = paragraphs(state.outline.get("modules", []))
    confirmed = bool(
        (selection.get("provider_response_received") or selection.get("model_input_confirmed"))
        and selection.get("provider") not in {None, "stub"}
    )
    claims = (state.verification or {}).get("claims", [])
    unverified_claims = [c for c in claims if c.get("status") == STATUS_UNVERIFIED]
    rows = []
    for requirement in requirements:
        review = reviews.get(requirement["id"], {})
        proofs = []
        for proof in requirement.get("sources", []):
            cid = proof.get("chunk_id", "")
            row = current.get(cid)
            valid = bool(
                row
                and str(row.doc_id) == proof.get("doc_id")
                and proof.get("hash")
                and identity_ok.get((cid, proof["hash"]))
                and hashlib.sha256(row.text.encode()).hexdigest() == proof.get("text_sha256")
                and normalized(proof.get("quote"))
                and normalized(proof["quote"]) in normalized(row.text)
            )
            receipt = receipts.get(cid, {})
            read = bool(
                valid
                and receipt.get("status") == "read"
                and receipt.get("hash") == proof.get("hash")
                and receipt.get("text_sha256") == proof.get("text_sha256")
                and proof.get("section_id") in receipt.get("section_ids", [])
            )
            fed = bool(
                read
                and cid in evidence
                and evidence[cid].chunk_hash == proof.get("hash")
                and evidence[cid].text == row.text
            )
            proofs.append({**proof, "valid": valid, "read": read, "prepared": fed, "model_input": fed and confirmed})
        valid = any(p["valid"] for p in proofs)
        read = any(p["read"] for p in proofs)
        fed = any(p["prepared"] for p in proofs)
        required_indices = {evidence[p["chunk_id"]].index for p in proofs if p["prepared"]}
        text = normalized(review.get("conclusion"))
        # Host formatting may add a step label. Bind each side explicitly;
        # neither fuzzy matching nor deleting arbitrary prefixes proves lineage.
        raw_text = normalized(review.get("raw_conclusion", review.get("conclusion")))
        raw_hits = [p for p in raw if raw_text and raw_text == normalized(p.get("text"))]
        visible_hits = [p for p in visible if text and text == normalized(p.get("text"))]
        cited = {int(i) for p in visible_hits for i in p.get("sources", []) if str(i).isdigit()}
        checked = [
            c
            for c in claims
            if normalized(c.get("text")) and text and normalized(c["text"]) in text and c.get("status") != STATUS_META
        ]
        # A review may adjudicate the fixed oracle, but only after exact output
        # binding and a current, actually fed original source match above.
        inline = {int(i) for p in visible_hits for i in re.findall(r"\[S(\d+)\]", str(p.get("text", "")))}
        supported = (
            bool(cited & required_indices)
            and (cited | inline) <= {e.index for e in state.evidence}
            and (
                bool(checked)
                and all(c.get("status") == STATUS_SUPPORTED for c in checked)
                or review.get("source_support") == "confirmed_by_human"
            )
        )
        if not explicit_requirements and not proofs:
            stage = "fact_binding_pending"
        elif not valid:
            stage = "source_unverifiable"
        elif not read:
            stage = "not_read"
        elif not fed:
            stage = "not_fed"
        elif not confirmed:
            stage = "input_unconfirmed"
        elif delivery.get("origin") != "model":
            stage = "generation_unavailable"
        elif raw_hits and selection.get("source_contract_error"):
            stage = "citation_unsupported"
        elif raw_hits and not visible_hits:
            stage = "removed_by_cleaning"
        elif visible_hits and not supported:
            stage = "citation_unsupported"
        elif raw_hits and visible_hits and supported:
            stage = "delivered"
        elif review.get("absent_from_generated") and not raw_hits:
            stage = "not_generated"
        else:
            stage = "delivery_unassessed"
        rows.append(
            {
                "id": requirement["id"],
                "requirement": requirement["text"],
                "sources": proofs,
                "stage": stage,
                "conclusion": review.get("conclusion", ""),
                "raw_conclusion": review.get("raw_conclusion", review.get("conclusion", "")),
                "reading_covered": read,
                "model_input_covered": fed and confirmed,
                "answer_covered": bool(raw_hits and visible_hits),
                "citation_supported": supported,
            }
        )
    written = state.compose_mode == "written" and confirmed and bool(visible)
    complete = bool(
        exhaustive
        and rows
        and all(r["stage"] == "delivered" for r in rows)
        and written
        and not delivery.get("lost_paragraphs")
        and not unverified_claims
        and selection.get("finish_reason") != "length"
    )
    status = "complete" if complete else "partial" if written and evidence else "unverifiable"
    # Source accounting is independent of whether task questions have an atomic
    # reviewed fact binding. An empty fact proof list is unknown, not invalid source.
    read_ids = {cid for doc in state.reading.get("documents", {}).values() for cid in doc.get("read_chunk_ids", [])}
    read_ids.update(cid for cid, receipt in receipts.items() if receipt.get("status") == "read")
    valid_ids = {
        cid
        for cid, e in evidence.items()
        if cid in current
        and e.chunk_hash
        and e.chunk_hash == current[cid].chunk_hash
        and e.doc_id == current[cid].doc_id
        and e.text == current[cid].text
    }
    sent_indices = set(selection.get("sent_indices", []))
    candidates = {e.get("chunk_id"): e for e in selection.get("candidate_evidence", [])}
    fed_ids = (
        {
            cid
            for cid in valid_ids
            if evidence[cid].index in sent_indices
            and candidates.get(cid, {}).get("text") == evidence[cid].text
            and candidates.get(cid, {}).get("chunk_hash") == evidence[cid].chunk_hash
        }
        if confirmed
        else set()
    )
    cited_indices = {int(i) for p in visible for i in p.get("sources", []) if str(i).isdigit()}
    required_ids = {
        cid for doc in state.reading.get("documents", {}).values() for cid in doc.get("required_chunk_ids", [])
    }
    source_coverage = {
        "required_chunk_ids": sorted(required_ids),
        "valid_chunk_ids": sorted(valid_ids),
        "read_chunk_ids": sorted(valid_ids & read_ids),
        "model_input_chunk_ids": sorted(fed_ids),
        "answer_cited_chunk_ids": sorted(cid for cid in fed_ids if evidence[cid].index in cited_indices),
        "reading_complete": bool(required_ids) and required_ids <= (valid_ids & read_ids),
        "model_input_complete": bool(required_ids) and required_ids <= fed_ids,
        "semantic_fact_coverage": "not_assessed",
    }
    return {
        "version": 1,
        "status": status,
        "source_coverage": source_coverage,
        "requirements_exhaustive": bool(exhaustive),
        "unverified_claims": unverified_claims,
        "requirements": rows,
        "model_input_confirmed": confirmed,
        "note": "完整交付需逐项原文、实际输入、可见结论及引用支持；未建立完整事实清单时仅作部分指南。",
        "dimensions": {
            "reading": "source receipts",
            "answer": "raw and visible conclusion bindings",
            "citations": "existing Verifier or explicit source-backed human acceptance review",
        },
    }
