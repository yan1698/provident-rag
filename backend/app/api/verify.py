from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth.deps import get_current_user
from app.core.config import Settings, get_settings
from app.generate.prompt import EvidenceItem
from app.models.user import User
from app.verify.service import VerificationResult, verify_answer

router = APIRouter(tags=["verify"])


class VerifyEvidenceIn(BaseModel):
    index: int = Field(..., ge=1)
    chunk_id: str = ""
    doc_id: str = ""
    page: int = 1
    text: str = Field(..., min_length=1)
    chunk_hash: str = ""


class VerifyRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=20000)
    evidence: list[VerifyEvidenceIn] = Field(default_factory=list)


class ClaimCheckOut(BaseModel):
    text: str
    citation_indices: list[int]
    lexical_coverage: float
    entailment_score: float | None
    status: str
    hallucination: bool
    source_gaps: list[str] = Field(default_factory=list)


class VerificationOut(BaseModel):
    claim_count: int
    supported_count: int
    hallucination_count: int
    coverage_rate: float
    hallucination_rate: float
    claims: list[ClaimCheckOut]
    unverified_count: int = 0


def to_verification_out(result: VerificationResult) -> VerificationOut:
    return VerificationOut(
        claim_count=result.claim_count,
        supported_count=result.supported_count,
        hallucination_count=result.hallucination_count,
        coverage_rate=result.coverage_rate,
        hallucination_rate=result.hallucination_rate,
        unverified_count=result.unverified_count,
        claims=[
            ClaimCheckOut(
                text=item.text,
                citation_indices=item.citation_indices,
                lexical_coverage=item.lexical_coverage,
                entailment_score=item.entailment_score,
                status=item.status,
                hallucination=item.hallucination,
                source_gaps=item.source_gaps,
            )
            for item in result.claims
        ],
    )


@router.post("/verify", response_model=VerificationOut)
def verify(
    body: VerifyRequest,
    settings: Settings = Depends(get_settings),
    _user: User | None = Depends(get_current_user),
) -> VerificationOut:
    answer = body.answer.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="answer must not be empty")

    evidence = [
        EvidenceItem(
            index=item.index,
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            page=item.page,
            text=item.text,
            chunk_hash=item.chunk_hash,
        )
        for item in body.evidence
    ]
    result = verify_answer(answer, evidence, settings=settings)
    return to_verification_out(result)
