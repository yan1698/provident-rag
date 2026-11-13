"""Day 7 — claim-evidence Verifier."""

from __future__ import annotations

__all__ = [
    "Claim",
    "ClaimCheck",
    "VerificationResult",
    "split_claims",
    "verify_answer",
]


def __getattr__(name: str):
    if name in {"Claim", "split_claims"}:
        from app.verify.claims import Claim, split_claims

        return {"Claim": Claim, "split_claims": split_claims}[name]
    if name in {"ClaimCheck", "VerificationResult", "verify_answer"}:
        from app.verify.service import ClaimCheck, VerificationResult, verify_answer

        return {
            "ClaimCheck": ClaimCheck,
            "VerificationResult": VerificationResult,
            "verify_answer": verify_answer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
