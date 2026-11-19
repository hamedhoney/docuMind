from typing import Literal

from pydantic import BaseModel, Field

VerificationStatus = Literal[
    "supported",
    "contradicted",
    "insufficient_evidence",
]


class VerifiedClaim(BaseModel):
    claim: str
    status: VerificationStatus
    citation_ids: list[int] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    explanation: str


class VerificationReport(BaseModel):
    claims: list[VerifiedClaim] = Field(default_factory=list)
    supported_count: int = 0
    contradicted_count: int = 0
    insufficient_count: int = 0
    support_rate: float | None = None
