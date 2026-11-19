from unittest.mock import Mock

from documind.verification.models import VerificationReport
from documind.verification.verifier import (
    EvidenceVerifier,
    ExtractedClaim,
)


def make_verifier():
    generator = Mock()
    generator.client = Mock()
    generator.model = "test-model"
    return EvidenceVerifier(generator)


def test_invalid_citation_is_rejected():
    verifier = make_verifier()

    claim = ExtractedClaim(
        claim="The experiment included 100 patients.",
        citation_ids=[99],
    )

    passages = [
        {
            "text": "The study included 20 patients.",
        }
    ]

    result = verifier.verify_claim(claim, passages)

    assert result.status == "insufficient_evidence"
    assert result.citation_ids == [99]


def test_missing_citation_is_rejected():
    verifier = make_verifier()

    claim = ExtractedClaim(
        claim="The treatment was effective.",
        citation_ids=[],
    )

    result = verifier.verify_claim(claim, [])

    assert result.status == "insufficient_evidence"


def test_empty_answer():
    verifier = make_verifier()

    report = verifier.verify("", [])

    assert isinstance(report, VerificationReport)
    assert report.claims == []
    assert report.support_rate is None


def test_supported_claim():
    verifier = make_verifier()

    verifier._complete = Mock(
        return_value={
            "status": "supported",
            "explanation": "Directly supported by the source.",
        }
    )

    claim = ExtractedClaim(
        claim="RAG combines retrieval and generation.",
        citation_ids=[1],
    )

    passages = [
        {"text": ("RAG combines information retrieval with language model generation.")}
    ]

    result = verifier.verify_claim(claim, passages)

    assert result.status == "supported"
    assert result.citation_ids == [1]
    assert len(result.evidence) == 1
