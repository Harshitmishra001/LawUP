"""Unit tests for CitationVerifier (Stage 5b citation grounding verification)."""

import pytest
from backend.app.agents.verifier_citation import CitationVerifier


@pytest.fixture
def verifier():
    return CitationVerifier(min_similarity_threshold=0.40, strict_mode=True)


def test_citation_verifier_initialization(verifier):
    assert verifier.min_similarity_threshold == 0.40
    assert verifier.strict_mode is True


def test_citation_verifier_supported_pass(verifier):
    clause = "Client may terminate this Agreement without cause upon giving thirty (30) days written notice."
    risks = [{"risk": "Termination Risk", "sub_category": "Termination For Convenience"}]
    references = [
        {
            "risk_category": "Termination Risk",
            "sub_category": "Termination For Convenience",
            "reference_text": "Commercial standard allows termination for convenience with 30 days notice.",
            "authority_type": "industry_norm",
            "similarity_score": 0.78,
        }
    ]

    result = verifier.run(clause, risks, references)
    assert result["verdict"] == "PASS"
    assert result["total_checked"] == 1
    assert result["total_supported"] == 1
    assert result["total_dropped"] == 0
    assert len(result["verified_references"]) == 1
    assert len(result["dropped_references"]) == 0
    assert result["details"][0]["supported"] is True


def test_citation_verifier_low_similarity_dropped(verifier):
    clause = "Client shall pay invoices within 30 days."
    risks = [{"risk": "Termination Risk", "sub_category": "Notice Period To Terminate"}]
    references = [
        {
            "risk_category": "Termination Risk",
            "sub_category": "Notice Period To Terminate",
            "reference_text": "Termination requires notice period of standard duration.",
            "authority_type": "industry_norm",
            "similarity_score": 0.22,  # Below 0.40 threshold
        }
    ]

    result = verifier.run(clause, risks, references)
    assert result["verdict"] == "WARN"
    assert result["total_checked"] == 1
    assert result["total_supported"] == 0
    assert result["total_dropped"] == 1
    assert len(result["verified_references"]) == 0
    assert len(result["dropped_references"]) == 1
    assert "below threshold" in result["dropped_references"][0]["reason"]


def test_citation_verifier_category_mismatch_dropped(verifier):
    clause = "Each party shall hold confidential information in strict trust."
    risks = [{"risk": "Confidentiality Scope", "sub_category": "Non-Disclosure"}]
    references = [
        {
            "risk_category": "Liability Exposure",
            "sub_category": "Cap On Liability",
            "reference_text": "Liability is capped at the total amount paid.",
            "authority_type": "industry_norm",
            "similarity_score": 0.85,
        }
    ]

    result = verifier.run(clause, risks, references)
    assert result["verdict"] == "WARN"
    assert result["total_supported"] == 0
    assert result["total_dropped"] == 1
    assert "Category mismatch" in result["details"][0]["reason"]


def test_citation_verifier_empty_inputs(verifier):
    # Empty clause
    res_empty_clause = verifier.run("", [{"risk": "Termination Risk"}], [{"risk_category": "Termination Risk", "reference_text": "Valid text", "similarity_score": 0.70}])
    assert res_empty_clause["total_supported"] == 0
    assert res_empty_clause["total_dropped"] == 1

    # Empty reference text
    res_empty_ref = verifier.run("Valid clause text", [{"risk": "Termination Risk"}], [{"risk_category": "Termination Risk", "reference_text": "  ", "similarity_score": 0.70}])
    assert res_empty_ref["total_supported"] == 0
    assert res_empty_ref["total_dropped"] == 1


def test_citation_verifier_mixed_references(verifier):
    clause = "Vendor agrees to a cap on liability equal to fees paid in the preceding 12 months."
    risks = [
        {"risk": "Liability Exposure", "sub_category": "Cap On Liability"},
        {"risk": "Termination Risk", "sub_category": "Notice Period To Terminate Renewal"},
    ]
    references = [
        {
            "risk_category": "Liability Exposure",
            "sub_category": "Cap On Liability",
            "reference_text": "Standard commercial contracts limit liability to trailing 12-month fees.",
            "similarity_score": 0.82,
        },
        {
            "risk_category": "Termination Risk",
            "sub_category": "Notice Period To Terminate Renewal",
            "reference_text": "Auto-renewal termination notice window.",
            "similarity_score": 0.15,  # Bad match
        },
    ]

    result = verifier.run(clause, risks, references)
    assert result["total_checked"] == 2
    assert result["total_supported"] == 1
    assert result["total_dropped"] == 1
    assert len(result["verified_references"]) == 1
    assert result["verified_references"][0]["risk_category"] == "Liability Exposure"


@pytest.mark.asyncio
async def test_citation_verifier_async_execution(verifier):
    clause = "Governing law shall be the laws of the State of New York."
    risks = [{"risk": "Dispute / Governing Law", "sub_category": "Governing Law"}]
    references = [
        {
            "risk_category": "Dispute / Governing Law",
            "sub_category": "Governing Law",
            "reference_text": "Choice of law clauses designate the substantive law governing the contract.",
            "similarity_score": 0.91,
        }
    ]

    result = await verifier.run_async(clause, risks, references)
    assert result["verdict"] == "PASS"
    assert result["total_supported"] == 1


def test_citation_verifier_non_strict_mode():
    lenient_verifier = CitationVerifier(min_similarity_threshold=0.50, strict_mode=False)
    clause = "Test clause"
    risks = [{"risk": "Termination Risk"}]
    references = [{"risk_category": "Termination Risk", "reference_text": "Some text", "similarity_score": 0.20}]

    result = lenient_verifier.run(clause, risks, references)
    assert result["total_dropped"] == 1
    # When strict_mode=False, all references are retained in verified_references
    assert len(result["verified_references"]) == 1
    assert result["verdict"] == "WARN"
