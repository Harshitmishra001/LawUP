"""Unit tests for MeaningVerifierAgent (bidirectional NLI verifier)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from backend.app.agents.verifier_meaning import MeaningVerifierAgent, verify_meaning


@pytest.fixture
def agent():
    return MeaningVerifierAgent(model_name="LawUP-Test", lm_studio_url="http://localhost:1234/v1")


def test_verifier_initialization(agent):
    assert agent.model_name == "LawUP-Test"
    assert agent.url == "http://localhost:1234/v1"
    assert agent.strict_mode is True


def test_verifier_empty_inputs(agent):
    res_empty_orig = agent.run("", "Valid rewrite text")
    assert res_empty_orig["verdict"] == "FAIL"
    assert res_empty_orig["confidence"] == "HIGH"
    assert res_empty_orig["original_entails_rewrite"] is False

    res_empty_rewr = agent.run("Valid original text", "   ")
    assert res_empty_rewr["verdict"] == "FAIL"
    assert res_empty_rewr["confidence"] == "HIGH"
    assert res_empty_rewr["rewrite_entails_original"] is False


def test_verifier_sync_pass(agent):
    mock_payload = {
        "original_entails_rewrite": True,
        "rewrite_entails_original": True,
        "verdict": "PASS",
        "failures": [],
        "confidence": "HIGH",
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(agent._sync_client.chat.completions, "create", return_value=mock_response):
        result = agent.run(
            original="Either party may terminate upon thirty days written notice.",
            rewrite="Either party can cancel this agreement by giving 30 days notice in writing.",
        )

        assert result["verdict"] == "PASS"
        assert result["original_entails_rewrite"] is True
        assert result["rewrite_entails_original"] is True
        assert result["failures"] == []
        assert result["confidence"] == "HIGH"


def test_verifier_sync_forward_entailment_hallucination(agent):
    mock_payload = {
        "original_entails_rewrite": False,
        "rewrite_entails_original": True,
        "verdict": "FAIL",
        "failures": [
            {
                "category": "HALLUCINATION",
                "description": "Rewrite invented a 10% late payment fee not in the original clause.",
                "original_span": None,
                "rewrite_span": "10% late fee",
            }
        ],
        "confidence": "HIGH",
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(agent._sync_client.chat.completions, "create", return_value=mock_response):
        result = agent.run(
            original="Payments are due on the first day of each calendar month.",
            rewrite="Payments are due on the 1st of each month with a 10% late fee if unpaid.",
        )

        assert result["verdict"] == "FAIL"
        assert result["original_entails_rewrite"] is False
        assert len(result["failures"]) == 1
        assert result["failures"][0]["category"] == "HALLUCINATION"


@pytest.mark.asyncio
async def test_verifier_async_backward_entailment_omission(agent):
    mock_payload = {
        "original_entails_rewrite": True,
        "rewrite_entails_original": False,
        "verdict": "FAIL",
        "failures": [
            {
                "category": "OMISSION",
                "description": "Rewrite dropped the mandatory 5-day cure period before termination.",
                "original_span": "provided that Company has 5 business days to cure",
                "rewrite_span": None,
            }
        ],
        "confidence": "HIGH",
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(agent._async_client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await agent.run_async(
            original="Either party may terminate upon material breach, provided that Company has 5 business days to cure.",
            rewrite="Either party may terminate if there is a material breach.",
        )

        assert result["verdict"] == "FAIL"
        assert result["rewrite_entails_original"] is False
        assert len(result["failures"]) == 1
        assert result["failures"][0]["category"] == "OMISSION"


def test_verifier_markdown_fence_and_preamble_parsing(agent):
    raw_markdown = """Here is your evaluation:
```json
{
  "original_entails_rewrite": true,
  "rewrite_entails_original": true,
  "verdict": "PASS",
  "failures": [],
  "confidence": "MEDIUM"
}
```
Hope this helps!"""

    mock_choice = MagicMock()
    mock_choice.message.content = raw_markdown
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(agent._sync_client.chat.completions, "create", return_value=mock_response):
        result = agent.run(
            original="The term shall be one year.",
            rewrite="This agreement lasts for one year.",
        )

        assert result["verdict"] == "PASS"
        assert result["original_entails_rewrite"] is True
        assert result["confidence"] == "MEDIUM"


def test_verifier_fail_closed_invalid_json(agent):
    mock_choice = MagicMock()
    mock_choice.message.content = "I cannot determine whether this clause preserves meaning."
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(agent._sync_client.chat.completions, "create", return_value=mock_response):
        result = agent.run(
            original="Original text here.",
            rewrite="Rewrite text here.",
        )

        assert result["verdict"] == "FAIL"
        assert result["confidence"] == "LOW"
        assert "error" in result
        assert result["original_entails_rewrite"] is False


def test_verifier_offline_connection_fallback(agent):
    with patch.object(
        agent._sync_client.chat.completions,
        "create",
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = agent.run(
            original="Contractor will be paid $50,000 within 15 days.",
            rewrite="Contractor will receive compensation soon.",
        )

        # Connection error should fall back to heuristic and flag missing numbers with LOW confidence
        assert result["verdict"] == "FAIL"
        assert result["confidence"] == "LOW"
        assert "error" in result
        assert any(f["category"] == "OMISSION" for f in result["failures"])


def test_verifier_heuristic_omission_check(agent):
    orig = "The executive receives a base compensation of $250,000 and 60 days severance."
    rewr = "The executive gets base compensation and severance benefits."

    report = agent.verify_heuristic(orig, rewr)
    assert report["verdict"] == "FAIL"
    assert report["rewrite_entails_original"] is False
    assert any("$250,000" in f["description"] for f in report["failures"])
    assert any("60 days" in f["description"] for f in report["failures"])


def test_verifier_heuristic_contradiction_check(agent):
    orig = "Licensee shall not distribute, sublicense, or share the source code."
    rewr = "The licensee can distribute the software freely."

    report = agent.verify_heuristic(orig, rewr)
    assert report["verdict"] == "FAIL"
    assert any(f["category"] == "CONTRADICTION" for f in report["failures"])


def test_verify_meaning_top_level_wrapper():
    mock_payload = {
        "original_entails_rewrite": True,
        "rewrite_entails_original": True,
        "verdict": "PASS",
        "failures": [],
        "confidence": "HIGH",
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_payload)
    mock_response = MagicMock(choices=[mock_choice])

    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        res = verify_meaning("Original legal clause", "Simplified legal clause")
        assert res["verdict"] == "PASS"
        assert res["original_entails_rewrite"] is True
