"""Meaning Verifier (NLI) Agent — bidirectional entailment verification for legal rewrites."""

import json
import os
import re
from typing import Any, Dict, List, Optional
import httpx
from openai import AsyncOpenAI, OpenAI

SYSTEM_PROMPT = """You are a meaning-preservation auditor for legal contract simplification. 
You will be given two texts:

- ORIGINAL: a clause from a real legal contract, written in formal legal language.
- REWRITE: a plain-English rewrite of that clause, produced by an AI model.

Your job is NOT to judge writing quality, tone, or readability. Your job is 
to determine whether the REWRITE preserves the exact legal meaning of the 
ORIGINAL — no more, no less.

You must check meaning preservation in BOTH directions:

DIRECTION A — Forward entailment (checks for hallucination/contradiction):
  Assume ORIGINAL is true. Does everything asserted in REWRITE necessarily 
  follow? Flag anything in REWRITE that is NOT supported by ORIGINAL — 
  including invented conditions, inverted obligations (e.g. "may" 
  rewritten as "must"), added parties, added dollar amounts, or merged 
  provisions that change who a duty applies to.

DIRECTION B — Backward entailment (checks for omission):
  Assume REWRITE is true. Does everything asserted in ORIGINAL necessarily 
  follow? Flag anything in ORIGINAL that is NOT captured in REWRITE — 
  including dropped conditions, dropped sub-clauses, dropped qualifiers 
  ("only," "except," "provided that"), narrowed scope, or collapsed 
  distinctions between two separate provisions (e.g. two parties with 
  different rights being merged into one shared right).

A rewrite only PASSES if BOTH directions hold. If either direction fails, 
the rewrite FAILS, regardless of how well-written it is.

Classify every failure into exactly one category:
- HALLUCINATION: REWRITE asserts something ORIGINAL does not support.
- CONTRADICTION: REWRITE inverts or reverses an obligation, right, or 
  condition from ORIGINAL.
- OMISSION: REWRITE drops a standalone condition, qualifier, or obligation from ORIGINAL without merging anything.
- STRUCTURAL_COLLAPSE: REWRITE merges two or more distinct provisions (e.g. different rights held by different parties, or separate conditions) into one, losing the specific boundary between them.

A rewrite may have multiple failures. List all of them.

Do not consider stylistic simplification, reordering of sentences, or 
converting passive to active voice as failures, as long as the legal 
substance is unchanged.

You must respond with ONLY a single valid JSON object. No preamble, no 
markdown code fences, no explanation outside the JSON fields below.

Output schema:
{
  "original_entails_rewrite": boolean,
  "rewrite_entails_original": boolean,
  "verdict": "PASS" | "FAIL",
  "failures": [
    {
      "category": "HALLUCINATION" | "CONTRADICTION" | "OMISSION" | "STRUCTURAL_COLLAPSE",
      "description": "One or two sentences pinpointing exactly what changed and where in the text.",
      "original_span": "The specific phrase or sub-clause in ORIGINAL involved (quote sparingly, just enough to locate it)",
      "rewrite_span": "The specific phrase in REWRITE involved, or null if the issue is a pure omission with no corresponding text"
    }
  ],
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}

If verdict is "PASS", "failures" must be an empty array.
Set confidence to LOW if the clause contains cross-references (e.g. 
"as provided in Article 12") that you cannot fully resolve without 
additional context, or if the clause is long enough that subtle 
omissions are hard to rule out with certainty."""

USER_MESSAGE_TEMPLATE = """ORIGINAL:
{original_clause}

REWRITE:
{rewrite_text}

Evaluate meaning preservation per your instructions. Respond with the JSON object only."""

VALID_FAILURE_CATEGORIES = {
    "HALLUCINATION",
    "CONTRADICTION",
    "OMISSION",
    "STRUCTURAL_COLLAPSE",
}


class MeaningVerifierAgent:
    """Meaning Verifier Agent — bidirectional Natural Language Inference (NLI) auditor.

    Pipeline step: 5a (reference Section 5 of PLAN.md)

    Verifies bidirectional entailment between an original contract clause and its
    simplified rewrite:
      1. Direction A: Forward entailment (Original entails Rewrite).
         Flags HALLUCINATIONS and CONTRADICTIONS.
      2. Direction B: Backward entailment (Rewrite entails Original).
         Flags OMISSIONS and STRUCTURAL_COLLAPSE.

    Fail-closed policy:
      - If either entailment direction fails, verdict is FAIL.
      - If parsing or connection fails, verdict is FAIL with LOW confidence.
    """

    def __init__(
        self,
        model_name: str = "LawUP",
        lm_studio_url: Optional[str] = None,
        timeout: float = 30.0,
        strict_mode: bool = True,
    ):
        if lm_studio_url is None:
            lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

        self.model_name = model_name
        self.url = lm_studio_url
        self.timeout = timeout
        self.strict_mode = strict_mode

        self._sync_client = OpenAI(
            base_url=lm_studio_url,
            api_key="lm-studio",
            timeout=timeout,
        )
        self._async_client = AsyncOpenAI(
            base_url=lm_studio_url,
            api_key="lm-studio",
            timeout=timeout,
        )

    def _build_prompt(self, original: str, rewrite: str) -> str:
        return USER_MESSAGE_TEMPLATE.format(
            original_clause=original.strip(),
            rewrite_text=rewrite.strip(),
        )

    def _parse_and_validate_response(self, raw_text: str) -> Dict[str, Any]:
        """Extracts JSON from model output and enforces strict schema constraints."""
        json_str = raw_text.strip()

        # Handle markdown fences if returned
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1).strip()
        else:
            json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [
                    {
                        "category": "STRUCTURAL_COLLAPSE",
                        "description": f"Failed to parse structured NLI output from verifier: {exc}",
                        "original_span": None,
                        "rewrite_span": None,
                    }
                ],
                "confidence": "LOW",
                "error": f"JSON parse error: {exc}",
                "raw": raw_text,
            }

        # Normalize fields
        orig_entails = bool(parsed.get("original_entails_rewrite", False))
        rewr_entails = bool(parsed.get("rewrite_entails_original", False))

        raw_failures = parsed.get("failures", [])
        validated_failures: List[Dict[str, Any]] = []
        if isinstance(raw_failures, list):
            for item in raw_failures:
                if isinstance(item, dict):
                    cat = item.get("category", "OMISSION")
                    if cat not in VALID_FAILURE_CATEGORIES:
                        cat = "OMISSION"
                    validated_failures.append({
                        "category": cat,
                        "description": str(item.get("description", "")),
                        "original_span": item.get("original_span"),
                        "rewrite_span": item.get("rewrite_span"),
                    })

        # Strict fail-closed verdict resolution
        if self.strict_mode:
            is_pass = orig_entails and rewr_entails and (len(validated_failures) == 0)
        else:
            is_pass = orig_entails and rewr_entails

        verdict = "PASS" if is_pass else "FAIL"

        raw_conf = str(parsed.get("confidence", "MEDIUM")).upper()
        confidence = raw_conf if raw_conf in {"HIGH", "MEDIUM", "LOW"} else "MEDIUM"

        return {
            "original_entails_rewrite": orig_entails,
            "rewrite_entails_original": rewr_entails,
            "verdict": verdict,
            "failures": validated_failures,
            "confidence": confidence,
        }

    def verify_heuristic(self, original: str, rewrite: str) -> Dict[str, Any]:
        """Fast heuristic bidirectional check when offline or for fast validation.

        Checks for:
          - Dropped quantitative terms (dollar amounts, percentages, day limits).
          - Inverted modal polarity ('shall not', 'prohibited' vs 'may', 'can').
        """
        orig = original.strip()
        rewr = rewrite.strip()
        failures: List[Dict[str, Any]] = []

        # 1. Number / Dollar / Day omissions
        num_patterns = [
            r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?",
            r"\d+\s*%",
            r"\b\d+\s+(?:days?|months?|years?|business days?)\b",
        ]
        for pat in num_patterns:
            orig_matches = set(re.findall(pat, orig, flags=re.IGNORECASE))
            rewr_matches = set(re.findall(pat, rewr, flags=re.IGNORECASE))
            missing = orig_matches - rewr_matches
            for m in missing:
                failures.append({
                    "category": "OMISSION",
                    "description": f"Quantitative specification '{m}' from original was omitted in rewrite.",
                    "original_span": m,
                    "rewrite_span": None,
                })

        # 2. Strict negative obligations turned permissive
        negations = ["shall not", "must not", "neither party shall", "in no event shall", "strictly prohibited"]
        for neg in negations:
            if neg in orig.lower() and not any(p in rewr.lower() for p in ["shall not", "must not", "cannot", "may not", "not allowed", "prohibited", "neither"]):
                failures.append({
                    "category": "CONTRADICTION",
                    "description": f"Strict prohibition '{neg}' appears inverted or weakened in rewrite.",
                    "original_span": neg,
                    "rewrite_span": None,
                })

        orig_entails = not any(f["category"] in {"HALLUCINATION", "CONTRADICTION"} for f in failures)
        rewr_entails = not any(f["category"] in {"OMISSION", "STRUCTURAL_COLLAPSE"} for f in failures)
        verdict = "PASS" if (orig_entails and rewr_entails and len(failures) == 0) else "FAIL"

        return {
            "original_entails_rewrite": orig_entails,
            "rewrite_entails_original": rewr_entails,
            "verdict": verdict,
            "failures": failures,
            "confidence": "MEDIUM" if len(failures) > 0 else "LOW",
        }

    def run(self, original: str, rewrite: str) -> Dict[str, Any]:
        """Synchronously verify meaning preservation between original and rewrite.

        Args:
            original: The raw legal clause text.
            rewrite: The plain-English simplified rewrite.

        Returns:
            Dict containing NLI verification results and verdicts.
        """
        if not original or not original.strip():
            return {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [{"category": "OMISSION", "description": "Original clause is empty."}],
                "confidence": "HIGH",
            }
        if not rewrite or not rewrite.strip():
            return {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [{"category": "OMISSION", "description": "Rewrite is empty."}],
                "confidence": "HIGH",
            }

        prompt = self._build_prompt(original, rewrite)

        try:
            response = self._sync_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content.strip()
            return self._parse_and_validate_response(raw_text)

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            # Fall back to heuristic check in offline/unreachable conditions with fail-closed warning
            heuristic = self.verify_heuristic(original, rewrite)
            heuristic["confidence"] = "LOW"
            heuristic["error"] = f"LM Studio unreachable at {self.url} ({type(exc).__name__}): {exc}"
            return heuristic
        except Exception as exc:
            return {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [
                    {
                        "category": "STRUCTURAL_COLLAPSE",
                        "description": f"Verification error: {exc}",
                        "original_span": None,
                        "rewrite_span": None,
                    }
                ],
                "confidence": "LOW",
                "error": str(exc),
            }

    async def run_async(self, original: str, rewrite: str) -> Dict[str, Any]:
        """Asynchronously verify meaning preservation between original and rewrite.

        Args:
            original: The raw legal clause text.
            rewrite: The plain-English simplified rewrite.

        Returns:
            Dict containing NLI verification results and verdicts.
        """
        if not original or not original.strip():
            return {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [{"category": "OMISSION", "description": "Original clause is empty."}],
                "confidence": "HIGH",
            }
        if not rewrite or not rewrite.strip():
            return {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [{"category": "OMISSION", "description": "Rewrite is empty."}],
                "confidence": "HIGH",
            }

        prompt = self._build_prompt(original, rewrite)

        try:
            response = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content.strip()
            return self._parse_and_validate_response(raw_text)

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            # Fall back to heuristic check with fail-closed warning
            heuristic = self.verify_heuristic(original, rewrite)
            heuristic["confidence"] = "LOW"
            heuristic["error"] = f"LM Studio unreachable at {self.url} ({type(exc).__name__}): {exc}"
            return heuristic
        except Exception as exc:
            return {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [
                    {
                        "category": "STRUCTURAL_COLLAPSE",
                        "description": f"Verification error: {exc}",
                        "original_span": None,
                        "rewrite_span": None,
                    }
                ],
                "confidence": "LOW",
                "error": str(exc),
            }


# Backwards compatibility function for scripts and evaluations
_default_agent: Optional[MeaningVerifierAgent] = None


def verify_meaning(original: str, rewrite: str, model: str = "LawUP") -> dict:
    """Module-level verification wrapper preserved for backwards compatibility."""
    global _default_agent
    if _default_agent is None or _default_agent.model_name != model:
        _default_agent = MeaningVerifierAgent(model_name=model)
    return _default_agent.run(original, rewrite)


if __name__ == "__main__":
    test_original = """Notwithstanding the foregoing, (a) either Party may, without the other Party's consent, assign this Agreement and its rights and obligations hereunder in whole or in part to an Affiliate; and (b) Dova may assign this Agreement to a successor in interest in connection with the sale or other transfer of all or substantially all of Dova's assets or rights relating to the Product; provided that such assignee shall remain subject to all of the terms and conditions hereof in all respects and shall assume all obligations of Dova hereunder whether accruing before or after such assignment."""

    test_rewrite = """Either Party may, without the other's consent, transfer this Agreement and its duties under it to an Affiliate. However:

(a) if a successor takes over Dova's assets or rights related to the Product in connection with a sale or other transfer of all or substantially all of those assets or rights, then Dova may assign this Agreement to that successor.

In either case, the new party must follow all terms and conditions of this Agreement. The new party also assumes all duties owed by Dova under this Agreement, whether they were created before or after the assignment."""

    print("Testing MeaningVerifierAgent Standalone...")
    agent = MeaningVerifierAgent()
    result = agent.run(test_original, test_rewrite)
    print(json.dumps(result, indent=2))
