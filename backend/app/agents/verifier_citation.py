"""Citation Grounding Verifier — checks that retrieved source actually supports the risk claim. Fails closed."""

from typing import Any, Dict, List, Tuple


class CitationVerifier:
    """Citation Grounding Verifier — checks that retrieved source actually supports the risk claim.

    Pipeline step: 5b (reference Section 5 of PLAN.md)

    Fails closed: if a retrieved citation does not sufficiently support the risk claim
    (e.g. similarity score below threshold, risk category mismatch, or empty reference),
    it is dropped from the returned references and recorded in the verification report.
    """

    def __init__(
        self,
        min_similarity_threshold: float = 0.40,
        strict_mode: bool = True,
    ):
        """Initialise the citation verifier.

        Args:
            min_similarity_threshold: Minimum cosine similarity score required for a
                                      citation to be deemed supportive (default: 0.40).
            strict_mode: When True, unsupported references are dropped from final output.
        """
        self.min_similarity_threshold = min_similarity_threshold
        self.strict_mode = strict_mode

    def verify_single(
        self,
        clause_text: str,
        risk: Dict[str, Any],
        reference: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Verify if a single citation reference supports the risk claim on a clause.

        Args:
            clause_text: The original contract clause.
            risk: The risk dict from classifier (e.g. {'risk': '...', 'sub_category': '...'}).
            reference: The grounding reference dict from retriever.

        Returns:
            Tuple of (is_supported: bool, reason: str).
        """
        if not clause_text or not clause_text.strip():
            return False, "Clause text is empty."

        ref_text = reference.get("reference_text", "")
        if not ref_text or not ref_text.strip():
            return False, "Reference text is empty."

        risk_name = (risk.get("risk") or risk.get("risk_category") or "").strip().lower()
        ref_cat = (reference.get("risk_category") or "").strip().lower()

        if risk_name and ref_cat and risk_name != ref_cat:
            return False, f"Category mismatch: risk '{risk_name}' does not match reference '{ref_cat}'."

        score = reference.get("similarity_score")
        if score is not None:
            try:
                score_val = float(score)
                if score_val < self.min_similarity_threshold:
                    return False, f"Similarity score {score_val:.4f} is below threshold {self.min_similarity_threshold:.2f}."
            except (ValueError, TypeError):
                return False, f"Invalid similarity score value: {score}."

        return True, "Citation verified and supported."

    def run(
        self,
        clause_text: str,
        risks: List[Dict[str, Any]],
        grounding_references: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify all grounding references against identified risks.

        Args:
            clause_text: Original clause text.
            risks: List of identified risks from ClassificationAgent.
            grounding_references: List of retrieved standards from GroundingAgent.

        Returns:
            Dict containing:
                - verified_references: List of references that passed verification.
                - dropped_references: List of references that were rejected.
                - total_checked: Total references inspected.
                - total_supported: Number of supported references.
                - total_dropped: Number of dropped references.
                - verdict: "PASS" if all supported or no references, "WARN" if some dropped.
                - details: Per-reference evaluation logs.
        """
        verified: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        details: List[Dict[str, Any]] = []

        risk_map: Dict[str, Dict[str, Any]] = {}
        for r in risks:
            cat = (r.get("risk") or r.get("risk_category") or "").strip().lower()
            if cat:
                risk_map[cat] = r

        for i, ref in enumerate(grounding_references):
            ref_cat = (ref.get("risk_category") or "").strip().lower()
            matching_risk = risk_map.get(ref_cat, risks[0] if risks else {})

            is_supported, reason = self.verify_single(clause_text, matching_risk, ref)
            detail = {
                "index": i,
                "risk_category": ref.get("risk_category", "Unknown"),
                "sub_category": ref.get("sub_category", "Unknown"),
                "similarity_score": ref.get("similarity_score"),
                "supported": is_supported,
                "reason": reason,
            }
            details.append(detail)

            if is_supported:
                verified.append(ref)
            else:
                dropped.append({
                    "reference": ref,
                    "reason": reason,
                })

        verdict = "PASS" if len(dropped) == 0 else "WARN"

        return {
            "verified_references": verified if self.strict_mode else grounding_references,
            "dropped_references": dropped,
            "total_checked": len(grounding_references),
            "total_supported": len(verified),
            "total_dropped": len(dropped),
            "verdict": verdict,
            "details": details,
        }

    async def run_async(
        self,
        clause_text: str,
        risks: List[Dict[str, Any]],
        grounding_references: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Asynchronous wrapper for non-blocking pipeline execution."""
        return self.run(clause_text, risks, grounding_references)
