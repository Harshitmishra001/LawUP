"""Grounding Agent — semantic retrieval of supporting statutes and industry norms."""

import os
from typing import List, Dict, Any

from backend.app.retrieval import GroundingRetriever


# Default corpus path relative to project root
_DEFAULT_CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "grounding_corpus",
    "grounding_corpus.json",
)


class GroundingAgent:
    """Grounding Agent — retrieves supporting statute/industry norm text for flagged risks.

    Pipeline step: 4 (reference Section 5 of PLAN.md)

    Wraps the GroundingRetriever (Sentence-Transformers semantic search) and
    provides a clean, agent-consistent interface for the orchestrator.
    """

    def __init__(self, corpus_path: str = None, top_k: int = 1):
        """Initialise the grounding agent.

        Args:
            corpus_path: Absolute or relative path to grounding_corpus.json.
                         Defaults to the standard project location.
            top_k: Number of grounding references to retrieve per risk category.
        """
        if corpus_path is None:
            corpus_path = _DEFAULT_CORPUS_PATH
        self.top_k = top_k
        self._retriever = GroundingRetriever(corpus_path=corpus_path)

    def run(self, risks: List[Dict[str, Any]], clause_text: str) -> List[Dict[str, Any]]:
        """Retrieve grounding references for each identified risk.

        Args:
            risks: List of risk dicts from the ClassificationAgent, each with
                   at least a 'risk' key (e.g. {'risk': 'Liability Exposure', ...}).
            clause_text: The original clause text, used as the semantic query.

        Returns:
            List of grounding reference dicts, each containing 'risk_category',
            'reference_text', 'authority_type', 'scope', and 'similarity_score'.
        """
        if not risks:
            return []

        grounding_refs = []
        seen_categories = set()

        for risk in risks:
            category = risk.get("risk")
            if not category or category in seen_categories:
                continue
            seen_categories.add(category)

            try:
                refs = self._retriever.retrieve_grounding(
                    risk_category=category,
                    clause_text=clause_text,
                    top_k=self.top_k,
                )
                grounding_refs.extend(refs)
            except Exception as e:
                print(f"WARNING: Grounding retrieval failed for category '{category}': {e}")

        return grounding_refs


if __name__ == "__main__":
    agent = GroundingAgent()

    test_risks = [{"risk": "Liability Exposure", "sub_category": "Uncapped Liability"}]
    test_clause = (
        "The Receiving Party shall indemnify and hold harmless the Disclosing Party "
        "against any consequential damages arising from a breach."
    )

    print("Testing GroundingAgent...")
    refs = agent.run(risks=test_risks, clause_text=test_clause)
    import json
    print(json.dumps(refs, indent=2))
