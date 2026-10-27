"""LawUP Multi-Agent System."""

from backend.app.agents.ingestion import IngestionAgent, ClauseUnit
from backend.app.agents.simplifier import SimplifierAgent
from backend.app.agents.classifier import ClassificationAgent
from backend.app.agents.grounding import GroundingAgent
from backend.app.agents.verifier_meaning import MeaningVerifierAgent, verify_meaning
from backend.app.agents.verifier_citation import CitationVerifier
from backend.app.agents.orchestrator import Orchestrator, OrchestratorResult, DocumentResult

__all__ = [
    "IngestionAgent",
    "ClauseUnit",
    "SimplifierAgent",
    "ClassificationAgent",
    "GroundingAgent",
    "MeaningVerifierAgent",
    "verify_meaning",
    "CitationVerifier",
    "Orchestrator",
    "OrchestratorResult",
    "DocumentResult",
]
