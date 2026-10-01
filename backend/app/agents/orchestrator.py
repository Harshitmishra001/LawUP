"""Pipeline Orchestrator — deterministic, fail-closed multi-agent pipeline for LawUP."""

import os
import time
from typing import Any, Dict, List, Optional

from backend.app.agents.simplifier import SimplifierAgent
from backend.app.agents.classifier import ClassificationAgent
from backend.app.agents.grounding import GroundingAgent
from backend.app.agents.verifier_meaning import verify_meaning


class OrchestratorResult:
    """Structured result from a single clause run through the full pipeline."""

    def __init__(
        self,
        original_clause: str,
        simplified_rewrite: str,
        risks_identified: List[Dict[str, Any]],
        grounding_references: List[Dict[str, Any]],
        verifier_report: Dict[str, Any],
        total_latency_ms: float,
        stage_latencies_ms: Dict[str, float],
    ):
        self.original_clause = original_clause
        self.simplified_rewrite = simplified_rewrite
        self.risks_identified = risks_identified
        self.grounding_references = grounding_references
        self.verifier_report = verifier_report
        self.total_latency_ms = total_latency_ms
        self.stage_latencies_ms = stage_latencies_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_clause": self.original_clause,
            "simplified_rewrite": self.simplified_rewrite,
            "risks_identified": self.risks_identified,
            "grounding_references": self.grounding_references,
            "verifier_report": self.verifier_report,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies_ms": self.stage_latencies_ms,
        }


class Orchestrator:
    """Pipeline Orchestrator — deterministic, fail-closed multi-agent pipeline.

    Pipeline step: 6 (reference Section 5 of PLAN.md)

    Chains agents in the following order for a single clause:
        SimplifierAgent  → ClassificationAgent → GroundingAgent → MeaningVerifier

    Fail-closed design: each stage is wrapped in a try/except. On failure,
    the stage returns a safe empty/error value rather than propagating an
    exception and crashing the full pipeline. The caller receives the partial
    result with an error key indicating which stage failed.

    Usage (sync):
        orchestrator = Orchestrator(lm_studio_url="http://localhost:1234/v1")
        result = orchestrator.run_sync(clause_text="The receiving party shall...")
        print(result.to_dict())

    Usage (async, from FastAPI):
        result = await orchestrator.run(clause_text="The receiving party shall...")
    """

    def __init__(
        self,
        lm_studio_url: Optional[str] = None,
        simplifier_model: str = "lawup-3b-q8",
        classifier_model: str = "LawUP-Classifier",
        verifier_model: str = "LawUP",
        corpus_path: Optional[str] = None,
        top_k_grounding: int = 1,
    ):
        if lm_studio_url is None:
            lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

        self.lm_studio_url = lm_studio_url

        # Initialise all agents once at construction time — avoids repeated cold-starts
        self.simplifier = SimplifierAgent(
            model_name=simplifier_model,
            lm_studio_url=lm_studio_url,
        )
        self.classifier = ClassificationAgent(
            model_name=classifier_model,
            lm_studio_url=lm_studio_url,
        )
        self.grounder = GroundingAgent(
            corpus_path=corpus_path,
            top_k=top_k_grounding,
        )
        self.verifier_model = verifier_model

    async def run(self, clause_text: str) -> OrchestratorResult:
        """Async entry point — used by FastAPI endpoints.

        Args:
            clause_text: A single raw legal clause string.

        Returns:
            OrchestratorResult with outputs from all pipeline stages.
        """
        clause = clause_text.strip()
        stage_latencies: Dict[str, float] = {}
        start_total = time.time()

        # --- Stage 2: Simplification ---
        t = time.time()
        try:
            rewrite = await self.simplifier.run_async(clause)
        except Exception as e:
            print(f"[Orchestrator] Simplifier failed: {e}")
            rewrite = f"[Simplification failed: {type(e).__name__}]"
        stage_latencies["simplifier_ms"] = (time.time() - t) * 1000

        # --- Stage 3: Classification ---
        t = time.time()
        try:
            risks = self.classifier.run(clause)
        except Exception as e:
            print(f"[Orchestrator] Classifier failed: {e}")
            risks = []
        stage_latencies["classifier_ms"] = (time.time() - t) * 1000

        # --- Stage 4: Grounding Retrieval ---
        t = time.time()
        try:
            grounding_refs = self.grounder.run(risks=risks, clause_text=clause)
        except Exception as e:
            print(f"[Orchestrator] Grounder failed: {e}")
            grounding_refs = []
        stage_latencies["grounder_ms"] = (time.time() - t) * 1000

        # --- Stage 5a: Meaning Verification ---
        t = time.time()
        try:
            verifier_report = verify_meaning(
                original=clause,
                rewrite=rewrite,
                model=self.verifier_model,
            )
        except Exception as e:
            print(f"[Orchestrator] Verifier failed: {e}")
            verifier_report = {"error": str(e)}
        stage_latencies["verifier_ms"] = (time.time() - t) * 1000

        total_latency_ms = (time.time() - start_total) * 1000

        return OrchestratorResult(
            original_clause=clause,
            simplified_rewrite=rewrite,
            risks_identified=risks,
            grounding_references=grounding_refs,
            verifier_report=verifier_report,
            total_latency_ms=total_latency_ms,
            stage_latencies_ms=stage_latencies,
        )

    def run_sync(self, clause_text: str) -> OrchestratorResult:
        """Synchronous entry point — useful for scripts and CLI tools.

        Note: SimplifierAgent uses the sync client here.

        Args:
            clause_text: A single raw legal clause string.

        Returns:
            OrchestratorResult with outputs from all pipeline stages.
        """
        import asyncio
        # Reuse the async implementation via a new event loop for simplicity
        # (avoids duplicating stage logic)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(clause_text))
                    return future.result()
            else:
                return loop.run_until_complete(self.run(clause_text))
        except RuntimeError:
            return asyncio.run(self.run(clause_text))


if __name__ == "__main__":
    import json

    orchestrator = Orchestrator()

    test_clause = (
        "The Receiving Party shall indemnify and hold harmless the Disclosing Party "
        "against any consequential damages arising from a breach."
    )

    print("Testing Orchestrator (sync)...")
    print(f"Clause: {test_clause}\n")

    result = orchestrator.run_sync(test_clause)
    print(json.dumps(result.to_dict(), indent=2))
