"""Pipeline Orchestrator — deterministic, fail-closed multi-agent pipeline for LawUP."""

import asyncio
import os
import time
from typing import Any, BinaryIO, Dict, List, Optional, Union
from pathlib import Path

from backend.app.agents.simplifier import SimplifierAgent
from backend.app.agents.classifier import ClassificationAgent
from backend.app.agents.grounding import GroundingAgent
from backend.app.agents.ingestion import IngestionAgent, ClauseUnit
from backend.app.agents.verifier_meaning import MeaningVerifierAgent


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


class DocumentResult:
    """Structured result from analyzing an entire multi-clause contract."""

    def __init__(
        self,
        total_clauses: int,
        clauses: List[Dict[str, Any]],
        total_latency_ms: float,
        risk_summary: Dict[str, int],
        verification_summary: Dict[str, int],
    ):
        self.total_clauses = total_clauses
        self.clauses = clauses
        self.total_latency_ms = total_latency_ms
        self.risk_summary = risk_summary
        self.verification_summary = verification_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_clauses": self.total_clauses,
            "clauses": self.clauses,
            "total_latency_ms": self.total_latency_ms,
            "risk_summary": self.risk_summary,
            "verification_summary": self.verification_summary,
        }


class Orchestrator:
    """Pipeline Orchestrator — deterministic, fail-closed multi-agent pipeline.

    Pipeline step: 6 (reference Section 5 of PLAN.md)

    Chains agents in the following order:
        IngestionAgent ➔ SimplifierAgent ➔ ClassificationAgent ➔ GroundingAgent ➔ MeaningVerifier

    Fail-closed design: each stage is wrapped in a try/except. On failure,
    the stage returns a safe empty/error value rather than propagating an
    exception and crashing the full pipeline.
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
        self.ingestion = IngestionAgent()
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
        self.verifier = MeaningVerifierAgent(
            model_name=verifier_model,
            lm_studio_url=lm_studio_url,
        )
        self.verifier_model = verifier_model

    def parse_document(
        self,
        input_data: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
    ) -> List[ClauseUnit]:
        """Stage 1: Extract structured clauses from document without running LLM pipeline."""
        return self.ingestion.ingest(input_data, filename=filename)

    async def run(self, clause_text: str) -> OrchestratorResult:
        """Run single-clause pipeline (Stages 2–5).

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
            risks = await asyncio.to_thread(self.classifier.run, clause)
        except Exception as e:
            print(f"[Orchestrator] Classifier failed: {e}")
            risks = []
        stage_latencies["classifier_ms"] = (time.time() - t) * 1000

        # --- Stage 4: Grounding Retrieval ---
        t = time.time()
        try:
            grounding_refs = await asyncio.to_thread(self.grounder.run, risks=risks, clause_text=clause)
        except Exception as e:
            print(f"[Orchestrator] Grounder failed: {e}")
            grounding_refs = []
        stage_latencies["grounder_ms"] = (time.time() - t) * 1000

        # --- Stage 5a: Meaning Verification ---
        t = time.time()
        try:
            verifier_report = await self.verifier.run_async(
                original=clause,
                rewrite=rewrite,
            )
        except Exception as e:
            print(f"[Orchestrator] Verifier failed: {e}")
            verifier_report = {
                "original_entails_rewrite": False,
                "rewrite_entails_original": False,
                "verdict": "FAIL",
                "failures": [{"category": "STRUCTURAL_COLLAPSE", "description": f"Verifier exception: {e}"}],
                "confidence": "LOW",
                "error": str(e),
            }
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

    async def run_document(
        self,
        document_data: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
        max_clauses: int = 50,
    ) -> DocumentResult:
        """Run full end-to-end multi-agent pipeline on a complete document (Stages 1–5).

        Args:
            document_data: Raw text, PDF/DOCX bytes, or stream.
            filename: Optional filename for format resolution.
            max_clauses: Maximum number of clauses to analyze in one run.

        Returns:
            DocumentResult containing clause analyses, summary metrics, and aggregate latency.
        """
        start_time = time.time()

        # Stage 1: Ingestion
        clause_units = self.parse_document(document_data, filename=filename)
        selected_units = clause_units[:max_clauses]

        analyzed_clauses: List[Dict[str, Any]] = []
        risk_summary: Dict[str, int] = {}
        verification_summary: Dict[str, int] = {"PASS": 0, "FAIL": 0, "ERROR": 0}

        for unit in selected_units:
            analysis = await self.run(unit.text)
            clause_record = {
                "index": unit.index,
                "heading": unit.heading,
                "word_count": unit.word_count,
                "analysis": analysis.to_dict(),
            }
            analyzed_clauses.append(clause_record)

            # Accumulate risk categories
            for r in analysis.risks_identified:
                cat = r.get("risk", "Unknown Risk")
                risk_summary[cat] = risk_summary.get(cat, 0) + 1

            # Accumulate verification verdict
            v = analysis.verifier_report.get("verdict", "ERROR")
            if v in verification_summary:
                verification_summary[v] += 1
            else:
                verification_summary["ERROR"] += 1

        total_latency = (time.time() - start_time) * 1000

        return DocumentResult(
            total_clauses=len(clause_units),
            clauses=analyzed_clauses,
            total_latency_ms=total_latency,
            risk_summary=risk_summary,
            verification_summary=verification_summary,
        )

    def run_sync(self, clause_text: str) -> OrchestratorResult:
        """Synchronous entry point for scripts and CLI tools."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.run(clause_text)).result()
            else:
                return loop.run_until_complete(self.run(clause_text))
        except RuntimeError:
            return asyncio.run(self.run(clause_text))
