import os
import sys
import time

# Add project root to sys.path so shared/ and backend/ are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from shared.prompt_utils import format_prompt
from backend.app.agents.orchestrator import Orchestrator

app = FastAPI(
    title="LawUP API",
    description="Clause-by-clause legal risk analysis pipeline backed by fine-tuned SmolLM3 models.",
    version="0.4.0",
)

# LM Studio URL — configure via environment variable
lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

# Initialise the orchestrator once at startup (loads Sentence-Transformers model etc.)
print("Initialising LawUP pipeline orchestrator...")
orchestrator = Orchestrator(lm_studio_url=lm_studio_url)

# Load tokenizer for input validation on /simplify
print("Loading tokenizer for input validation...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
print("Ready.")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ClauseRequest(BaseModel):
    clause: str


class SimplifyResponse(BaseModel):
    rewrite: str
    latency_ms: float


class AnalyzeResponse(BaseModel):
    original_clause: str
    simplified_rewrite: str
    risks_identified: list
    grounding_references: list
    verifier_report: dict
    total_latency_ms: float
    stage_latencies_ms: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "lm_studio_url": lm_studio_url}


@app.post("/simplify", response_model=SimplifyResponse)
async def simplify_clause(request: ClauseRequest):
    """
    Stage 2 only — simplify a single clause into plain English.

    Validates that the formatted prompt stays within 1024 tokens before
    forwarding to the LoRA-adapted SmolLM3 model via LM Studio.
    """
    start_time = time.time()

    if not request.clause or not request.clause.strip():
        raise HTTPException(status_code=400, detail="Input clause cannot be empty.")

    prompt = format_prompt(request.clause)

    # Exact token validation on the formatted prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Formatted prompt exceeds 1024 tokens (got {len(tokens)}).",
        )

    try:
        rewrite = await orchestrator.simplifier.run_async(request.clause.strip())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Simplifier unavailable: {str(e)}")

    return SimplifyResponse(
        rewrite=rewrite,
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_clause(request: ClauseRequest):
    """
    End-to-end LawUP pipeline — runs all four stages for a single clause:

        Stage 2: SimplifierAgent   — plain-English rewrite via LoRA SmolLM3
        Stage 3: ClassificationAgent — multi-label risk detection via LoRA SmolLM3
        Stage 4: GroundingAgent    — semantic retrieval from statute corpus
        Stage 5a: MeaningVerifier  — LLM-based bi-directional entailment check

    All stages are fail-closed: a stage failure returns a safe empty/error value
    rather than crashing the whole request.
    """
    clause = request.clause.strip()
    if not clause:
        raise HTTPException(status_code=400, detail="Input clause cannot be empty.")

    try:
        result = await orchestrator.run(clause)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    return AnalyzeResponse(
        original_clause=result.original_clause,
        simplified_rewrite=result.simplified_rewrite,
        risks_identified=result.risks_identified,
        grounding_references=result.grounding_references,
        verifier_report=result.verifier_report,
        total_latency_ms=result.total_latency_ms,
        stage_latencies_ms=result.stage_latencies_ms,
    )
