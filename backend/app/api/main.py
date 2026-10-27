import os
import sys
import time

# Add project root to sys.path so shared/ and backend/ are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer

from shared.prompt_utils import format_prompt
from backend.app.agents.orchestrator import Orchestrator
from backend.app.models import (
    ClauseRequest,
    ContractTextRequest,
    SimplifyResponse,
    AnalyzeResponse,
    IngestResponse,
    DocumentAnalyzeResponse,
)

app = FastAPI(
    title="LawUP API",
    description="Clause-by-clause legal risk analysis pipeline backed by fine-tuned SmolLM3 models.",
    version="0.5.0",
)

# LM Studio URL — configure via environment variable
lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")

# Initialise the orchestrator once at startup (loads Sentence-Transformers model etc.)
print("Initialising LawUP pipeline orchestrator...")
orchestrator = Orchestrator(lm_studio_url=lm_studio_url)

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B", local_files_only=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    return _tokenizer


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "lm_studio_url": lm_studio_url}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_contract_text(request: ContractTextRequest):
    """
    Stage 1: Ingest and segment raw contract text into structured clause units.

    Extracts section headings (e.g. "1. SERVICES", "ARTICLE 4"), cleans text,
    and discards fragments below the minimum length threshold.
    """
    if not request.contract_text or not request.contract_text.strip():
        raise HTTPException(status_code=400, detail="contract_text cannot be empty.")

    try:
        clause_units = orchestrator.parse_document(request.contract_text)
        return IngestResponse(
            total_clauses=len(clause_units),
            clauses=[c.to_dict() for c in clause_units],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_contract_file(file: UploadFile = File(...)):
    """
    Stage 1 (File Upload): Ingest and parse PDF, DOCX, TXT, or MD documents.

    Extracts text and segments the document into structured clause units.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        clause_units = orchestrator.parse_document(content, filename=file.filename)
        return IngestResponse(
            total_clauses=len(clause_units),
            clauses=[c.to_dict() for c in clause_units],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


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
    tokens = get_tokenizer().encode(prompt, add_special_tokens=False)
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
    End-to-end LawUP pipeline for a single clause:

        Stage 2: SimplifierAgent     — plain-English rewrite via LoRA SmolLM3
        Stage 3: ClassificationAgent — multi-label risk detection via LoRA SmolLM3
        Stage 4: GroundingAgent      — semantic retrieval from statute corpus
        Stage 5a: MeaningVerifier    — LLM-based bi-directional entailment check

    All stages are fail-closed: a stage failure returns a safe empty/error value
    rather than crashing the whole request.
    """
    clause = request.clause.strip()
    if not clause:
        raise HTTPException(status_code=400, detail="Input clause cannot be empty.")

    try:
        result = await orchestrator.run(clause)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Inference backend unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    return AnalyzeResponse(
        original_clause=result.original_clause,
        simplified_rewrite=result.simplified_rewrite,
        risks_identified=result.risks_identified,
        grounding_references=result.grounding_references,
        verifier_report=result.verifier_report,
        citation_report=result.citation_report,
        total_latency_ms=result.total_latency_ms,
        stage_latencies_ms=result.stage_latencies_ms,
    )


@app.post("/analyze-document", response_model=DocumentAnalyzeResponse)
async def analyze_document_text(request: ContractTextRequest):
    """
    Full end-to-end multi-agent pipeline for an entire raw contract (Stages 1–5):

        Stage 1: IngestionAgent      — segmentation into structured clause units
        Stage 2: SimplifierAgent     — plain-English rewrites
        Stage 3: ClassificationAgent — multi-label risk identification
        Stage 4: GroundingAgent      — statutory & industry standard citations
        Stage 5a: MeaningVerifier    — bi-directional NLI verification
    """
    if not request.contract_text or not request.contract_text.strip():
        raise HTTPException(status_code=400, detail="contract_text cannot be empty.")

    try:
        max_c = request.max_clauses or 50
        doc_result = await orchestrator.run_document(request.contract_text, max_clauses=max_c)
        return DocumentAnalyzeResponse(
            total_clauses=doc_result.total_clauses,
            clauses=doc_result.clauses,
            total_latency_ms=doc_result.total_latency_ms,
            risk_summary=doc_result.risk_summary,
            verification_summary=doc_result.verification_summary,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Inference backend unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")


@app.post("/analyze-document/file", response_model=DocumentAnalyzeResponse)
async def analyze_document_file(file: UploadFile = File(...), max_clauses: int = 50):
    """
    Full document analysis via PDF, DOCX, TXT, or MD file upload.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        doc_result = await orchestrator.run_document(content, filename=file.filename, max_clauses=max_clauses)
        return DocumentAnalyzeResponse(
            total_clauses=doc_result.total_clauses,
            clauses=doc_result.clauses,
            total_latency_ms=doc_result.total_latency_ms,
            risk_summary=doc_result.risk_summary,
            verification_summary=doc_result.verification_summary,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Inference backend unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")


# ---------------------------------------------------------------------------
# Interactive Web Dashboard Mount
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_frontend_dir = os.path.join(_project_root, "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

