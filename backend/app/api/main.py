import os
import sys
import time

# Add root to sys.path so we can import shared
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import httpx
from transformers import AutoTokenizer
from shared.prompt_utils import format_prompt

app = FastAPI(title="LawUP Simplifier API (LMStudio Backend)")

# Initialize OpenAI client pointed at LMStudio's local server or env var
lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
client = AsyncOpenAI(base_url=lm_studio_url, api_key="lm-studio")

# Load only the tokenizer to compute true token counts for validation
# This is lightweight and doesn't require downloading/loading model weights
print("Loading tokenizer for input validation...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

class SimplifyRequest(BaseModel):
    clause: str

class SimplifyResponse(BaseModel):
    rewrite: str
    latency_ms: float

@app.post("/simplify", response_model=SimplifyResponse)
async def simplify_clause(request: SimplifyRequest):
    start_time = time.time()
    
    if not request.clause or not request.clause.strip():
        raise HTTPException(status_code=400, detail="Input clause cannot be empty.")
        
    prompt = format_prompt(request.clause)
    
    # EXACT 1024 Token validation on the FORMATTED prompt using the real tokenizer
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    num_tokens = len(tokens)
    
    if num_tokens > 1024:
        raise HTTPException(
            status_code=400, 
            detail=f"Formatted prompt exceeds maximum length of 1024 tokens (got {num_tokens})."
        )
        
    try:
        # Deterministic greedy decoding via LMStudio
        response = await client.completions.create(
            model="lawup-3b-q8", # Name doesn't strictly matter for LMStudio, but good practice
            prompt=prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["<|endoftext|>"] # Explicitly stop on the training EOS token to prevent run-on generation
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"LM Studio not reachable at {lm_studio_url}. Is the local server running?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to communicate with LMStudio: {str(e)}")
        
    rewrite = response.choices[0].text.strip()
    latency_ms = (time.time() - start_time) * 1000
    
    return SimplifyResponse(
        rewrite=rewrite,
        latency_ms=latency_ms
    )


class AnalyzeResponse(BaseModel):
    original_clause: str
    simplified_rewrite: str
    risks_identified: list
    grounding_references: list
    verifier_report: dict
    total_latency_ms: float

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_clause(request: SimplifyRequest):
    """
    End-to-end LawUP Orchestrator Pipeline:
    1. Clause Extraction -> (Passed in request)
    2. Simplifier LoRA -> (AsyncOpenAI call to LM Studio)
    3. Classifier LoRA -> (ClassificationAgent)
    4. Grounding Retrieval -> (GroundingRetriever)
    5. NLI Meaning Verifier -> (verify_meaning)
    """
    start_time = time.time()
    clause = request.clause.strip()

    if not clause:
        raise HTTPException(status_code=400, detail="Input clause cannot be empty.")

    # We import agents here to avoid blocking startup if dependencies are missing
    from backend.app.agents.classifier import ClassificationAgent
    from backend.app.retrieval import GroundingRetriever
    from backend.app.agents.verifier_meaning import verify_meaning

    # --- Step 2: Simplification ---
    prompt = format_prompt(clause)
    try:
        response = await client.completions.create(
            model="lawup-3b-q8",
            prompt=prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["<|endoftext|>"]
        )
        rewrite = response.choices[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simplifier failed: {str(e)}")

    # --- Step 3: Classification ---
    try:
        classifier = ClassificationAgent(lm_studio_url=lm_studio_url)
        risks = classifier.run(clause)
    except Exception as e:
        print(f"Classifier error: {e}")
        risks = []

    # --- Step 4: Grounding Retrieval ---
    grounding_refs = []
    try:
        retriever = GroundingRetriever(corpus_path="grounding_corpus/grounding_corpus.json")
        for risk in risks:
            cat = risk.get("risk")
            if cat:
                refs = retriever.retrieve_grounding(risk_category=cat, clause_text=clause, top_k=1)
                grounding_refs.extend(refs)
    except Exception as e:
        print(f"Retrieval error: {e}")

    # --- Step 5: Meaning Verifier ---
    try:
        verifier_report = verify_meaning(original=clause, rewrite=rewrite, model="lawup-verifier")
    except Exception as e:
        print(f"Verifier error: {e}")
        verifier_report = {"error": str(e)}

    latency = (time.time() - start_time) * 1000

    return AnalyzeResponse(
        original_clause=clause,
        simplified_rewrite=rewrite,
        risks_identified=risks,
        grounding_references=grounding_refs,
        verifier_report=verifier_report,
        total_latency_ms=latency
    )
