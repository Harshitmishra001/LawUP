# Audit Report: LawUP

## Summary
All planned milestones and audit gaps are now fully resolved. The multi-agent pipeline connects Ingestion (PDF, DOCX, TXT, MD), fine-tuned SmolLM3-3B QLoRA Simplifier, 11-category CUAD Classifier, Sentence-Transformers Grounding Retriever, Bidirectional Meaning Verifier (NLI), Citation Verifier, and a complete interactive web dashboard in `frontend/`. All CI quality gates, unit tests, and validation scripts pass cleanly.

## Claims vs Reality

| Feature | Claimed where | Status | Evidence |
|---|---|---|---|
| Ingestion Agent | `README.md:58` | REAL | `backend/app/agents/ingestion.py` supports PDF, DOCX, TXT segmentation, wired to `orchestrator.run_document` and API routes. |
| Simplifier Agent | `README.md:59` | REAL | `backend/app/agents/simplifier.py:11-92` calls local LM Studio with greedy decoding & `<\|endoftext\|>` stop token. Fine-tuned model live on HF. |
| Classifier Agent | `README.md:60` | REAL | `backend/app/agents/classifier.py:22-68` outputs structured JSON risks. Benchmark verified on CUAD test split (0.7294 Micro F1). |
| Grounding Retriever | `README.md:61` | REAL | `backend/app/retrieval.py:8-74` runs Sentence-Transformers semantic search over `grounding_corpus/grounding_corpus.json` (60 entries). 76.9% accuracy verified by `backend/tests/test_retrieval.py:50`. |
| Meaning Verifier (NLI) | `README.md:62` | REAL | `backend/app/agents/verifier_meaning.py` implements bidirectional NLI entailment logic, async/sync APIs, fail-closed parsing, and heuristic fallback. |
| Citation Verifier | `README.md:63` | REAL | `backend/app/agents/verifier_citation.py` validates citation support against risk claims and drops ungrounded references. |
| Frontend / Report UI | `README.md:64` | REAL | Interactive web dashboard in `frontend/` with contract upload (PDF/DOCX/TXT), side-by-side legalese vs plain English, risk tags, statutory grounding, bidirectional NLI badges, and stage latency metrics. |
| End-to-End Orchestrator | `README.md:131` | REAL | `backend/app/agents/orchestrator.py` wires Ingestion ➔ Simplifier ➔ Classifier ➔ Grounder ➔ Verifier with non-blocking async execution and latency tracking. |
| CI Quality Gates | `README.md:180` | REAL | Schema validation, retrieval precision, document ingestion, meaning verification, and data quality tests all execute and pass in CI. |

---

## Issues by Severity

### Critical
1. **CI Data Quality Gate Silently Skips on GitHub** (`data/scripts/tests/test_data_quality.py`):
   *Status:* [RESOLVED]
   Committed `data/scripts/tests/fixtures/sample_simplification_pairs.jsonl` and updated `test_data_quality.py` with fallback fixture logic. All 4 tests execute and pass in CI without skipping.

2. **Incomplete `backend/requirements.txt` Causes Immediate Startup Crash** (`backend/requirements.txt`):
   *Status:* [RESOLVED]
   Added `sentence-transformers`, `torch`, `httpx`, and file parsing dependencies (`pypdf`, `python-docx`) to `backend/requirements.txt`.

### High
3. **IngestionAgent Disconnected From API and Pipeline** (`backend/app/agents/ingestion.py`):
   *Status:* [RESOLVED]
   Wired `IngestionAgent` directly into `Orchestrator.run_document()` and added `/ingest`, `/ingest/file`, `/analyze-document`, and `/analyze-document/file` endpoints.

4. **Complete Backend Outage Returns HTTP 200 OK After a 35-Second Hang** (`backend/app/agents/orchestrator.py`):
   *Status:* [RESOLVED]
   Implemented immediate early-exit with HTTP 503 on primary backend connection failure and fail-closed handling across all agent stages.

### Medium
5. **Synchronous Blocking Network & Inference Calls in Async Endpoint** (`backend/app/agents/orchestrator.py`):
   *Status:* [RESOLVED]
   Migrated agent inference calls to `AsyncOpenAI` client and non-blocking event-loop delegates (`asyncio.to_thread`).

6. **Dead and Stubbed Training Scaffolding** (`training/data_loader.py`):
   *Status:* [RESOLVED]
   Cleaned up dead scaffolding; active data preparation pipeline is maintained in `data/scripts/`.

### Low
7. **Empty Models Package** (`backend/app/models/`):
   *Status:* [RESOLVED]
   Centralized schemas into `backend/app/models/schemas.py` and exported through `backend/app/models/__init__.py`.

8. **Hardcoded User Paths in Scratch Scripts** (`scratch/extract_23_uids.py`, `scratch/map_uids.py`):
   *Status:* [RESOLVED]
   Replaced hardcoded Windows absolute paths with dynamic `Path.home()` and relative `Path(__file__)` references.

9. **Strict Equality in Grounding Retrieval Despite "Soft-Matching" Comment** (`backend/app/retrieval.py`):
   *Status:* [RESOLVED]
   Normalized casing and whitespace (`.strip().lower()`) across both queries and corpus entries.

---

## Placeholders and Hardcoded Values
- `backend/app/agents/verifier_citation.py`: Fully implemented with fail-closed category and similarity checks.
- `training/data_loader.py`: Removed unused stub file.
- `scratch/extract_23_uids.py`: Updated to use dynamic `Path.home()` and relative paths.
- `scratch/map_uids.py`: Updated to use dynamic `Path.home()` and relative paths.
- LM Studio URLs: Configured to read `LM_STUDIO_URL` from environment variables across all agents with local fallback.

---

## Incomplete and Dead Files
- `backend/app/agents/verifier_citation.py`: Implemented CitationVerifier (REAL).
- `training/data_loader.py`: Removed.
- `backend/app/models/__init__.py`: Cleanly exports all Pydantic schemas.
- `frontend/`: Interactive web dashboard implemented with file upload and side-by-side analysis (REAL).

---

## Verified by Running

| Command | Result | Notes |
|---|---|---|
| `ruff check .` | PASSED (0 errors) | Formatting rules configured in `pyproject.toml` |
| `python scripts/validate_corpus_schema.py` | PASSED | Validated all 60 entries in `grounding_corpus.json` |
| `pytest data/scripts/tests/test_data_quality.py -v` | PASSED (4 passed in 1.28s) | Executes cleanly both locally and in CI using fixture fallback |
| `pytest backend/tests/test_ingestion.py -v` | PASSED (7 passed in 0.95s) | Tests PDF, DOCX, TXT parsing and regex segmentation |
| `pytest backend/tests/test_verifier_meaning.py -v` | PASSED (11 passed in 4.70s) | Tests bidirectional NLI entailment, heuristic fallback, and API models |
| `pytest backend/tests/test_verifier_citation.py -v` | PASSED (8 passed in 35.18s) | Tests fail-closed citation grounding verification |
| `pytest backend/tests/test_dashboard_ui.py -v` | PASSED (4 passed in 61.50s) | Tests static dashboard serving, assets, and health probe routing |
| `PYTHONPATH=. pytest backend/tests/test_retrieval.py -v` | PASSED (1 passed in 78.86s) | Evaluated 26 test queries; scored 76.92% (threshold: >= 75%) |
| `GET /health` via FastAPI TestClient | PASSED (HTTP 200) | `{'status': 'ok', 'lm_studio_url': 'http://localhost:1234/v1'}` |


---

## What Actually Works
1. **Fine-Tuned Models:** Both `lawup-simplifier-smollm3-3b` and `lawup-classifier-smollm3-3b` exist, are public on Hugging Face, and export valid Q8_0 GGUFs.
2. **Retrieval Grounding Engine (`backend/app/retrieval.py`):** In-memory cosine similarity search over 60 curated legal standards across 12 risk categories achieves 76.92% top-1 accuracy.
3. **Pipeline Orchestrator (`backend/app/agents/orchestrator.py`):** Successfully initialises and coordinates agent calls with per-stage latency telemetry.
4. **FastAPI Application (`backend/app/api/main.py`):** Liveness probes, token validation on `/simplify` (1024 token ceiling via HuggingFace tokenizer), and OpenAPI docs at `/docs` function properly.
5. **Corpus Validation (`scripts/validate_corpus_schema.py`):** Enforces strict schema conformity (`risk_category`, `sub_category`, `reference_text`, `authority_type`, `scope`).

---

## Prioritized Fix Plan & Resolution Status

1. [x] **Add missing dependencies to `backend/requirements.txt`** (`backend/requirements.txt:1-11`): Added `sentence-transformers`, `torch`, `httpx`, `pypdf`, `python-docx`, `python-multipart`.
2. [x] **Fix CI test skipping for data quality** (`data/scripts/tests/test_data_quality.py`): Committed test fixture `data/scripts/tests/fixtures/sample_simplification_pairs.jsonl` with automatic fallback logic in `test_data_quality.py`.
3. [x] **Fail fast on backend disconnection in `/analyze`** (`backend/app/agents/orchestrator.py`, `backend/app/api/main.py`): Added `ConnectionError` propagation and HTTP 503 Service Unavailable handling across `/analyze`, `/analyze-document`, and `/analyze-document/file`.
4. [x] **Wire `IngestionAgent` to the API** (`backend/app/api/main.py`, `backend/app/agents/ingestion.py`): Built multi-format document parser (PDF, DOCX, TXT/MD), added `/ingest`, `/ingest/file`, `/analyze-document`, and 7 unit tests.
5. [x] **Convert synchronous calls in `Orchestrator.run` to non-blocking** (`backend/app/agents/orchestrator.py`): Wired `MeaningVerifierAgent.run_async` and wrapped `classifier.run` / `grounder.run` in `asyncio.to_thread`.
6. [x] **Normalize risk categories in `retrieve_grounding`** (`backend/app/retrieval.py:46-51`): Implemented case-insensitive and whitespace-stripped matching for risk categories.
7. [x] **Delete dead stub `training/data_loader.py`**: Removed unused file with `git rm`.
8. [x] **Relocate Pydantic models to `backend/app/models/`** (`backend/app/models/schemas.py`): Extracted schemas to `schemas.py` and exported via `backend/app/models/__init__.py`.
9. [x] **Remove hardcoded machine paths from scratch scripts** (`scratch/extract_23_uids.py`, `scratch/map_uids.py`): Replaced machine-specific absolute paths with relative `pathlib.Path(__file__)` paths.
10. [x] **Add pytest directory ignores to `pyproject.toml`** (`pyproject.toml`): Added `[tool.pytest.ini_options]` with `norecursedirs` and `pythonpath`.
11. [x] **Implement Citation Grounding Verifier** (`backend/app/agents/verifier_citation.py`): Built Stage 5b verifier enforcing category matching and similarity thresholding with fail-closed filtering.
12. [x] **Update Dockerfile and Environment Config** (`backend/Dockerfile`, `.env.example`): Added healthcheck probe, grounding corpus volume copy, port mapping, and sample environment variables.
13. [x] **Implement Interactive Web Dashboard** (`frontend/`): Built responsive contract review UI with document file upload, side-by-side comparison, risk tags, statutory grounding, NLI verification badges, and stage latency metrics.
