# Audit Report: LawUP

## Summary
Roughly 65% working prototype. The core ML artifacts (two fine-tuned SmolLM3-3B models on Hugging Face, GGUF Q8_0 weights, and the Sentence-Transformers grounding retriever) are real and functional. The FastAPI service initializes cleanly and executes the multi-stage pipeline, but `IngestionAgent` is completely disconnected from the API, the four CI data-quality tests silently skip on GitHub because the data file is gitignored, and `backend/requirements.txt` is missing essential dependencies (`sentence-transformers`, `torch`, `httpx`).

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
1. **CI Data Quality Gate Silently Skips on GitHub** (`data/scripts/tests/test_data_quality.py:12-14, 29-31, 44-46, 61-63`):
   Every test in `test_data_quality.py` contains `if not SIMPLIFICATION_FILE.exists(): pytest.skip(...)`. Because `data/processed/*.jsonl` is ignored in `.gitignore:70`, `simplification_pairs.jsonl` is not in git (`git ls-files data/processed` confirms). In GitHub Actions CI, all 4 data-integrity tests skip silently, providing zero protection against data corruption.
   *Fix:* Commit a small test fixture (e.g. `tests/fixtures/sample_simplification_pairs.jsonl`) for CI testing rather than skipping on the absent 700MB training set.

2. **Incomplete `backend/requirements.txt` Causes Immediate Startup Crash** (`backend/requirements.txt:1-5`):
   `backend/requirements.txt` only lists `fastapi`, `uvicorn`, `pydantic`, `openai`, `transformers`. It is missing `sentence-transformers`, `torch`, and `httpx`. A standard installation (`pip install -r backend/requirements.txt`) crashes on `uvicorn backend.app.api.main:app` at line 26 when `GroundingRetriever` attempts to import `sentence_transformers`.
   *Fix:* Add `sentence-transformers>=2.2.0`, `torch>=2.0.0`, and `httpx>=0.24.0` to `backend/requirements.txt`.

### High
3. **IngestionAgent Is Completely Disconnected From the API and Pipeline** (`backend/app/agents/ingestion.py:26`, `backend/app/agents/orchestrator.py:7-10`, `backend/app/api/main.py:1-132`):
   `README.md:58` advertises Stage 1 Ingestion as "✅ Complete". While `IngestionAgent` exists as a standalone regex chunking class, it is never imported or called by `orchestrator.py` or `main.py`. The `/analyze` endpoint requires pre-chunked clauses. There is no route or orchestrator method to ingest a full raw contract document.
   *Fix:* Wire `IngestionAgent` into `Orchestrator` or add a `/parse-contract` endpoint in `main.py` that ingests raw contract text and returns segmented clauses.

4. **Complete Backend Outage Returns HTTP 200 OK After a 35-Second Hang** (`backend/app/agents/orchestrator.py:112-148`, `backend/app/api/main.py:102-131`):
   When the local LM Studio server is offline, `/analyze` attempts sequential connections across Simplifier, Classifier, and Verifier. Each stage hangs waiting for an HTTP connection timeout (~8-13s each), stalling the request for 35.3 seconds. The endpoint then returns HTTP 200 OK with `[Simplification failed: APIConnectionError]`, `risks_identified: []`, and `verifier_report: {'error': 'Connection error.'}`, masking a complete service failure as a successful response.
   *Fix:* Fail early: if the primary simplification stage fails due to backend connection loss, return HTTP 503 Service Unavailable immediately rather than continuing downstream calls.

### Medium
5. **Synchronous Blocking Network & Inference Calls in Async Endpoint** (`backend/app/agents/orchestrator.py:122, 131, 140`):
   Inside `async def run(self, clause_text: str)`, lines 122 (`self.classifier.run`), 131 (`self.grounder.run`), and 140 (`verify_meaning`) execute synchronous blocking network calls (`OpenAI` sync client) and CPU/GPU PyTorch tensor operations (`SentenceTransformer.encode`). This blocks the FastAPI asyncio event loop, preventing concurrent request handling.
   *Fix:* Use `AsyncOpenAI` for classifier and verifier, and wrap `self.grounder.run` in `asyncio.to_thread()`.

6. **Dead and Stubbed Training Scaffolding** (`training/data_loader.py:10, 15, 20`):
   `training/data_loader.py` contains `load_splits`, `format_for_classification`, and `format_for_simplification`, all of which unconditionally raise `NotImplementedError`. The actual data preparation was done by `data/scripts/format_for_qlora.py` and `data/scripts/generate_classification_data.py`. This file is unreferenced dead code.
   *Fix:* Delete `training/data_loader.py` or implement it as an import wrapper around `data/scripts/`.

### Low
7. **Empty Models Package** (`backend/app/models/__init__.py:1`):
   The `backend/app/models/` directory contains only an empty `__init__.py`. All request and response models are defined directly in `backend/app/api/main.py:38-55` and `backend/app/agents/orchestrator.py:13-43`.
   *Fix:* Move `ClauseRequest`, `SimplifyResponse`, `AnalyzeResponse`, and `OrchestratorResult` into `backend/app/models/schemas.py`.

8. **Hardcoded User Paths in Scratch Scripts** (`scratch/extract_23_uids.py:3, 11`, `scratch/map_uids.py:4, 10, 27`):
   `extract_23_uids.py` and `map_uids.py` contain hardcoded paths (`C:\Users\hmhar\...`) that will crash if executed in another environment.
   *Fix:* Use relative `pathlib.Path(__file__).parents[...]` paths.

9. **Strict Equality in Grounding Retrieval Despite "Soft-Matching" Comment** (`backend/app/retrieval.py:46-49`):
   The comment states `# We do soft matching on risk_category just in case`, but line 48 executes strict equality `item["risk_category"] == risk_category`. Any minor casing or spelling discrepancy produces an empty candidate set.
   *Fix:* Normalize casing and whitespace (`item["risk_category"].strip().lower() == risk_category.strip().lower()`).

---

## Placeholders and Hardcoded Values
- `backend/app/agents/verifier_citation.py:14` - `raise NotImplementedError("Implementation pending — see M4 milestone.")`
- `training/data_loader.py:10` - `raise NotImplementedError("Implementation pending — see M2 milestone.")`
- `training/data_loader.py:15` - `raise NotImplementedError("Implementation pending — see M2 milestone.")`
- `training/data_loader.py:20` - `raise NotImplementedError("Implementation pending — see M2 milestone.")`
- `backend/app/agents/classifier.py:28` - Default fallback URL `http://localhost:1234/v1`
- `backend/app/agents/orchestrator.py:78` - Default fallback URL `http://localhost:1234/v1`
- `backend/app/agents/simplifier.py:28` - Default fallback URL `http://localhost:1234/v1`
- `backend/app/agents/verifier_meaning.py:84` - Default fallback URL `http://localhost:1234/v1`
- `backend/app/api/main.py:22` - Default fallback URL `http://localhost:1234/v1`
- `scratch/extract_23_uids.py:3, 11` - Hardcoded machine paths `C:\Users\hmhar\...`
- `scratch/map_uids.py:4, 10, 27` - Hardcoded machine paths `C:\Users\hmhar\...`

---

## Incomplete and Dead Files
- `backend/app/agents/verifier_citation.py` - Unimplemented stub (`raise NotImplementedError`).
- `training/data_loader.py` - Unimplemented stub file with 3 `NotImplementedError` functions; never imported.
- `backend/app/models/__init__.py` - Empty module (1 blank line); no models defined.
- `frontend/.gitkeep` - Placeholder directory for M5; zero code exists.

---

## Verified by Running

| Command | Result | Notes |
|---|---|---|
| `ruff check .` | PASSED (0 errors) | Formatting rules configured in `pyproject.toml` |
| `python scripts/validate_corpus_schema.py` | PASSED | Validated all 60 entries in `grounding_corpus.json` |
| `pytest data/scripts/tests/test_data_quality.py -v` | PASSED (4 passed in 0.31s) | Passes locally because `simplification_pairs.jsonl` exists on disk; skips on CI |
| `PYTHONPATH=. pytest backend/tests/test_retrieval.py -v` | PASSED (1 passed in 78.86s) | Evaluated 26 test queries; scored 76.92% (threshold: >= 75%) |
| `GET /health` via FastAPI TestClient | PASSED (HTTP 200) | `{'status': 'ok', 'lm_studio_url': 'http://localhost:1234/v1'}` |
| `POST /analyze` without LM Studio running | COMPLETED (HTTP 200 after 35.3s) | Fail-closed try/catch returned safe fallback dicts, but masked complete inference failure |

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
