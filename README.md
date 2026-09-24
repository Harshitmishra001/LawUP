# LawUP: Clause-by-Clause Legal Risk Analysis & Verification Pipeline

[![CI](https://github.com/Harshitmishra001/LawUP/actions/workflows/ci.yml/badge.svg)](https://github.com/Harshitmishra001/LawUP/actions/workflows/ci.yml)
[![Status: Milestone 5 | Frontend Development](https://img.shields.io/badge/Status-Milestone%205%20%7C%20Frontend%20Active-blue.svg)](#roadmap--milestone-progress)
[![Base Model: SmolLM3-3B](https://img.shields.io/badge/Base%20Model-SmolLM3--3B-orange.svg)](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

---

## 📌 What is LawUP?

**LawUP** is an agentic, deterministic AI system that analyzes commercial contracts and employment agreements clause-by-clause. Rather than acting as an unconstrained conversational chatbot, LawUP operates as a structured, fail-closed multi-agent pipeline designed to prevent hallucinations and provide legally grounded analysis.

For every clause in a contract, LawUP:
1. **Simplifies** dense legalese into clear, understandable English without altering legal obligations.
2. **Classifies** multi-label legal risks across a vetted legal taxonomy.
3. **Retrieves Grounding References** (statutes and industry standards) using semantic vector search.
4. **Verifies Meaning Preservation** using Natural Language Inference (NLI) entailment checks to ensure the rewrite hasn't inverted or dropped critical obligations.

---

## 🏛️ Pipeline Architecture

LawUP replaces single-prompt LLM generation with a **fail-closed orchestrator pipeline**:

```
[Raw Contract Clause]
         │
         ▼
 ┌──────────────────────┐
 │ Simplification Agent │ ──► LoRA-adapted SmolLM3-3B (Greedy Decoding)
 └──────────────────────┘
         │
         ▼
 ┌──────────────────────┐
 │ Classification Agent │ ──► LoRA-adapted SmolLM3-3B (11 Risk Categories)
 └──────────────────────┘
         │
         ▼
 ┌──────────────────────┐
 │   Grounding Engine   │ ──► Sentence-Transformers Semantic Search
 └──────────────────────┘
         │
         ▼
 ┌──────────────────────┐
 │ NLI Verifier Agent   │ ──► Dual-Direction Entailment Verification
 └──────────────────────┘
         │
         ▼
[Structured Verified Output / Report]
```

### Multi-Agent Breakdown

| Stage | Component | Technical Implementation | Status |
|:---:|:---|:---|:---:|
| **1** | **Ingestion** | Document parsing and unit segmentation | ✅ Complete |
| **2** | **Simplifier Agent** | Fine-tuned `SmolLM3-3B` QLoRA model generating plain-English rewrites (stops on `<\|endoftext\|>`) | ✅ Complete |
| **3** | **Classifier Agent** | Fine-tuned `SmolLM3-3B` QLoRA model predicting structured JSON multi-label risks | ✅ Complete |
| **4** | **Grounding Retriever** | Semantic vector retrieval over curated statute and industry standard database | ✅ Complete |
| **5a** | **Meaning Verifier** | Bi-directional NLI entailment gate checking for omissions or obligations inversions | ✅ Complete |
| **5b** | **Citation Verifier** | Verification of retrieved statutory references against claim contents | ⏳ Scheduled |
| **6** | **Frontend / Report** | Interactive Next.js UI with real-time risk badges, side-by-side comparisons & verifier badges | ⏳ In Progress (M5) |

---

## 🤖 Released Fine-Tuned Models

Both core models are fine-tuned on the **Contract Understanding Atticus Dataset (CUAD)** using QLoRA, merged, and exported to **Q8_0 GGUF** for high-speed, 100% offline local inference on consumer GPUs/CPUs:

| Model | Task | Hugging Face Repository | Format | Benchmark |
|:---|:---|:---|:---:|:---:|
| **`lawup-simplifier-smollm3-3b`** | Legal Clause Simplification | [HeavenlyDem0n/lawup-simplifier-smollm3-3b](https://huggingface.co/HeavenlyDem0n/lawup-simplifier-smollm3-3b) | LoRA / GGUF Q8_0 | Adversarial Holdout Tested |
| **`lawup-classifier-smollm3-3b`** | Multi-Label Risk Classification | [HeavenlyDem0n/lawup-classifier-smollm3-3b](https://huggingface.co/HeavenlyDem0n/lawup-classifier-smollm3-3b) | LoRA / GGUF Q8_0 | **Micro F1: 0.7294**<br>Recall: **0.7750** |

### Offline Privacy & Inference
Contracts often contain sensitive and confidential information. By quantizing models to GGUF, LawUP can run entirely locally through [LM Studio](https://lmstudio.ai/) or `llama.cpp` without transmitting contract clauses to external third-party APIs.

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Harshitmishra001/LawUP.git
cd LawUP

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
pip install -r data/scripts/requirements.txt
pip install sentence-transformers pytest ruff
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration connects to LM Studio at `http://localhost:1234/v1`:
```env
LM_STUDIO_URL=http://localhost:1234/v1
```

### 3. Load the Model in LM Studio
1. Open **LM Studio**.
2. Search and download `HeavenlyDem0n/lawup-classifier-smollm3-3b` (or `lawup-simplifier-smollm3-3b`).
3. Start the **Local Inference Server** on port `1234`.

### 4. Run the FastAPI Backend Server

```bash
uvicorn backend.app.api.main:app --reload --host 127.0.0.1 --port 8000
```
Open your browser to interactive API documentation at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📡 API Reference

### `POST /analyze` (End-to-End Orchestrator)
Chains Simplification, Risk Classification, Grounding Retrieval, and NLI Verification.

**Sample Request:**
```json
POST /analyze
Content-Type: application/json

{
  "clause": "The Receiving Party shall indemnify and hold harmless the Disclosing Party against any consequential damages arising from a breach."
}
```

**Sample Response:**
```json
{
  "original_clause": "The Receiving Party shall indemnify and hold harmless the Disclosing Party against any consequential damages arising from a breach.",
  "simplified_rewrite": "You promise to protect us from any problems or damages resulting from your breach.",
  "risks_identified": [
    {
      "risk": "Liability Exposure",
      "sub_category": "Uncapped Liability"
    }
  ],
  "grounding_references": [
    {
      "risk_category": "Liability Exposure",
      "sub_category": "Liquidated Damages",
      "reference_text": "Industry Standard: Liquidated damages clauses must represent a reasonable pre-estimate of actual anticipated losses...",
      "authority_type": "industry_norm",
      "scope": "general",
      "similarity_score": 0.4361
    }
  ],
  "verifier_report": {
    "verdict": "PASS",
    "original_entails_rewrite": true,
    "rewrite_entails_original": true
  },
  "total_latency_ms": 1420.5
}
```

### `POST /simplify`
Dedicated low-latency endpoint for clause simplification with 1024 token prompt bounds checking.

---

## 🧪 Testing & Quality Gates

Run the automated test suites:

```bash
# Data quality and schema integrity tests
pytest data/scripts/tests/test_data_quality.py -v

# Validate grounding corpus schema
python scripts/validate_corpus_schema.py

# Backend semantic retrieval test suite
PYTHONPATH=. pytest backend/tests/test_retrieval.py -v
```

---

## 📂 Repository Structure

```
LawUP/
├── .github/workflows/       # CI/CD GitHub Actions pipelines
├── backend/
│   ├── app/
│   │   ├── agents/          # Multi-agent implementations (simplifier, classifier, verifier)
│   │   ├── api/             # FastAPI endpoints & orchestrator (/analyze, /simplify)
│   │   └── retrieval.py     # Sentence-Transformers semantic grounding engine
│   ├── tests/               # Backend & retrieval unit tests
│   └── requirements.txt     # Backend service dependencies
├── data/
│   ├── processed/           # Processed holdout sets & test artifacts
│   └── scripts/             # Data prep, mapping & curation pipelines
├── grounding_corpus/        # Curated statutory & industry norm references
├── hf-release/              # Model cards & Hugging Face release assets
├── training/
│   ├── configs/             # QLoRA training YAML configurations
│   ├── eval/                # Evaluation & adversarial verification scripts
│   └── train_classifier.py  # Self-healing QLoRA fine-tuning script
├── PLAN.md                  # Comprehensive architectural master plan
└── README.md                # Project documentation
```

---

## 🗺️ Roadmap & Milestone Progress

- [x] **Milestone 1 — Taxonomy & Data Pipeline:** CUAD preprocessing, label taxonomy alignment, adversarial holdout extraction.
- [x] **Milestone 2 — Simplification Model:** SmolLM3-3B QLoRA fine-tuning, GradScaler stabilization, checkpoint exports.
- [x] **Milestone 3 — Inference & Backend Foundation:** FastAPI service setup, LM Studio client integration, token validation.
- [x] **Milestone 4 — Classification & Verification:** QLoRA multi-label classifier (F1: 0.729), semantic grounding retrieval engine, dual-direction NLI verifier, and `/analyze` orchestrator pipeline.
- [ ] **Milestone 5 — Frontend Development:** Next.js / Tailwind CSS web application featuring side-by-side diff views, risk cards, and verification badges.
- [ ] **Milestone 6 — Deployment & Production Plumbing:** Containerized deployment on Hugging Face Spaces + Vercel.

---

## ⚖️ Attributions & Disclaimer

### Data Attribution
Training data derived from the **Contract Understanding Atticus Dataset (CUAD) v1** by The Atticus Project, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Model Attribution
Base model: **SmolLM3-3B** by Hugging Face, licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### Disclaimer
> **LawUP provides automated document analysis for informational purposes only and does not provide legal advice.** It is not a substitute for professional legal counsel. Critical legal decisions should always be reviewed by a licensed attorney.
