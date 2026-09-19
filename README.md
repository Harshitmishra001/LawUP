# LawUP

![Status: Milestone 5 (Frontend) In Development](https://img.shields.io/badge/status-Milestone%205%20%7C%20Frontend%20Development-blue)

---

## What is LawUP?

LawUP is an agentic AI system that analyzes contracts clause-by-clause, producing plain-language rewrites, citation-grounded risk flags, and verification badges. It is **not a chatbot** — it is a structured pipeline where every claim is checked. Each output is either verified against a source or explicitly marked as unverified, following a fail-closed design.

---

## Architecture Overview

LawUP processes contracts through a deterministic, multi-agent pipeline:

Ingestion ➝ Simplification ➝ Classification ➝ Grounding ➝ Verification ➝ Report

| Stage | Agent | Purpose | Status |
|-------|-------|---------|--------|
| 1 | **Ingestion** | Parse uploaded PDF/DOCX, extract clause-level text units | ✅ Complete |
| 2 | **Simplification** | Generate plain-language rewrites via LoRA-adapted SmolLM3 | ✅ Complete |
| 3 | **Classification** | Multi-label clause risk classification via LoRA-adapted SmolLM3 | ✅ Complete |
| 4 | **Grounding** | Retrieve supporting statute/standard-clause text for flagged clauses | ✅ Complete |
| 5a | **Meaning Verification** | NLI-style entailment check — rewrite preserves original obligations | ✅ Complete |
| 5b | **Citation Verification** | Check that retrieved source actually supports the risk claim | ⏳ Pending |
| 6 | **Report / Frontend** | Assemble structured output with verification badges in React UI | ⏳ In Progress |

Both verification gates **fail closed**: if a check cannot confirm correctness, the output is flagged, not silently passed.

---

## Released Models

Our fine-tuned LoRA models and GGUF exports are publicly available on Hugging Face:
- **[lawup-simplifier-smollm3-3b](https://huggingface.co/HeavenlyDem0n/lawup-simplifier-smollm3-3b)**: Simplifies dense legalese into plain English.
- **[lawup-classifier-smollm3-3b](https://huggingface.co/HeavenlyDem0n/lawup-classifier-smollm3-3b)**: Multi-label classifier that detects contract risks (e.g., Indemnification, Termination).

These models are exported as **Q8_0 GGUF** files and are designed to be run locally and entirely offline via llama.cpp or LM Studio.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base model | SmolLM3-3B (QLoRA fine-tuned) |
| Backend | FastAPI (Python) with Orchestrator pattern |
| Inference | LM Studio (llama.cpp) for secure offline API hosting |
| Frontend | Next.js (React) |
| Training | QLoRA via PEFT + bitsandbytes, targeting Colab T4 |
| Grounding | Sentence-Transformers for semantic retrieval over JSON statute corpus |

---

## Current Status

**The AI and Backend architecture is 100% complete!** 
The /analyze endpoint in FastAPI successfully chains together the Simplification model, the Classification model, the Sentence-Transformers Semantic Retriever, and the NLI Verifier into a single cohesive response. 

**Next Steps:** Building the Next.js React frontend to visualize these risk badges and verified citations.

---

## Data Attribution

Training data derived from the **Contract Understanding Atticus Dataset (CUAD) v1** by The Atticus Project, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Model Attribution

Base model: **SmolLM3-3B** by Hugging Face, licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Disclaimer

> **LawUP provides informational analysis only, not legal advice.** Consult a qualified professional for anything high-stakes.

---

## License

TBD
