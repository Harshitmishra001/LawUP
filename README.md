# LawUP

![Status: Phase 1 — In Development](https://img.shields.io/badge/status-Phase%201%20·%20In%20Development-yellow)

---

## What is LawUP?

LawUP is an agentic AI system that analyzes contracts clause-by-clause, producing plain-language rewrites, citation-grounded risk flags, and verification badges. It is **not a chatbot** — it is a structured pipeline where every claim is checked. Each output is either verified against a source or explicitly marked as unverified, following a fail-closed design.

---

## Architecture Overview

LawUP processes contracts through a deterministic, multi-agent pipeline:

```
Ingestion → Simplification → Classification → Grounding → Verification → Report
```

| Stage | Agent | Purpose |
|-------|-------|---------|
| 1 | **Ingestion** | Parse uploaded PDF/DOCX, extract clause-level text units |
| 2 | **Simplification** | Generate plain-language rewrites via LoRA-adapted SmolLM3 |
| 3 | **Classification** | Multi-label clause risk classification via LoRA-adapted SmolLM3 |
| 4 | **Grounding** | Retrieve supporting statute/standard-clause text for flagged clauses |
| 5a | **Meaning Verification** | NLI-style entailment check — rewrite preserves original obligations |
| 5b | **Citation Verification** | Check that retrieved source actually supports the risk claim |
| 6 | **Report** | Assemble structured output with verification badges |

Both verification gates **fail closed**: if a check cannot confirm correctness, the output is flagged, not silently passed.

---

## Phase 1 Scope

Phase 1 targets two common, high-impact contract types:

- **Freelance / service contracts**
- **Employment offer letters**

### Phase 1 Coverage Gaps

> The Phase 1 classifier does not cover **Confidentiality Scope** or **Indemnification** — CUAD has no annotated categories for either. These are the first candidates for augmentation in Phase 2.

---

## Released Models

The fine-tuned LoRA adapter for the Simplification Agent is publicly available on Hugging Face:
- [lawup-simplifier-smollm3-3b](https://huggingface.co/HeavenlyDem0n/lawup-simplifier-smollm3-3b) (Q8 GGUF / LoRA Adapter)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Base model | SmolLM3-3B (QLoRA fine-tuned) |
| Backend | FastAPI (Python) |
| Frontend | Next.js |
| Training | QLoRA via PEFT + bitsandbytes, targeting Colab T4 |
| Grounding | Sentence-Transformers for retrieval over statute corpus |

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
