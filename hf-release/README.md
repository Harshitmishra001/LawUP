---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM3-3B
tags:
- legal
- contract-analysis
- text-simplification
- lora
- qlora
---
# LawUP

## Model Description
LawUP is a fine-tuned version of SmolLM3-3B using QLoRA. It is designed to rewrite complex legal contract clauses into plain English, making them easier to understand. 

**Note: This repository contains the LoRA adapter weights only (~120MB), not a standalone model.** You must load this adapter on top of the base SmolLM3-3B model.

**Important Note:** This model is a fine-tuned simplification model ONLY. It does not have an automated meaning-preservation verifier built-in. It will not fact-check its own rewrites, and may occasionally drop or alter important legal meaning from the original text.

## Disclaimer
> [!CAUTION]
> **This is NOT legal advice.** This model's outputs have NOT been reviewed by an attorney and must NOT be relied on for real contractual decisions. Please read the full [Disclaimer](DISCLAIMER.md).

## Training Data Provenance
The training data for LawUP was derived from the [CUAD (Contract Understanding Atticus Dataset)](https://huggingface.co/datasets/theatticusproject/cuad).
© The Atticus Project, Inc., licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 

**Citation:**
Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. *"CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review."* NeurIPS 2021.

## Intended Use
LawUP is a portfolio and research project demonstrating the capabilities of agentic contract analysis using small language models. 

## Limitations
- **Verification Under-Flagging (Zero-Shot Limit):** The model's outputs were tested against a strict bidirectional entailment verifier. On a 23-clause adversarial holdout set, it achieved a **52.2% pass rate (12/23)**. 
- **Failure Breakdown:** The failures (non-exclusive) included Omission (7), Hallucination (4), Contradiction (2), and Structural Collapse (0). 
- **Note on Verification:** A manual review of the logs showed that the zero-shot 3B meaning-preservation verifier used for this baseline occasionally under-flags errors (e.g., missing a massive omission while correctly flagging a minor contradiction in the same clause). Therefore, a `FAIL` verdict acts as a floor, not a ceiling. This confirms that zero-shot LLMs are too weak for robust safety-critical legal checks, and LawUP will distill the verifier into a specialized DeBERTa NLI model in future milestones.
- **Model Size:** The model has 3B parameters and may miss subtle legal distinctions, conditions, or exceptions in long, multi-clause texts.

## How to Use

### Ollama Modelfile Instructions
To use this adapter with Ollama, you must create a `Modelfile` that pulls the base model and applies this LoRA adapter. Make sure to download `LawUP-lora-f16.gguf` to the same directory as the Modelfile.

```dockerfile
FROM hf.co/HuggingFaceTB/SmolLM3-3B-GGUF:Q8_0
ADAPTER ./LawUP-lora-f16.gguf

TEMPLATE """You are a legal-plain-language expert.

Rewrite the following contract clause in clear, everyday English that a non-lawyer can understand.

RULES:
1. Preserve ALL obligations, conditions, deadlines, and parties exactly.
2. Do NOT add, remove, or invert any condition or obligation.
3. Use short sentences and active voice.
4. Keep defined terms (e.g., "Licensee", "Effective Date") unchanged.
5. Output ONLY the rewritten clause — no commentary.

Original Clause:
{{ .Prompt }}

Plain English Rewrite:
"""
PARAMETER stop "<|endoftext|>"
```

### llama.cpp Loading Instructions
You can load the base model and apply the adapter using `llama-cli`:
```bash
llama-cli -m SmolLM3-3B-Q8_0.gguf --lora LawUP-lora-f16.gguf -p "You are an expert legal assistant. Your task is to rewrite the following complex legal clause into plain, easy-to-understand English. Ensure that you preserve all original obligations, conditions, and permissions without adding or removing any meaning.\n\nOriginal Clause:\n[Your legal clause here]\n\nPlain English Rewrite:\n"
```
