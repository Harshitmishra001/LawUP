# LawUP: Project Explainer & Technical Deep Dive

Welcome to LawUP. If you are reading this, you are likely picking up the project for the first time, reviewing its architecture, or perhaps you are a future version of the original developer trying to remember why certain decisions were made. 

This document is a comprehensive, from-scratch educational report. It assumes you understand general software engineering and programming, but it assumes **zero prior context** about LawUP, Natural Language Processing (NLP), LoRA fine-tuning, or the legal domain.

By the end of this document, you will understand exactly what LawUP is, how its dataset was built and cleaned, how we fine-tuned a 3-billion parameter model on consumer hardware, and most importantly, why its dual-verification architecture is the only thing standing between a useful AI legal assistant and a dangerous liability.

---

## 1. What is LawUP and Why Does it Exist?

### The Problem
Legal contracts are famously dense, layered with jargon, and structurally hostile to the average reader. Whether it is a freelance service agreement, an employment offer letter, or a privacy policy, contracts are written by lawyers, for lawyers, to be argued over in court. For a non-lawyer, reading a contract is an exercise in missing the hidden traps. 

When Large Language Models (LLMs) like ChatGPT became popular, the obvious solution seemed to be: "Just upload the PDF and ask the AI to summarize it." 

However, generic LLMs fail at legal analysis in two very specific, dangerous ways:
1. **They hallucinate confidently:** They will tell you a clause means something it doesn't, or they will invent a statutory right that does not exist in the text.
2. **They distort meaning when simplifying:** When asked to make a sentence "easier to read," an LLM will frequently soften a strict legal obligation ("The Developer *shall* deliver...") into an optional suggestion ("The Developer *may* deliver..."), or it will completely drop a qualifying condition. For a non-lawyer, these silent distortions are impossible to catch, and they completely change the legal reality of the contract.

### The Core Idea: Dual-Verification
LawUP is an agentic AI system designed to solve these exact problems. It takes a contract and breaks it down clause by clause. For every clause, it attempts to produce a **plain English rewrite** and flag any **inherent risks**. 

But unlike a generic wrapper around ChatGPT, LawUP employs a **dual-verification pipeline**. Before any rewrite or risk flag is shown to the user, it must pass through two automated judges (verifiers):
1. **Meaning-Preservation Verifier:** Checks if the plain English rewrite *exactly preserves* the legal obligations of the original text. If it dropped a condition or inverted a right, the rewrite is rejected and hidden from the user.
2. **Citation-Grounding Verifier:** Checks if the identified risk is actually supported by a real statute or a recognized standard contract clause. If the AI cannot ground its claim in a real citation, the flag is suppressed.

This is what makes LawUP a serious portfolio project rather than a weekend API wrapper. The system is designed to **fail closed**. If it is not confident that a simplification is legally faithful, it simply shows the original dense text rather than lying to the user. The verifiers are not optional polish; they *are* the product.

---

## 2. The Dataset: CUAD

To teach an AI to understand contracts, you need a lot of contracts. LawUP's foundation is built on **CUAD** (the Contract Understanding Atticus Dataset).

### What is CUAD?
CUAD is an open-source dataset created by the Atticus Project. It contains over 500 real commercial contracts sourced from the SEC's EDGAR database (where public companies file their paperwork). Crucially, these contracts have been annotated by dozens of real, human legal experts. The lawyers went through the texts and highlighted specific clauses, tagging them into **41 distinct categories** (like "Non-Compete," "Governing Law," "Audit Rights," "Termination for Convenience," etc.).

CUAD is licensed under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Practically, this means we are completely free to use, modify, and distribute the dataset for commercial or non-commercial purposes, as long as we explicitly give credit (attribution) to the Atticus Project in our documentation and model cards. 

### Why CUAD? (And its Limitations)
CUAD was chosen because it provides high-quality, expert-labeled ground truth for legal language. However, it has limitations. Because it is sourced from corporate SEC filings (mergers, acquisitions, joint ventures), it is heavily biased toward commercial law. 

It does **not** contain everyday consumer contracts like residential leases. If we tried to ask our CUAD-trained model to find the "security deposit terms" in a residential lease, it would fail silently because it has never seen one. LawUP explicitly limits its Phase 1 scope to freelance, service, and employment agreements (which share DNA with CUAD's commercial contracts), leaving residential leases for a future phase requiring a custom dataset.

### From First Principles: Clauses and Categories
In this context, a **"clause"** is just a paragraph or a specific sentence within a contract that deals with one specific topic. 

The lawyers who built CUAD created 41 highly specific categories. For example, they had separate categories for "Post-Termination Services" and "Termination for Convenience." For LawUP, providing a user with 41 granular legal tags is overwhelming. We programmatically mapped and condensed these 41 categories down into roughly **11 broader risk categories**. This simplification makes the final UI much more digestible for a non-lawyer, turning highly technical legal taxonomy into practical risk warnings.

---

## 3. Data Quality: Shortcuts and Bugs Before Training

Machine learning models are only as good as the data they consume. "Garbage in, garbage out" is a cliché because it is universally true. Before we could train our model, we had to make strategic data choices and aggressively audit for bugs. 

### Shortcut 1: Embracing the Pre-Segmented CSV
**What it was:** Extracting clause boundaries from raw PDFs or unstructured text is notoriously difficult in legal tech. Rather than building a complex clause segmenter from scratch for Phase 1, we made a deliberate architectural choice to use a pre-segmented master CSV of the CUAD dataset. 
**The Benefit:** This shortcut allowed us to bypass a massive engineering hurdle (raw document ingestion) and focus entirely on the core problem: fine-tuning the model for simplification and building the verifiers. 
**The Tradeoff:** We accepted that the CSV occasionally stripped surrounding context or truncated clauses to fit cell limits. Production ingestion will eventually require a custom segmenter (as noted in our Phase 1 plans), but for training the initial LoRA, this was a highly beneficial tradeoff that accelerated development.

### Bug 1: Redaction Placeholder Contamination
**What it was:** Because CUAD comes from public SEC filings, companies frequently redact confidential information (like dollar amounts or trade secrets) before filing. These show up in the text as `[***]` or `[OMITTED]`.
**The Harm:** If we trained our model to simplify text containing `[***]`, the model would internalize this as standard legal English. When asked to simplify a user's normal contract in production, the model would randomly insert `[***]` into its output, effectively hallucinating redactions.
**The Fix:** We caught exactly **199 contaminated clauses** during our data audit. We added cleaning logic to the pipeline to identify and strip these placeholders before the model ever saw them.

### Bug 2: Label Hallucination (Mislabeled Headers)
**What it was:** In some cases, the human annotators had highlighted a section header (e.g., "7. NON-COMPETE") but accidentally missed highlighting the actual body text of the clause. 
**The Harm:** The model would be taught that the mere phrase "7. NON-COMPETE" is a complete legal obligation. It would learn to associate short headers with complex risk categories, leading to massive false-positive flags in production.
**The Fix:** We found exactly **7 mislabeled headers** masquerading as substantive clauses. We filtered these out, ensuring the model only trained on actual, meat-on-the-bone legal obligations. After all filtering and cleaning, we locked in a pristine final training set of exactly **3,420 clauses**.

**The Lesson:** Data quality auditing is not a chore; it is the most highly leveraged engineering work you can do. A model will happily learn exactly what you teach it, including your mistakes.

---

## 4. Building the Training Data for Simplification

With clean clauses extracted from CUAD, we needed to teach the model *how* to simplify them. LLMs learn via examples. We needed to create **training pairs**: an original, dense legal clause, and a target plain-English rewrite.

### Generating the Positive Examples
We obviously could not hire a team of lawyers to rewrite thousands of clauses. Instead, we used a highly capable "teacher" model—DeepSeek V4 Flash—via an API. We fed it the original CUAD clauses and gave it a strict prompt: *Rewrite this in plain English, preserve all obligations exactly, do not add or remove meaning, use active voice.* 

This generated thousands of high-quality "positive" training pairs.

### Generating the Negative Examples (Corruptions)
If you only show a model good examples, it learns what to do, but it doesn't learn the boundaries of what *not* to do. This is especially important for our downstream verifiers. We needed contrastive "negative" examples.

We generated these programmatically (for free) using rule-based corruptions:
- **Modal Swap:** Finding the word "shall" and replacing it with "may" (turning an obligation into an option).
- **Condition Appended:** Adding a fake condition to the end of a sentence (e.g., "...provided that the moon is full").
- **Temporal Inversion:** Swapping "before" with "after".

These taught the model exactly what semantic distortions look like.

### The Adversarial Holdout Set
Before doing any of this, we hand-picked **23 exceptionally difficult clauses** and completely sequestered them. These were multi-party clauses with nested conditions and complex cross-references. We called this the "adversarial holdout set." 

We strictly excluded these from the training data. Why? Because if the model saw them during training, it would just memorize the answers. By keeping them hidden, we could use them later to test the model's true ability to generalize to hard, unseen problems.

---

## 5. Fine-Tuning: QLoRA on SmolLM3-3B

With our dataset ready, it was time to train.

### Base Models and Fine-Tuning
Training an LLM from absolute scratch (random weights) requires millions of dollars in supercomputer time and trillions of words. Instead, we start with a **base model**—in our case, `SmolLM3-3B`, a 3-billion parameter model released by HuggingFace. It already knows how to speak English, understand grammar, and follow basic instructions.

**Fine-tuning** is the process of slightly adjusting the model's internal "brain" (its weights) using our specific training pairs, teaching it the specialized skill of legal simplification.

### LoRA (Low-Rank Adaptation)
Even fine-tuning a 3B model normally requires massive GPUs. As a portfolio project, we were constrained to a single, cheap Google Colab T4 GPU (16GB VRAM). 

To make this fit, we used **LoRA** (specifically configured with `rank=16` and `alpha=32`). Instead of updating all 3 billion weights in the model, LoRA freezes the entire base model and injects tiny, trainable "adapter" layers into it. It is like leaving a textbook intact and just adding a few sticky notes to the pages. We only had to train these sticky notes, drastically reducing memory and compute requirements.

### Quantization (QLoRA)
To fit the massive base model into the 16GB GPU in the first place, we used **Quantization** (specifically NF4 4-bit). Quantization compresses the model's weights by reducing their precision. Think of it like taking a high-res photo and saving it as a slightly lower-res JPEG. It looks almost identical, but takes up a fraction of the space. Combining Quantization and LoRA gives us **QLoRA**, allowing us to train a 3B model on a single cheap GPU.

### The Hardware Precision Crash
During Milestone 2, we hit a wall. The training library (PEFT) forcefully initialized our new LoRA adapters in `bfloat16` (a specific type of 16-bit decimal math). However, the T4 GPU is older (Compute Capability 7.5) and physically lacks the hardware to do `bfloat16` math natively. 

When PyTorch's `GradScaler` (a tool used to prevent numbers from rounding down to zero during training) tried to process these bfloat16 numbers, it crashed instantly with a `NotImplementedError`. 

**The Fix:** We had to implement a "monkeypatch" (a surgical code override at runtime) to explicitly disable the GradScaler. This was a calculated risk—disabling it meant we might suffer from "gradient underflow" (numbers becoming too small and vanishing). Fortunately, we monitored the training closely and the gradients remained healthy. The model trained successfully.

### "Checkpoint-300"
During training, the system saves snapshots of the model's brain periodically (checkpoints). We configured the trainer to evaluate the model's performance on a hidden validation set every 100 steps. We found that the model achieved its best performance at step 300 (which we call `checkpoint-300`). Training continued to step 382, but the performance started getting slightly worse (overfitting). We wisely discarded the later steps and kept `checkpoint-300` as our final, optimal adapter.

---

## 6. The Verification Layer: The Core of the Project

We now had a fine-tuned model capable of simplifying legal text. But as established in Section 1, generating fluent text is not enough. We had to verify it.

### What is Entailment?
To verify a rewrite, we rely on a concept from NLP called **Entailment**. 
In plain language: *If Statement A is true, does Statement B have to be true?*
- If yes, A entails B.
- If no, they might be contradictory, or B might be making things up.

### The Single-Direction Trap
Initially, one might think: "I'll just check if the Original Clause entails the Rewrite. If the rewrite is true based on the original, it's a good rewrite!"

This is a dangerous trap. Let's use a simple analogy.
- **Original:** "You must wash the car, mow the lawn, and take out the trash."
- **Rewrite:** "You must wash the car."

If the original is true, is the rewrite true? **Yes.** If you have to do all three, it is logically true that you have to wash the car. A standard entailment model will score this as a "PASS". 
But as a contract simplification, this is a disaster! The rewrite completely dropped two massive obligations (mowing the lawn, taking out the trash). 

A single-direction check from Original -> Rewrite only catches **hallucinations** (if the rewrite added "and paint the house") or **contradictions** (if it said "do not wash the car"). It completely misses **omissions**.

### The Solution: Bidirectional Entailment
To catch omissions, we must run the check in reverse: *If the Rewrite is true, does the Original have to be true?*
- Rewrite: "You must wash the car."
- Original: "You must wash the car, mow the lawn, and take out the trash."
Does the rewrite entail the original? **No.** Washing the car does not guarantee the lawn gets mowed. The entailment fails, and we successfully catch the omission.

Therefore, LawUP runs **Bidirectional Entailment**:
1. **Forward (Original → Rewrite):** Catches Hallucinations and Contradictions.
2. **Backward (Rewrite → Original):** Catches Omissions.

### The Failure Categories
When the verifier catches an error, it categorizes it. Here are real examples from our adversarial testing:

- **HALLUCINATION:** The rewrite invents something not in the text.
  - *Real Example:* In a Non-Compete clause (UID: WPPPLC), the rewrite mentioned a "Restricted Territory." The original clause didn't define this territory in that specific paragraph.
- **CONTRADICTION:** The rewrite actively inverts a rule.
  - *Real Example:* In a Cooperation Agreement (UID: URSCORPNEW), the original said Jana could make public statements *only if* the company made an announcement prior to the March Board Meeting. The rewrite said Jana could make statements *about* the announcement, inverting the condition.
- **OMISSION:** The rewrite silently drops a rule.
  - *Real Example:* In an Audit Rights clause (UID: DovaPharmaceuticals), the original limited "for cause" audits strictly to "significant compliance problems." The rewrite dropped this limitation, implying audits could happen for any reason.
- **STRUCTURAL_COLLAPSE:** A special type of omission where two distinct rules applied to two different parties are recklessly merged into one vague rule by the AI, losing the boundaries of who can do what. 
  *(Note: Initially, our prompt didn't explain this well enough, causing the AI to lump these in with standard Omissions. We had to explicitly tighten the prompt to separate them.)*

---

## 7. The Debugging Story: Engineering Reality

Building LawUP was not a straight line. We made several critical mistakes. Documenting these is vital because they teach engineering resilience.

### Mistake 1: Premature Completion (M3)
At the start of Milestone 3, we successfully got the fine-tuned model generating rewrites locally using a tool called LM Studio. We declared the milestone complete. The project owner had to push back hard: generating text is just a demo. M3 was scoped to build the *verifier*. We had skipped the core product feature just because the generator was working. 
**Lesson:** Never mistake a running script for a completed feature. Re-read the spec.

### Mistake 2: The `.startswith()` Corruption
When extracting the 23 adversarial clauses for our holdout set, we used a python string method `uid.startswith(target_uid)` to find the matches. 
Because some UIDs shared prefixes, this loose matching grabbed 582 clauses instead of 23. This is terrifying because it failed silently—the code didn't crash; it just fed the wrong, massive dataset into the evaluation pipeline. 
**Lesson:** Exact-match bugs fail silently. Always use strict equality (`==`) for unique identifiers, and always assert the expected length of your output (`assert len(results) == 23`).

### Mistake 3: Swapping Forward and Backward Entailment
In an early design document, we wrote down the logic for bidirectional entailment but swapped the directions. We wrote that (Rewrite → Original) caught hallucinations. As proved in Section 6, this is logically backward.
**Lesson:** Semantic checks are brain-bending. Always write out a plain-language analogy (like the car-washing example) to ground your logic before writing code.

### Mistake 4: Runaway Generation (The Stop Token Bug)
During our final evaluation run, the model generated the plain English rewrites, but then just kept talking. It outputted conversational garbage like *"Now it's your turn! Please rewrite the following..."* and started generating entirely new, unrelated contracts. 
**The Root Cause:** When we trained the model, we appended a specific end-of-string marker (`<|endoftext|>`). But when we ran the evaluation script, we accidentally told it to stop generating when it saw a completely different marker (`<|im_end|>`, used by ChatML). Because the model never output the ChatML token, it just kept vomiting text until it hit its maximum length limit.
**Lesson:** Inference configurations must perfectly mirror training configurations. A single mismatched token can contaminate your entire downstream evaluation.

### Mistake 5: JSON Parsing Failures
Our verifier asks the LLM to output its final judgment as a strict JSON object. Sometimes, the LLM would output the perfect JSON, but append helpful (and fatal) text like *"Here is your JSON:"* before it, or *"Hope this helps!"* after it. Our naive python code tried to parse the whole string and crashed.
**The Fix:** We replaced naive string stripping with a robust Regular Expression (`re.search(r'(\{.*\})', raw_text, re.DOTALL)`) designed to mathematically extract only the first valid JSON block and ignore everything else.

---

## 8. Current Results and What They Mean

With all bugs squashed and the pipeline uncontaminated, we ran the 23 adversarial clauses through our fine-tuned generator and zero-shot 3B meaning-preservation verifier. 

**The Final Baseline Metrics:**
- **Overall Pass Rate:** 52.2% (12/23)
- **HALLUCINATION:** 4
- **CONTRADICTION:** 2
- **OMISSION:** 7
- **STRUCTURAL_COLLAPSE:** 0

### Interpreting the Numbers
At first glance, a 52% pass rate sounds terrible. But context is everything:
1. **Small Sample Size:** `n=23` is statistically tiny.
2. **Adversarial Difficulty:** These aren't average clauses. They are the 23 absolute hardest, most convoluted multi-party nightmares we could find in the dataset. They were hand-picked to break the model.

However, the most important finding is not the pass rate. The most important finding is a phenomenon we call **Partial Detection (Under-flagging)**. 
When manually reviewing the verifier's logs, we found that even when the verifier marked a clause as a `FAIL`, it often missed the biggest error. For example, in the `URSCORPNEW` clause, the generator silently dropped a critical "3-day remedy" condition. The verifier missed this entirely, flagging the clause as a `FAIL` only because of a minor, unrelated contradiction.

**The Conclusion:** A "FAIL" from a 3-billion parameter zero-shot LLM is a floor, not a ceiling. It proves that the model is too weak for production verification. This successfully validates the core thesis of LawUP: we cannot rely on zero-shot generalist models for safety-critical legal checks. We must proceed with our plan to distill the verifier into a highly specialized, fine-tuned DeBERTa NLI model.

---

## 9. What's Next?

LawUP is moving forward systematically. The next immediate steps are:

1. **Milestone 4 (Citation-Grounding):** Building the second half of the dual-verification pipeline. We will build the retrieval system that maps risk flags back to concrete source statutes and standard clauses.
2. **Verifier Distillation:** Taking the baselines from M3 and M4 and training robust, fast DeBERTa models to replace the weak zero-shot LLM verifiers.
3. **Backend & Frontend:** Wrapping the DAG pipeline into a FastAPI backend and building the React user interface.

*(Note: Preparing the model for a public release on Hugging Face is a separate, parallel packaging task that does not block the core engineering milestones.)*

In **Phase 2**, once the core pipeline is locked, we will explore advanced features:
- **Scenario Coverage:** Did the contract forget to mention what happens if the company goes bankrupt?
- **Redline Mode:** Suggesting specific textual edits to fix risky clauses.
- **Anomaly Scoring:** Using clause-embeddings to flag if a clause is mathematically "weird" compared to industry standards.

LawUP is not about chatting with PDFs. It is about building a deterministic, verifiable, and fail-safe pipeline for legal understanding.
