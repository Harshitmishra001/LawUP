# LawUP — Project Plan

**Status:** Pre-build / design-locked for Phase 1
**Owner:** Harshit
**Purpose of this document:** This is the single source of truth for what LawUP is, why it's built the way it is, and in what order to build it. Read this file in full before starting any coding session. If a task isn't covered here, or a design decision here seems ambiguous, ask before improvising — do not silently invent behavior, especially around the safety-critical items in Section 7.

---

## 1. What LawUP Is

LawUP is an agentic AI system that takes a contract (starting with freelance/service agreements and employment offer letters) and produces, clause by clause:
1. A **plain-language rewrite** of what the clause actually means.
2. A **risk flag** if the clause falls into a known risky category, with a citation to the source that justifies the flag.
3. A **verified badge** — both the rewrite and the flag have passed an internal check before being shown to the user.

It is not a chatbot that answers questions about an uploaded PDF. It is a structured pipeline where every claim it makes is checked against something before being displayed.

## 2. Why This Exists (and why it isn't a wrapper)

Generic LLM tools that "chat with your contract" fail in two specific, well-documented ways:
- They **hallucinate legal claims confidently** — asserting a clause means something it doesn't, or citing a law that doesn't say what's claimed.
- They **quietly distort meaning when simplifying** — softening "shall" into "may," dropping a condition, inverting an obligation — with no way for a non-lawyer reader to catch it.

LawUP's whole design exists to close those two gaps, not to re-implement contract summarization:
- **Citation-grounding verifier**: no risk flag is shown unless a retrieval step found a real source clause/statute that supports it, and a verification step confirmed the source actually says what's being claimed.
- **Meaning-preservation verifier**: no plain-language rewrite is shown unless an entailment check confirms it didn't drop or invert an obligation from the original.

If you strip out these two verifiers, this project is just a demo. They are not optional polish — they are the product.

## 3. Non-Goals (explicit scope boundaries)

LawUP does **not**:
- Predict case outcomes or litigation odds.
- Give jurisdiction-specific procedural/litigation strategy advice.
- Claim to replace a lawyer for anything high-stakes.
- Present itself as a source of legal advice. Every screen that shows a risk flag carries "this is informational, not legal advice — consult a professional for anything high-stakes" as a persistent UI element, not a footnote.

These aren't just liability hedges — they're product-design decisions that keep the tool honest about what it can and can't verify.

## 4. Document Scope by Phase

| Phase | Document types | Data source | Status |
|---|---|---|---|
| **Phase 1 (build first)** | Freelance/service contracts, employment offer letters | CUAD (clause taxonomy transfers directly — see Section 6) | This plan |
| Phase 2 | + Loan/EMI/BNPL agreements, ToS/privacy policies, NDAs | CUAD extended + synthetic augmentation | Later |
| Phase 3 | + Residential rental agreements (India-specific) | **Requires a new, separately curated dataset** — CUAD does not cover residential leases (see Section 6.1) | Later — do not attempt with CUAD alone |

**Why Phase 1 is freelance/employment and not rental agreements:** CUAD's 510 contracts are sourced from SEC EDGAR filings tied to corporate transactions — M&A, corporate finance, investments, IPOs, licensing, supply/distribution agreements. Concepts like non-compete, IP assignment, indemnification, liability caps, and termination transfer cleanly to freelance and employment contracts. Residential lease concerns (security deposit terms, maintenance obligations, notice periods under tenancy law, lock-in periods) are essentially absent from CUAD. Do not try to force rental-agreement coverage out of a CUAD-only classifier — it will silently underperform on a domain it was never trained for. Rental agreements are a real, valuable future phase, but they need their own small hand-curated Indian-tenancy dataset first.

## 5. System Architecture

```
Upload (PDF/DOCX)
      │
      ▼
[1] Ingestion Agent — parse document, extract clause-level text units
      │  (Phase 1 uses pre-segmented clause data for training;
      │   production ingestion must segment raw uploaded contracts —
      │   this is a nontrivial step, see Section 8.4)
      ▼
[2] Simplification Agent (LoRA) ──┐
      │                            │
[3] Classification Agent (LoRA) ───┤── run in parallel per clause
      │                            │
      ▼                            ▼
[4] Grounding Agent — for any clause flagged in [3], retrieve the
    real supporting statute/standard-clause text from the grounding
    corpus (Section 9), using the document's stated jurisdiction/
    document type as a filter
      │
      ▼
[5a] Verifier — Meaning Preservation (Core Phase 1)
     Checks the Step [2] rewrite against the original clause via
     an NLI-style entailment check. Fails closed: if the rewrite
     doesn't clearly entail the original's obligations, do not show
     it — show the original clause with a "simplification unavailable"
     note instead of a possibly-wrong rewrite.
      │
[5b] Verifier — Citation Grounding (Core Phase 1)
     Checks that the retrieved source from Step [4] actually supports
     the risk claim from Step [3]. Fails closed: if unsupported, drop
     the flag rather than show an ungrounded claim.
      │
      ▼
[6] Confidence Gate
     If classifier confidence on a clause is below threshold, mark it
     "needs human review" instead of guessing either way. This is the
     single most important trust mechanism in the whole system —
     silent overconfidence is the main way legal-info tools become
     dangerous. Do not remove or bypass this gate for demo purposes.
      │
      ▼
[7] Report Assembly — clause-by-clause: original | plain-language
    rewrite (or "unavailable") | risk flag + citation (or none, or
    "needs review") | verified badge
```

**Orchestration recommendation:** implement this as an explicit, deterministic pipeline (plain Python control flow / a simple state machine), not an open-ended agent framework with autonomous tool selection. This is a well-defined DAG with fixed steps and fail-closed gates — determinism and testability matter more here than flexibility. If you want a visualization/state-management layer, a lightweight graph library is fine, but the actual step logic should not be "let the LLM decide what to do next." This is a design recommendation, not a hard requirement — flag it if you think a different approach is warranted before switching.

## 6. Data Pipeline (Phase 1)

### 6.1 Source dataset: CUAD
- **Contract Understanding Atticus Dataset (CUAD) v1** — 510 commercial contracts, 13,000+ expert annotations across 41 clause categories, sourced from SEC EDGAR. License: CC BY 4.0 (cite The Atticus Project in the README/about page).
- Available via Hugging Face `datasets` (`theatticusproject/cuad-qa` or `theatticusproject/cuad`), the master clauses CSV, or the original SQuAD-2.0-style JSON from the GitHub repo (`TheAtticusProject/cuad`) / Zenodo archive. Confirm current dataset identifiers at implementation time — do not assume the exact HF dataset config without checking the dataset card first.
- **First implementation step, non-negotiable:** pull `category_descriptions.csv` directly from the CUAD GitHub repo and confirm the taxonomy mapping in Section 6.2 against the real category names before writing any label-mapping code. Do not hand-code the mapping from memory/assumption.
- **Known class imbalance:** independent benchmarking of CUAD's test split found roughly 30% positive / 70% negative examples per category. Handle this deliberately (class weighting or targeted oversampling) — do not train a naive classifier and be surprised by collapsed recall on the positive class.

### 6.2 Condensed risk taxonomy (confirmed against `category_descriptions.csv` — July 2026)

**Structural rule:** the classifier outputs both the rolled-up LawUP risk category *and* the fine-grained CUAD sub-category label. The rollup is a coarse UI tag; the specific CUAD label survives downstream to Steps 4/5 (grounding and verification) so explanations and citations are precise, not generic. This matters most for large rollup buckets like IP Ownership Risk (9 sub-categories with very different implications).

| LawUP risk category | CUAD categories (exact names from `category_descriptions.csv`) |
|---|---|
| Termination Risk | `Termination For Convenience`, `Notice Period To Terminate Renewal`, `Post-Termination Services` |
| Auto-Renewal Trap | `Renewal Term` |
| Non-Compete / Exclusivity Overreach | `Non-Compete`, `Exclusivity`, `Competitive Restriction Exception`, `No-Solicit Of Customers`, `No-Solicit Of Employees` |
| IP Ownership Risk | `IP Ownership Assignment`, `License Grant`, `Affiliate License-Licensee`, `Affiliate License-Licensor`, `Joint IP Ownership`, `Irrevocable Or Perpetual License`, `Non-Transferable License`, `Unlimited/All-You-Can-Eat-License`, `Source Code Escrow` |
| Liability Exposure | `Cap On Liability`, `Uncapped Liability`, `Liquidated Damages`, `Warranty Duration` |
| Assignment/Control Risk | `Anti-Assignment`, `Change Of Control`, `Third Party Beneficiary` |
| Confidentiality Scope | *(Phase 1 placeholder — no CUAD categories exist for this; see coverage-gap note below)* |
| Dispute / Governing Law | `Governing Law`, `Covenant Not To Sue` |
| Audit/Insurance Burden | `Audit Rights`, `Insurance` |
| Reputational Restriction | `Non-Disparagement` |
| Preferential/Exclusivity Terms | `Most Favored Nation`, `Rofr/Rofo/Rofn` |
| Commercial Commitment Constraints | `Price Restrictions`, `Revenue/Profit Sharing`, `Volume Restriction`, `Minimum Commitment` |
| *(Metadata — separate extraction task, not a risk flag)* | `Document Name`, `Parties`, `Agreement Date`, `Effective Date`, `Expiration Date` |

### Phase 2: Scope Expansion & Application (Post-M7)
*Only once Phase 1 pipeline is fully verified end-to-end.*

1. **Taxonomy Expansion**: Incorporate `Confidentiality Scope` and `Indemnification` using semi-supervised teacher-assisted labeling (not bare keyword heuristics).
2. **Context Window Extension**: Explore YaRN for 64K+ full-document context if clause-level processing proves insufficient for certain risks.
3. **Scenario Coverage Checker (Grey Area Analysis)**:
   - **What it is**: A feature to check if the contract's text explicitly addresses, excludes, or remains silent on specific realistic stress-test scenarios (e.g., "what if termination happens mid-deliverable").
   - **Architecture**: Reuses the Phase 1 meaning-preservation verifier (NLI-style entailment). Given a clause and a hypothetical scenario, the verifier classifies the relationship as **Covered (Entailment)**, **Excluded (Contradiction)**, or **Grey Area / Silent (Neutral)**.
   - **Safety Boundary**: This is strictly *coverage analysis* based on text literalism, never *outcome prediction* (which is explicitly a non-goal). It will never predict how a dispute would resolve, only whether the text currently addresses it.
   - **Implementation**: Hand-write 3-5 curated, deterministic scenarios per risk category (e.g., Termination Risk, IP Ownership) for the v1 library to prevent hallucination risks before exploring dynamic LLM generation.

**Phase 1 coverage gaps (document explicitly in README and model card):**
- **Confidentiality Scope**: CUAD contains no annotated confidentiality/NDA category. The risk category is kept as a structural placeholder; the Phase 1 classifier does not cover it.
- **Indemnification**: Listed conceptually under Liability Exposure in earlier drafts, but CUAD has no Indemnification annotation category. Liability Exposure is trained only on `Cap On Liability`, `Uncapped Liability`, `Liquidated Damages`, and `Warranty Duration`.
- Both are first candidates for augmentation post-M2 via teacher-model-assisted labeling on raw CUAD contract text — not via keyword heuristics, which would over-trigger.

**Singleton-bucket monitoring (check after `build_splits.py` runs):** Auto-Renewal Trap (`Renewal Term`) and Reputational Restriction (`Non-Disparagement`) each roll up a single CUAD category. Pull their positive-example counts after splitting — if severely imbalanced, these may need per-category class weighting or wider confidence-gate thresholds in M2, rather than a single global imbalance correction.

### 6.3 Splits and eval sets
- **Split by contract, not by clause.** Splitting at the clause level lets near-duplicate clauses from the same contract leak across train/val/test and inflates the reported F1 — this is the same category of mistake as a bad CV split with target leakage. Every split decision in this project should be made at the document level.
- Report **per-category F1 and macro-F1** on the held-out CUAD test split.
- Maintain a **small hand-curated adversarial eval set** (a handful of real "sounds fine, isn't" clauses) as a standing check, separate from the CUAD test split — CUAD alone won't catch well-lawyered clauses written to sound benign.

### 6.4 Simplification data (no public parallel corpus exists for this)
- Generate synthetic clause → plain-language pairs via a stronger teacher model (distillation), the same general technique used to build recent CUAD-derived instruction datasets. This is a legitimate, citable method — document the exact prompt used for generation in `data/scripts/` so it's reproducible.
- Generate both positive (accurate rewrite) and negative/contrastive examples (a rewrite that subtly drops or inverts an obligation) so the meaning-preservation verifier in Step 5a has real negative examples to be evaluated against, not just positives.

## 7. Safety & Trust Non-Negotiables

These apply regardless of how far behind schedule or how tempting a shortcut looks for a demo:
1. No risk flag ships without a citation.
2. No citation ships without passing the citation-grounding verifier.
3. No rewrite ships without passing the meaning-preservation verifier — show the original clause instead if it fails.
4. Low-confidence classifications are marked "needs human review," never silently guessed.
5. "Not legal advice" messaging is a persistent UI element wherever a flag is shown, not a one-time disclaimer.
6. The grounding agent must know the document's jurisdiction/type before citing law — never default silently to one jurisdiction's statutes.
7. No persistent storage of raw uploaded documents beyond the session unless a clear, explicit data-retention design is agreed on first (real contracts contain names, addresses, salaries — treat this as a decision to make deliberately, not a default to fall into).

## 8. Model Training Plan

### 8.1 Base model
As of mid-2026, small (~3B) instruct models suitable for QLoRA on a free Colab T4: **SmolLM3-3B** (Apache 2.0, Hugging Face's own model with a fully published training recipe — useful for writing up methodology honestly), **Qwen3 small variants**, **Llama-3.2-3B-Instruct**, **Phi-4-mini-instruct**. This list moves fast — re-check current small-model leaderboards at the point you actually start training rather than locking this in now, since there's no fixed deadline and a better small model may exist by then. Whatever is chosen, it needs to be small enough for QLoRA on a single T4 (16GB) and have a permissive license (Apache 2.0/MIT) for a resume/portfolio project.

### 8.2 Task setup
Multi-task LoRA on one base model:
- Task A: clause risk classification (multi-label, against Section 6.2's taxonomy)
- Task B: plain-language simplification (seq2seq style, via distillation data from 6.4)
Use a task-prefix or instruction-format approach to route the same adapter between tasks rather than training two separate adapters, unless evaluation shows task interference — check this empirically, don't assume it either way.

### 8.3 Training environment
- Google Colab free tier (T4, ~16GB VRAM, session limits apply). Checkpoint frequently — free-tier sessions disconnect without warning.
- QLoRA (4-bit base + LoRA adapters) to fit comfortably in T4 memory.
- Export the adapter (not a merged model) for portability; decide at deployment time whether to merge for inference speed or keep adapter-based for flexibility.

### 8.4 Known hard problems to test explicitly, not assume away
- **Clause segmentation on raw uploaded contracts.** CUAD's master CSV comes pre-segmented — production ingestion of a raw PDF/DOCX does not. Real contracts have messy paragraph breaks and cross-references ("subject to clause 4.2"). This needs its own eval, not an assumption that a naive paragraph-splitter is good enough.
- **OCR quality** for scanned documents — a confidence check on extraction quality should gate whether downstream analysis even runs.

## 9. Grounding Corpus

- Small, curated corpus of statute/standard-contract-language text relevant to Phase 1 document types, indexed for retrieval with jurisdiction/document-type metadata attached to each entry.
- Quality over coverage for v1 — a small, verified, correctly-cited corpus beats a large scraped one with uncertain provenance, given Section 7's non-negotiables depend entirely on this corpus being trustworthy.

## 10. Backend

- FastAPI service on Hugging Face Spaces.
- Structure: one module per pipeline step (ingestion, simplifier, classifier, grounding, verifier_meaning, verifier_citation, orchestrator), each independently testable.
- Inference: quantized CPU inference or HF Spaces ZeroGPU (free, shared A100 slices for eligible Spaces) — decide based on measured latency once the model is trained; don't over-provision speculatively.

## 11. Frontend

- Next.js on Vercel.
- UI/UX explored and drafted in Google Stitch (multi-screen generation, exports HTML/Tailwind), handed off via Stitch's Antigravity export path.
- **Important:** Stitch's code export is currently HTML/CSS/Tailwind, not native React. Treat Stitch output as a strong first draft for visual direction and layout, not drop-in code — translate it into proper React components rather than assuming it's production-ready as exported.
- Core screen to get right: the clause-by-clause report view (original | rewrite | flag+citation | verified badge), readable by someone who isn't a lawyer, without overwhelming them.

## 12. Deployment

- Vercel (frontend) ↔ HF Spaces (backend API) over a simple REST interface.
- Secrets/env vars for any API keys — never hardcoded.
- No deploy to a public URL until local test gates (Section 13 checklist) pass. Confirm with Harshit before any production deploy step, even if the pipeline seems to be working locally.

## 13. Build Order / Milestones

- [ ] **M1 — Taxonomy + data pipeline**: pull CUAD, confirm category mapping against real `category_descriptions.csv`, build contract-level splits, generate simplification distillation data, hand-curate adversarial eval set.
- [ ] **M2 — Model training**: QLoRA fine-tune on Colab, evaluate against held-out CUAD test set + adversarial set, export adapter.
- [ ] **M3 — Grounding corpus**: curate and index statute/standard-clause text for Phase 1 document types.
- [ ] **M4 — Backend agent pipeline**: implement each step in Section 5 independently, with unit tests per module before wiring the full chain together.
- [ ] **M5 — Frontend**: Stitch exploration → Antigravity translation into Next.js, wired to backend API.
- [ ] **M6 — Deployment plumbing**: Vercel + HF Spaces, environment config, end-to-end smoke test.
- [ ] **M7 — Repo scaffolding**: set up early (see Section 14) so nothing gets bolted on inconsistently later — ideally done alongside M1, not after.

Work in this order. Don't start frontend polish before the model and verifiers exist — there's nothing real to show yet.

## 14. Repo Structure

```
LawUP/
  PLAN.md                 (this file)
  README.md
  data/
    raw/                  (gitignored — CUAD downloads)
    processed/
    scripts/               (data prep, label mapping, distillation generation)
  training/
    configs/
    notebooks/             (Colab QLoRA training notebooks)
    eval/
  backend/
    app/
      agents/
        ingestion.py
        simplifier.py
        classifier.py
        grounding.py
        verifier_meaning.py
        verifier_citation.py
        orchestrator.py
      api/
      models/              (adapter loading, inference wrappers)
    tests/
    Dockerfile
    requirements.txt
  frontend/                (Next.js app; Stitch output translated here)
  grounding_corpus/
    statutes/
    indexing_scripts/
  .env.example
  .gitignore
```

## 15. Success Metrics

- Classifier: per-category F1 and macro-F1 on CUAD held-out test split; pass rate on the hand-curated adversarial set.
- Simplification: meaning-preservation pass rate (entailment threshold), readability improvement (e.g., Flesch-Kincaid delta) on passing rewrites only.
- End-to-end: citation precision (spot-checked manually against real statute text), latency per document, confidence-gate trigger rate (how often the system correctly says "needs review" rather than guessing).

## 16. Open Questions for Later Phases

- Phase 3 rental-agreement dataset: build plan for hand-curating Indian tenancy-agreement clause examples (source, size, labeling process) — not started.
- Multi-language support — not scoped yet.
- Redline/diff mode and clause-embedding anomaly scoring (flagging statistical outliers even without a matching known-risk category) — designed conceptually, not yet specced for implementation.

---

**Note to whoever is implementing this (human or agent):** Sections 6.1's "confirm against the real file" instruction and Section 7's non-negotiables are the two places in this document where guessing instead of checking would do real damage — one to the project's technical credibility, the other to someone relying on this tool's output. Everywhere else, ask if something is unclear rather than assuming.
