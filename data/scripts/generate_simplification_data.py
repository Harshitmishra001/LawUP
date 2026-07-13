"""Generate clause-simplification training pairs.

Uses DeepSeek V4 Flash via OpenRouter for positive rewrites.
Uses programmatic rule-based corruptions for contrastive negatives (free).

Includes:
  - Explicit reasoning disable via OpenRouter's reasoning.effort="none"
  - finish_reason guard: only saves examples that completed cleanly ("stop")
  - Incremental writing with resume support
  - Running budget tracking with hard stop

Usage:
    python -m data.scripts.generate_simplification_data --dry-run
    python -m data.scripts.generate_simplification_data --test-batch
    python -m data.scripts.generate_simplification_data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "deepseek/deepseek-v4-flash"
MAX_TOKENS = 1500

# Estimated DeepSeek V4 Flash costs on OpenRouter
COST_PER_1M_INPUT = 0.14
COST_PER_1M_OUTPUT = 0.28
MAX_BUDGET = 1.40
DRY_RUN_BUDGET_WARNING = 1.30

# ---------------------------------------------------------------------------
# Prompt templates (system prompt is IDENTICAL across all calls for caching)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a legal-plain-language expert."

POSITIVE_REWRITE_USER = """Rewrite the following contract clause in clear, everyday English that a non-lawyer can understand.

RULES:
1. Preserve ALL obligations, conditions, deadlines, and parties exactly.
2. Do NOT add, remove, or invert any condition or obligation.
3. Use short sentences and active voice.
4. Keep defined terms (e.g., "Licensee", "Effective Date") unchanged.
5. Output ONLY the rewritten clause — no commentary.

ORIGINAL CLAUSE:
\"\"\"
{clause_text}
\"\"\"

PLAIN-LANGUAGE REWRITE:
"""

# ---------------------------------------------------------------------------
# State & Tracking
# ---------------------------------------------------------------------------

current_spend = 0.0
rejected_count = 0

def _get_api_key() -> str:
    load_dotenv()
    key = os.getenv("Open_router_key") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("API key not found. Please set 'Open_router_key' in .env")
    return key


def _call_teacher_model(user_prompt: str, api_key: str) -> tuple[str, dict[str, int], str]:
    """Call OpenRouter API. Returns (response_text, usage_dict, finish_reason)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/hmhar/LawUP",
        "X-Title": "LawUP",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
        # Explicitly disable reasoning to prevent silent token burn
        "reasoning": {
            "effort": "none"
        }
    }

    import time
    for attempt in range(4):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if not resp.ok:
                logger.error(f"API Error {resp.status_code}: {resp.text}")
                sys.exit(1)
            break
        except requests.exceptions.RequestException as e:
            if attempt < 3:
                logger.warning(f"Network error on attempt {attempt+1}, retrying in 5s... Error: {e}")
                time.sleep(5)
            else:
                logger.error(f"Network error persisted after 3 retries. Exiting. Error: {e}")
                sys.exit(1)

    data = resp.json()
    choice = data["choices"][0]
    finish_reason = choice.get("finish_reason", "unknown")
    message = choice["message"]
    content = message.get("content")

    if content is None:
        content = ""

    usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
    return content.strip(), usage, finish_reason


def _update_and_check_budget(usage: dict[str, int]):
    global current_spend
    in_tokens = usage.get("prompt_tokens", 0)
    out_tokens = usage.get("completion_tokens", 0)
    cost = (in_tokens / 1_000_000) * COST_PER_1M_INPUT + (out_tokens / 1_000_000) * COST_PER_1M_OUTPUT
    current_spend += cost

    if current_spend > MAX_BUDGET:
        logger.error(f"HARD STOP: Running spend ${current_spend:.4f} exceeds max budget ${MAX_BUDGET:.2f}.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Programmatic Negative Generation
# ---------------------------------------------------------------------------

def _generate_rule_based_negative(positive_rewrite: str) -> tuple[str, str]:
    """Applies a random rule-based corruption to the positive rewrite.
    
    Returns (corrupted_text, corruption_type_label).
    The corruption_type is a precise, labeled description of what was changed.
    """
    if not positive_rewrite:
        return "", "empty_input"

    corruptions = [
        # Obligation inversion
        (r"\b(shall|will|must)\b", "may", "obligation_to_permission"),
        (r"\b(may|can)\b", "shall", "permission_to_obligation"),
        (r"\b(agrees to)\b", "does not agree to", "obligation_negated"),
        (r"\b(is required to)\b", "is not required to", "obligation_negated"),
        (r"\b(is responsible for)\b", "is not responsible for", "obligation_negated"),

        # Deadline/number distortion
        (r"\b(\d{1,3})\s+(days?)\b", r"5 \2", "deadline_number_changed"),
        (r"\b(\d{1,3})\s+(months?)\b", r"1 \2", "deadline_number_changed"),
        (r"\b(days|months|years)\b", "weeks", "time_unit_changed"),

        # Logical/scope inversion
        (r"\b(including)\b", "excluding", "inclusion_to_exclusion"),
        (r"\b(excluding)\b", "including", "exclusion_to_inclusion"),
        (r"\b(before)\b", "after", "temporal_inversion"),
        (r"\b(after)\b", "before", "temporal_inversion"),
        (r"\b(all)\b", "no", "scope_inversion"),
        (r"\b(either party)\b", "only one party", "party_scope_changed"),
        (r"\b(both parties)\b", "one party", "party_scope_changed"),
        (r"\b(written)\b", "verbal", "formality_changed"),

        # Condition injection (fallback — always matches)
        (r"\.$", " unless the other party objects in writing.", "condition_appended"),
    ]

    random.shuffle(corruptions)
    for pattern, replacement, corruption_type in corruptions:
        corrupted, count = re.subn(pattern, replacement, positive_rewrite, count=1, flags=re.IGNORECASE)
        if count > 0:
            return corrupted, corruption_type

    # Ultimate fallback
    return positive_rewrite + " However, this provision is not enforceable.", "disclaimer_appended"

# ---------------------------------------------------------------------------
# Data loading & Resume logic
# ---------------------------------------------------------------------------

import re

# Redaction regex: matches [***], [*], [REDACTED], [Date], or any brackets with upper case / asterisks
REDACTION_REGEX = re.compile(r'\[\s*\*{1,}\s*\]|\[\s*REDACTED\s*\]|\[\s*Date\s*\]|\[\s*[A-Z_]+\s*\]', re.IGNORECASE)

def _load_positive_clauses(input_file: Path) -> list[dict[str, Any]]:
    records = []
    with input_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("has_clause"):
                clause_text = rec["clause_text"]
                words = len(clause_text.split())
                
                # Filter out headers, bare dates, and label-only annotations
                if words < 5:
                    logger.debug(f"Skipping tiny clause ({words} words): {clause_text}")
                    continue
                    
                # Filter out clauses with redactions/placeholders to prevent LLM hallucinations
                if REDACTION_REGEX.search(clause_text):
                    logger.debug(f"Skipping redacted clause: {clause_text[:100]}...")
                    continue
                    
                rec_id = f"{rec['contract_name']}_{rec['cuad_category']}"
                rec["_uid"] = rec_id
                rec["clause_words"] = words
                records.append(rec)
    return records


def _get_processed_uids(output_file: Path) -> set[str]:
    processed = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                uid = rec.get("_uid")
                if uid and rec.get("is_faithful"):
                    processed.add(uid)
    return processed

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _dry_run(records: list[dict[str, Any]]):
    """Print payload and accurately estimate cost."""
    logger.info("=" * 80)
    logger.info("EXACT REQUEST PAYLOAD (Dry Run)")
    logger.info("=" * 80)
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": POSITIVE_REWRITE_USER.format(clause_text="[CUAD CLAUSE TEXT]")}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
        "reasoning": {"effort": "none"}
    }
    print("URL:", OPENROUTER_URL)
    print("HEADERS: Authorization=Bearer sk-or-v1-****, HTTP-Referer=https://github.com/hmhar/LawUP, X-Title=LawUP")
    print("PAYLOAD:", json.dumps(payload, indent=2))
    
    # Cost projection
    num_records = len(records)
    
    # Calculate exact distribution
    clause_lengths = [r["clause_words"] for r in records]
    avg_words = sum(clause_lengths) / len(clause_lengths) if clause_lengths else 0
    
    prompt_template_tokens = int((len(SYSTEM_PROMPT.split()) + len(POSITIVE_REWRITE_USER.split())) * 1.3)
    
    total_est_input_tokens = 0
    total_est_output_tokens = 0
    
    for words in clause_lengths:
        # tokens ≈ words * 1.3
        clause_tokens = int(words * 1.3)
        input_tokens = prompt_template_tokens + clause_tokens
        
        # Simplification usually reduces length by ~20%. 
        # Bounded by MAX_TOKENS so we don't project impossible output sizes.
        est_output = min(int(clause_tokens * 0.8) + 20, MAX_TOKENS)
        
        total_est_input_tokens += input_tokens
        total_est_output_tokens += est_output
        
    est_cost = (total_est_input_tokens / 1_000_000) * COST_PER_1M_INPUT + (total_est_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
    
    logger.info("=" * 80)
    logger.info("PROJECTED COST CALCULATION (Proportional to Real Distribution)")
    logger.info("=" * 80)
    logger.info(f"  Target count:     {num_records} clauses (filtered < 5 words)")
    logger.info(f"  API calls:        {num_records} (positives only)")
    logger.info(f"  Max Tokens Cap:   {MAX_TOKENS}")
    logger.info(f"  Total input:      {total_est_input_tokens:,} tokens × $0.14/1M = ${total_est_input_tokens/1e6 * COST_PER_1M_INPUT:.4f}")
    logger.info(f"  Total output:     {total_est_output_tokens:,} tokens × $0.28/1M = ${total_est_output_tokens/1e6 * COST_PER_1M_OUTPUT:.4f}")
    logger.info(f"  PROJECTED TOTAL:  ${est_cost:.4f}")
    
    if est_cost > DRY_RUN_BUDGET_WARNING:
        logger.warning(f"⚠ Projected cost ${est_cost:.4f} > ${DRY_RUN_BUDGET_WARNING:.2f}! Review before proceeding.")


def run_generation(
    input_file: Path,
    output_file: Path,
    rejected_file: Path | None = None,
    dry_run: bool = False,
    test_batch: bool = False,
):
    global rejected_count
    records = _load_positive_clauses(input_file)
    if not records:
        logger.warning("No positive clauses found.")
        return

    if dry_run:
        _dry_run(records)
        return

    api_key = _get_api_key()

    if rejected_file is None:
        rejected_file = output_file.parent / "rejected_examples.jsonl"

    if test_batch:
        # Pick 3 diverse examples: shortest, longest, median
        if len(records) >= 3:
            sorted_by_len = sorted(records, key=lambda r: len(r["clause_text"]))
            picks = [
                sorted_by_len[0],                           # shortest
                sorted_by_len[len(sorted_by_len) // 2],     # median
                sorted_by_len[-1],                           # longest
            ]
            records = picks
        else:
            records = records[:3]
        logger.info("Running 3-example TEST BATCH (shortest / median / longest)...")
    else:
        processed = _get_processed_uids(output_file)
        records = [r for r in records if r["_uid"] not in processed]
        logger.info(f"Found {len(processed)} already processed. {len(records)} remaining.")
        if not records:
            return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a", encoding="utf-8") as fh, \
         rejected_file.open("a", encoding="utf-8") as rh:
        for idx, rec in enumerate(records):
            clause = rec["clause_text"]
            clause_words = len(clause.split())
            logger.info(f"Processing {idx+1}/{len(records)}: {rec['contract_name']} - {rec['cuad_category']} ({clause_words} words)")

            pos_prompt = POSITIVE_REWRITE_USER.format(clause_text=clause)

            # Positive rewrite via LLM
            pos_rewrite, usage, finish_reason = _call_teacher_model(pos_prompt, api_key)
            _update_and_check_budget(usage)

            in_tokens = usage.get("prompt_tokens", 0)
            out_tokens = usage.get("completion_tokens", 0)
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) if isinstance(usage.get("completion_tokens_details"), dict) else 0

            if test_batch:
                print(f"\n{'='*80}")
                print(f"TEST BATCH OUTPUT {idx+1}")
                print(f"{'='*80}")
                print(f"CLAUSE LENGTH: {clause_words} words")
                print(f"ORIGINAL:\n{clause}")
                print(f"\nFAITHFUL REWRITE (LLM):\n{pos_rewrite}")
                print(f"\nfinish_reason: {finish_reason}")
                print(f"Tokens: prompt={in_tokens}, completion={out_tokens}, reasoning={reasoning_tokens}")

            # SAFETY GATE: Only save if finish_reason == "stop"
            if finish_reason != "stop":
                logger.warning(
                    f"REJECTED: finish_reason='{finish_reason}' for {rec['_uid']}. "
                    f"Tokens: prompt={in_tokens}, completion={out_tokens}, reasoning={reasoning_tokens}. "
                    f"NOT saved as training data."
                )
                reject_record = {
                    "_uid": rec["_uid"],
                    "finish_reason": finish_reason,
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "partial_content": pos_rewrite[:200] if pos_rewrite else "",
                    "clause_words": clause_words,
                }
                rh.write(json.dumps(reject_record, ensure_ascii=False) + "\n")
                rh.flush()
                rejected_count += 1
                if test_batch:
                    print(f"⚠ REJECTED — not saved as training data")
                continue

            pos_result = {
                "_uid": rec["_uid"],
                "original_clause": clause,
                "rewrite": pos_rewrite,
                "is_faithful": True,
                "cuad_category": rec["cuad_category"],
                "contract_name": rec["contract_name"],
            }
            fh.write(json.dumps(pos_result, ensure_ascii=False) + "\n")

            # Negative pair via deterministic rule corruption
            neg_rewrite, corruption_type = _generate_rule_based_negative(pos_rewrite)
            neg_result = {
                "_uid": rec["_uid"],
                "original_clause": clause,
                "rewrite": neg_rewrite,
                "is_faithful": False,
                "corruption_type": corruption_type,
                "cuad_category": rec["cuad_category"],
                "contract_name": rec["contract_name"],
            }
            fh.write(json.dumps(neg_result, ensure_ascii=False) + "\n")
            fh.flush()

            if test_batch:
                print(f"\nUNFAITHFUL REWRITE (Programmatic: '{corruption_type}'):\n{neg_rewrite}")
                print(f"{'='*80}\n")

            # Periodic status every 100 examples
            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx+1}/{len(records)} | Spend: ${current_spend:.4f} | Rejected: {rejected_count}")

    logger.info(f"Done. Total spend: ${current_spend:.4f} | Total rejected: {rejected_count}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/simplification_pairs.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-batch", action="store_true")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    try:
        run_generation(
            input_file=args.input_file,
            output_file=args.output_file,
            dry_run=args.dry_run,
            test_batch=args.test_batch,
        )
    except Exception:
        logger.exception("Generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
