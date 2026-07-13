import os
import json
import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
PAIRS_FILE = os.path.join(PROCESSED_DIR, "simplification_pairs.jsonl")
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
MAX_SEQ_LENGTH = 1024

def get_split_contracts():
    """Extract contract names belonging to each split from the raw CUAD data."""
    splits = {"train": set(), "val": set(), "test": set()}
    for split_name in splits.keys():
        filepath = os.path.join(PROCESSED_DIR, f"{split_name}.jsonl")
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                splits[split_name].add(data["contract_name"])
    return splits

def format_prompt(original_clause):
    # Standard instruction format for simplification
    prompt = (
        "You are an expert legal assistant. Your task is to rewrite the following complex legal clause "
        "into plain, easy-to-understand English. Ensure that you preserve all original obligations, "
        "conditions, and permissions without adding or removing any meaning.\n\n"
        f"Original Clause:\n{original_clause}\n\n"
        "Plain English Rewrite:\n"
    )
    return prompt

import random
from collections import defaultdict

def main():
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits_contracts = get_split_contracts()
    
    # Store valid examples grouped by contract_name
    valid_examples_by_contract = defaultdict(list)
    
    total_processed = 0
    dropped_unfaithful = 0
    dropped_length = 0
    dropped_uids = []

    with open(PAIRS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total_processed += 1
            
            if not data.get("is_faithful", False):
                dropped_unfaithful += 1
                continue
                
            uid = data["_uid"]
            contract_name = data["contract_name"]
            
            # Explicit isolation check: ensure contract is strictly in CUAD train split
            if contract_name not in splits_contracts["train"]:
                logger.error(f"Leakage or unknown contract detected: {contract_name} is not in CUAD train split!")
                # Depending on how strict, we could crash here. We will just drop or log it.
                # For safety, let's strictly assert:
                assert contract_name not in splits_contracts["test"], f"CRITICAL LEAKAGE: {contract_name} is in test split!"
                assert contract_name not in splits_contracts["val"], f"CRITICAL LEAKAGE: {contract_name} is in val split!"
                continue

            prompt = format_prompt(data["original_clause"])
            completion = data["rewrite"]
            full_text = prompt + completion + tokenizer.eos_token
            
            tokens = tokenizer(full_text, truncation=False)
            num_tokens = len(tokens["input_ids"])
            
            if num_tokens > MAX_SEQ_LENGTH:
                dropped_length += 1
                dropped_uids.append(uid)
                continue
                
            valid_examples_by_contract[contract_name].append({
                "uid": uid,
                "text": full_text,
                "category": data.get("cuad_category", "Unknown")
            })

    # Internal 90/10 Split by Contract
    all_contracts = list(valid_examples_by_contract.keys())
    # Sort for deterministic behavior before random shuffle (with fixed seed)
    all_contracts.sort()
    random.seed(42)
    random.shuffle(all_contracts)
    
    num_val_contracts = max(1, int(len(all_contracts) * 0.10))
    val_contracts = set(all_contracts[:num_val_contracts])
    train_contracts = set(all_contracts[num_val_contracts:])
    
    output_data = {"train": [], "val": []}
    category_counts = {"train": defaultdict(int), "val": defaultdict(int)}

    for contract_name, examples in valid_examples_by_contract.items():
        target_split = "val" if contract_name in val_contracts else "train"
        output_data[target_split].extend(examples)
        for ex in examples:
            category_counts[target_split][ex["category"]] += 1

    # Save datasets
    for split_name in ["train", "val", "test"]:
        out_path = os.path.join(PROCESSED_DIR, f"{split_name}_qlora.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            if split_name in output_data:
                for ex in output_data[split_name]:
                    f.write(json.dumps({"uid": ex["uid"], "text": ex["text"]}) + "\n")
            logger.info(f"Saved {len(output_data.get(split_name, []))} examples to {out_path}")

    # Log dropped UIDs
    dropped_log_path = os.path.join(PROCESSED_DIR, "dropped_length_uids.txt")
    with open(dropped_log_path, 'w', encoding='utf-8') as f:
        for uid in dropped_uids:
            f.write(uid + "\n")

    logger.info("=== Data Formatting Report ===")
    logger.info(f"Total pairs processed: {total_processed}")
    logger.info(f"Dropped (unfaithful negatives reserved for M4): {dropped_unfaithful}")
    logger.info(f"Dropped (exceeded {MAX_SEQ_LENGTH} tokens): {dropped_length}")
    logger.info(f"Total remaining for training/eval: {len(output_data['train']) + len(output_data['val'])}")
    logger.info(f"Train/Val Contract Split: {len(train_contracts)} train / {len(val_contracts)} val")
    logger.info(f"Train/Val Example Split: {len(output_data['train'])} train / {len(output_data['val'])} val")
    
    logger.info("=== Validation Set Category Breakdown ===")
    for cat, count in sorted(category_counts["val"].items()):
        logger.info(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
