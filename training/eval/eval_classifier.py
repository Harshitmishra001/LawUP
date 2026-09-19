import json
import os
import argparse
import time
import re
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

def parse_json_array(text):
    """Attempt to robustly extract a JSON list from model output."""
    try:
        # Sometimes models wrap in markdown blocks
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        clean_text = clean_text.strip()
        
        # Regex to find the first array
        match = re.search(r'\[(.*?)\]', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
            
        return json.loads(clean_text)
    except Exception:
        # Fallback if totally unparseable
        return []

def main():
    parser = argparse.ArgumentParser(description="Evaluate LawUP Classifier on test_cls.jsonl")
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, default="checkpoints_cls/final", help="Path to trained adapter")
    parser.add_argument("--test_file", type=str, default="data/processed/test_cls.jsonl", help="Path to test file")
    parser.add_argument("--output_file", type=str, default="training/eval/classifier_eval_report.txt", help="Output report")
    parser.add_argument("--max_examples", type=int, default=-1, help="Max examples to evaluate (for quick tests)")
    args = parser.parse_args()

    print(f"Loading Base Model: {args.base_model} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # `load_in_4bit` cannot be passed directly to from_pretrained in new transformers.
    # It must go through BitsAndBytesConfig.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # `torch_dtype` kwarg is deprecated — use `dtype` instead
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    print(f"Loading Adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    print(f"Loading Test Data from {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]

    if args.max_examples > 0:
        lines = lines[:args.max_examples]

    examples = [json.loads(line) for line in lines]
    print(f"Evaluating on {len(examples)} examples.")

    # Determine pad token id safely (0 is a valid token id, so don't use `or`)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    y_true = []
    y_pred = []

    start_time = time.time()
    for ex in tqdm(examples):
        msgs = ex["messages"]  # list of {role, content} dicts

        # Extract ground truth labels from the assistant message
        assistant_msg = next((m for m in msgs if m["role"] == "assistant"), None)
        if assistant_msg:
            true_labels_raw = parse_json_array(assistant_msg["content"])
            # Labels are dicts like {"risk": "...", "sub_category": "..."}
            # Use the "risk" field as the label key
            true_labels = [item.get("risk", str(item)) for item in true_labels_raw
                           if isinstance(item, dict)]
        else:
            true_labels = []

        # Build prompt: system + user only (no assistant turn) with add_generation_prompt=True
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,  # greedy — don't pass temperature when do_sample=False
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred_labels_raw = parse_json_array(generated_text)
        pred_labels = [item.get("risk", str(item)) for item in pred_labels_raw
                       if isinstance(item, dict)]

        y_true.append(set(true_labels))
        y_pred.append(set(pred_labels))

    eval_time = time.time() - start_time
    print(f"Evaluation finished in {eval_time:.2f}s ({eval_time/len(examples):.2f}s per example)")

    # Compute Metrics
    all_categories = set()
    for t in y_true: all_categories.update(t)
    for p in y_pred: all_categories.update(p)

    tp_global = 0
    fp_global = 0
    fn_global = 0

    cat_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for true_set, pred_set in zip(y_true, y_pred):
        # Global
        tp_global += len(true_set.intersection(pred_set))
        fp_global += len(pred_set - true_set)
        fn_global += len(true_set - pred_set)

        # Per category
        for cat in all_categories:
            if cat in true_set and cat in pred_set:
                cat_metrics[cat]["tp"] += 1
            elif cat in pred_set and cat not in true_set:
                cat_metrics[cat]["fp"] += 1
            elif cat in true_set and cat not in pred_set:
                cat_metrics[cat]["fn"] += 1

    def calc_p_r_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    p_glob, r_glob, f1_glob = calc_p_r_f1(tp_global, fp_global, fn_global)

    report = []
    report.append("=== LawUP Classifier Evaluation Report ===")
    report.append(f"Model: {args.base_model} + {args.adapter_path}")
    report.append(f"Test Set Size: {len(examples)}")
    report.append(f"Speed: {eval_time/len(examples):.2f} sec/example")
    report.append("")
    report.append("--- Global Metrics (Micro-Averaged) ---")
    report.append(f"Precision: {p_glob:.4f}")
    report.append(f"Recall:    {r_glob:.4f}")
    report.append(f"F1 Score:  {f1_glob:.4f}")
    report.append("")
    report.append("--- Per-Category Metrics ---")

    # Sort categories alphabetically
    for cat in sorted(all_categories):
        m = cat_metrics[cat]
        p, r, f1 = calc_p_r_f1(m["tp"], m["fp"], m["fn"])
        support = m["tp"] + m["fn"]
        report.append(f"{cat:<35} | P: {p:.2f} | R: {r:.2f} | F1: {f1:.2f} | Support: {support}")

    report_text = "\n".join(report)
    print("\n" + report_text)

    # Safely create output directory (handles case where dirname is empty string)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to {args.output_file}")

if __name__ == "__main__":
    main()

