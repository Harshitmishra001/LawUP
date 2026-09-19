import json
import logging
from pathlib import Path
import random
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_classification_data(split_name: str):
    input_file = Path(f"data/processed/{split_name}.jsonl")
    output_file = Path(f"data/processed/{split_name}_cls.jsonl")
    
    if not input_file.exists():
        logger.error(f"Missing {input_file}")
        return

    # Group by contract_name
    contracts = {}
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c_name = rec["contract_name"]
            if c_name not in contracts:
                contracts[c_name] = {"context": rec.get("context", ""), "positives": []}
            if rec.get("has_clause") and rec.get("lawup_risk_category") != "_metadata":
                contracts[c_name]["positives"].append(rec)

    output_records = []
    
    for c_name, data in contracts.items():
        positives = data["positives"]
        
        # Group positive clauses by exact text to handle multi-label
        # To handle slight overlap, we can normalize whitespace
        clause_map = {}
        for p in positives:
            text = p["clause_text"].strip()
            # simple deduplication by text
            if text not in clause_map:
                clause_map[text] = set()
            risk_cat = p["lawup_risk_category"]
            sub_cat = p["cuad_category"]
            clause_map[text].add((risk_cat, sub_cat))
            
        for text, labels in clause_map.items():
            # Create a multi-label positive example
            risks = []
            for r, s in labels:
                risks.append({"risk": r, "sub_category": s})
            
            output_records.append({
                "contract_name": c_name,
                "clause_text": text,
                "labels": risks,
                "is_positive": True
            })
            
        # Sample negative examples from the context
        # We split context by double newline (paragraphs)
        context = data["context"]
        if context:
            paragraphs = [p.strip() for p in context.split("\n\n") if len(p.split()) > 10]
            # filter out paragraphs that contain any positive clause
            negative_paras = []
            for p in paragraphs:
                is_overlap = False
                for pos_text in clause_map.keys():
                    # Check if pos_text is in p or p is in pos_text
                    if pos_text in p or p in pos_text:
                        is_overlap = True
                        break
                if not is_overlap:
                    negative_paras.append(p)
            
            # Sample up to 2 negative paragraphs per contract to balance
            sampled_negs = random.sample(negative_paras, min(2, len(negative_paras)))
            for n_text in sampled_negs:
                output_records.append({
                    "contract_name": c_name,
                    "clause_text": n_text,
                    "labels": [],
                    "is_positive": False
                })

    # Prepare for QLoRA
    # Format: 
    # SYSTEM: You are a legal risk classification expert.
    # USER: Analyze this contract clause and identify any legal risks. Output a JSON list of risks.
    # ORIGINAL CLAUSE:
    # """{text}"""
    # ASSISTANT: [JSON list]
    
    qlora_records = []
    system_prompt = "You are a legal risk classification expert."
    user_prompt_template = "Identify all applicable risk categories and sub-categories present in this clause. Output as a JSON list. Each risk should have a 'risk' field and a 'sub_category' field.\n\nORIGINAL CLAUSE:\n\"\"\"\n{clause_text}\n\"\"\""
    
    for r in output_records:
        user_msg = user_prompt_template.format(clause_text=r["clause_text"])
        ast_msg = json.dumps(r["labels"], indent=2)
        qlora_records.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ast_msg}
            ]
        })
        
    random.shuffle(qlora_records)
    with output_file.open("w", encoding="utf-8") as f:
        for q in qlora_records:
            f.write(json.dumps(q) + "\n")
            
    logger.info(f"{split_name}: Wrote {len(qlora_records)} classification pairs (positives & negatives) to {output_file.name}")

if __name__ == "__main__":
    generate_classification_data("train")
    generate_classification_data("val")
    generate_classification_data("test")
