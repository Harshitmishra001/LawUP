import json

with open('data/processed/adversarial_holdout_uids.txt', 'r', encoding='utf-8') as f:
    prefixes = [line.strip() for line in f if line.strip()]

total = 0
with open('data/processed/simplification_pairs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if any(data["_uid"].startswith(p) for p in prefixes):
            total += 1

print(f"Total matching clauses: {total}")
