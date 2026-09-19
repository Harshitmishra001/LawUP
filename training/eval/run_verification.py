import json
import os
import sys

# Add root to sys.path so we can import shared
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.app.agents.verifier_meaning import verify_meaning  # noqa: E402

INPUT_FILE = "training/eval/adversarial_verification_results.jsonl"
OUTPUT_FILE = "training/eval/adversarial_verification_report.md"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run verify_adversarial.py first.")
        sys.exit(1)
        
    results = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
                
    print(f"Found {len(results)} rewrites to verify.")
    
    verified_results = []
    passes = 0
    low_confidence_count = 0
    failure_counts = {
        "HALLUCINATION": 0,
        "CONTRADICTION": 0,
        "OMISSION": 0,
        "STRUCTURAL_COLLAPSE": 0
    }
    
    for i, res in enumerate(results):
        print(f"Verifying [{i+1}/{len(results)}]: {res['uid']}...")
        verification = verify_meaning(res["original"], res["rewrite"])
        
        # Merge verification data into result
        res["verification"] = verification
        verified_results.append(res)
        
        if verification.get("verdict") == "PASS":
            passes += 1
            
        if verification.get("confidence") == "LOW":
            low_confidence_count += 1
            
        for failure in verification.get("failures", []):
            cat = failure.get("category")
            if cat in failure_counts:
                failure_counts[cat] += 1
            else:
                failure_counts[cat] = failure_counts.get(cat, 0) + 1
                
    # Generate Markdown Report
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Adversarial Meaning-Preservation Verification Report\n\n")
        
        pass_rate = (passes / len(results)) * 100 if results else 0
        f.write(f"**Overall Pass Rate:** {pass_rate:.1f}% ({passes}/{len(results)})\n")
        f.write("> *Note: This is evaluated on a small curated n={len(results)} adversarial holdout set. Failure categorization is the primary signal.*\n\n")
        
        f.write("### Failure Breakdown\n")
        for cat, count in failure_counts.items():
            f.write(f"- **{cat}:** {count}\n")
            
        f.write(f"\n**Low Confidence Flags:** {low_confidence_count}\n")
        f.write("---\n\n")
        
        # Write LOW confidence first
        f.write("## ⚠️ Priority Review (LOW Confidence)\n\n")
        has_low = False
        for res in verified_results:
            if res["verification"].get("confidence") == "LOW":
                has_low = True
                write_result(f, res)
        if not has_low:
            f.write("*No low confidence results flagged.*\n\n")
            
        f.write("---\n\n")
        f.write("## ❌ Failed Verifications\n\n")
        has_fails = False
        for res in verified_results:
            if res["verification"].get("verdict") == "FAIL" and res["verification"].get("confidence") != "LOW":
                has_fails = True
                write_result(f, res)
        if not has_fails:
            f.write("*No high-confidence failures found.*\n\n")
            
        f.write("---\n\n")
        f.write("## ✅ Passed Verifications\n\n")
        for res in verified_results:
            if res["verification"].get("verdict") == "PASS" and res["verification"].get("confidence") != "LOW":
                write_result(f, res)
                
    print(f"Report written to {OUTPUT_FILE}")

def write_result(f, res):
    v = res["verification"]
    f.write(f"### UID: `{res['uid']}`\n")
    f.write(f"**Verdict:** {v.get('verdict', 'ERROR')} | **Confidence:** {v.get('confidence', 'UNKNOWN')}\n")
    f.write(f"- Original Entails Rewrite: `{v.get('original_entails_rewrite')}`\n")
    f.write(f"- Rewrite Entails Original: `{v.get('rewrite_entails_original')}`\n\n")
    
    if v.get("failures"):
        f.write("**Failures:**\n")
        for fail in v["failures"]:
            f.write(f"- **{fail.get('category')}**: {fail.get('description')}\n")
            if fail.get('original_span'):
                f.write(f"  - *Original span*: \"{fail['original_span']}\"\n")
            if fail.get('rewrite_span'):
                f.write(f"  - *Rewrite span*: \"{fail['rewrite_span']}\"\n")
        f.write("\n")
        
    f.write("**Original Clause:**\n")
    f.write(f"> {res['original'].replace(chr(10), chr(10) + '> ')}\n\n")
    f.write("**SmolLM3 Rewrite:**\n")
    f.write(f"```text\n{res['rewrite']}\n```\n\n")
    f.write("---\n\n")

if __name__ == "__main__":
    main()
