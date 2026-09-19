import os
import sys
import json
import time
import httpx
from openai import OpenAI
import statistics

# Add root to sys.path so we can import shared
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from shared.prompt_utils import format_prompt  # noqa: E402

UIDS_FILE = "data/processed/adversarial_holdout_uids.txt"
PAIRS_FILE = "data/processed/simplification_pairs.jsonl"
OUTPUT_FILE = "training/eval/adversarial_verification_results.md"

# Initialize synchronous OpenAI client pointed at LM Studio
lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")

def main():
    print("Loading UIDs...")
    with open(UIDS_FILE, 'r', encoding='utf-8') as f:
        target_uids = set(line.strip() for line in f if line.strip())
        
    print(f"Found {len(target_uids)} UIDs. Loading original texts...")
    adversarial_examples = {}
    with open(PAIRS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Check if the exact UID is in our holdout list
            if data["_uid"] in target_uids:
                adversarial_examples[data["_uid"]] = data["original_clause"]
                
    assert len(adversarial_examples) == 23, f"Expected 23 unique adversarial clauses, got {len(adversarial_examples)}"
        
    print(f"Connecting to LMStudio at localhost:1234...")
    
    results = []
    latencies = []
    
    print("Running generation...")
    for uid, original_text in adversarial_examples.items():
        print(f"Processing UID: {uid}...")
        prompt = format_prompt(original_text)
        
        start_time = time.time()
        try:
            response = client.completions.create(
                model="LawUP",
                prompt=prompt,
                max_tokens=512,
                temperature=0.0,
                stop=["<|endoftext|>", "\n\nOriginal Clause:", "\n\nNow it's your turn"]
            )
            rewrite = response.choices[0].text.strip()
        except httpx.ConnectError:
            print("ERROR: LM Studio not reachable at localhost:1234. Is the local server running?")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR calling LMStudio: {e}")
            rewrite = f"[ERROR: {str(e)}]"
            
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        
        results.append({
            "uid": uid,
            "original": original_text,
            "rewrite": rewrite,
            "latency_ms": latency
        })
        
    print(f"Generation complete. Latency: Min {min(latencies):.2f}ms | Median {statistics.median(latencies):.2f}ms | Max {max(latencies):.2f}ms")
    
    print(f"Writing results to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Adversarial Holdout Verification Results\n\n")
        f.write(f"**Latency Metrics:** Min {min(latencies):.2f}ms | Median {statistics.median(latencies):.2f}ms | Max {max(latencies):.2f}ms\n")
        f.write("**Backend:** LMStudio (Q8 GGUF)\n")
        f.write("**Decoding:** Greedy (`temperature=0.0`)\n\n")
        f.write("---\n\n")
        
        for res in results:
            f.write(f"### UID: `{res['uid']}`\n")
            f.write(f"**Latency:** {res['latency_ms']:.2f} ms\n\n")
            f.write("**Original Clause:**\n")
            f.write(f"> {res['original'].replace(chr(10), chr(10) + '> ')}\n\n")
            f.write("**SmolLM3 Rewrite:**\n")
            f.write(f"```text\n{res['rewrite']}\n```\n\n")
            f.write("---\n\n")
            
    jsonl_output = OUTPUT_FILE.replace('.md', '.jsonl')
    print(f"Writing raw JSONL to {jsonl_output}...")
    with open(jsonl_output, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
