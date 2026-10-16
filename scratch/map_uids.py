import re
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
holdout_path = root_dir / "data" / "processed" / "adversarial_holdout_uids.txt"

# Read the 23 prefixes
with open(holdout_path, 'r', encoding='utf-8') as f:
    prefixes = [line.strip() for line in f if line.strip()]

assert len(prefixes) == 23, "Expected 23 prefixes"

# Read the 33 candidates
candidates_md = Path.home() / ".gemini" / "antigravity" / "brain" / "03dc701e-63f2-47c5-a03a-84712fac3580" / "adversarial_candidates.md"
if candidates_md.exists():
    with open(candidates_md, 'r', encoding='utf-8') as f:
        text = f.read()
else:
    text = ""

candidate_uids = re.findall(r'\*\*UID:\*\*\s*(.+)', text)
candidate_uids = [u.strip() for u in candidate_uids if u.strip()]

# Find matching full UIDs
final_uids = []
for prefix in prefixes:
    matches = [uid for uid in candidate_uids if uid.startswith(prefix)]
    if matches:
        final_uids.append(matches[0]) # Take the first match if multiple
    else:
        print(f"Warning: No match found for prefix {prefix}")

print(f"Found {len(final_uids)} full UIDs")

if final_uids:
    with open(holdout_path, 'w', encoding='utf-8') as f:
        for uid in final_uids:
            f.write(uid + "\n")
