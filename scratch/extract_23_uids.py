import re

with open(r'C:\Users\hmhar\.gemini\antigravity\brain\03dc701e-63f2-47c5-a03a-84712fac3580\adversarial_candidates.md', 'r', encoding='utf-8') as f:
    text = f.read()

uids = re.findall(r'\*\*UID:\*\*\s*(.+)', text)
uids = [u.strip() for u in uids if u.strip()]

assert len(uids) == 23, f"Expected exactly 23 UIDs, but found {len(uids)}"

from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
output_path = root_dir / "data" / "processed" / "adversarial_holdout_uids.txt"

with open(output_path, 'w', encoding='utf-8') as f:
    for uid in uids:
        f.write(f"{uid}\n")

print(f"Successfully extracted and wrote {len(uids)} UIDs to adversarial_holdout_uids.txt")
