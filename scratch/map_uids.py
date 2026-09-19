import re

# Read the 23 prefixes
with open(r'c:\Users\hmhar\Projects\LawUP\data\processed\adversarial_holdout_uids.txt', 'r', encoding='utf-8') as f:
    prefixes = [line.strip() for line in f if line.strip()]

assert len(prefixes) == 23, "Expected 23 prefixes"

# Read the 33 candidates
with open(r'C:\Users\hmhar\.gemini\antigravity\brain\03dc701e-63f2-47c5-a03a-84712fac3580\adversarial_candidates.md', 'r', encoding='utf-8') as f:
    text = f.read()

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

with open(r'c:\Users\hmhar\Projects\LawUP\data\processed\adversarial_holdout_uids.txt', 'w', encoding='utf-8') as f:
    for uid in final_uids:
        f.write(uid + "\n")
