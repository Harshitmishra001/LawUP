import re
from pathlib import Path

candidates_path = Path.home() / ".gemini" / "antigravity" / "brain" / "03dc701e-63f2-47c5-a03a-84712fac3580" / "adversarial_candidates.md"
if not candidates_path.exists():
    candidates_path = Path("adversarial_candidates.md")

if candidates_path.exists():
    with open(candidates_path, "r", encoding="utf-8") as f:
        text = f.read()
else:
    text = ""

uids = re.findall(r"\*\*UID:\*\*\s*(.+)", text)
uids = [u.strip() for u in uids if u.strip()]

if uids:
    assert len(uids) == 23, f"Expected exactly 23 UIDs, but found {len(uids)}"

root_dir = Path(__file__).resolve().parent.parent
output_path = root_dir / "data" / "processed" / "adversarial_holdout_uids.txt"

if uids:
    with open(output_path, "w", encoding="utf-8") as f:
        for uid in uids:
            f.write(f"{uid}\n")
    print(f"Successfully extracted and wrote {len(uids)} UIDs to adversarial_holdout_uids.txt")

