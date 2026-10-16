import json
import re
from pathlib import Path

# Paths to data files
DATA_DIR = Path("data/processed")
SIMPLIFICATION_FILE = DATA_DIR / "simplification_pairs.jsonl"
FIXTURE_FILE = Path("data/scripts/tests/fixtures/sample_simplification_pairs.jsonl")
HOLDOUT_FILE = DATA_DIR / "adversarial_holdout_uids.txt"


def get_target_file():
    """Returns the full simplification file if present, else falls back to the committed test fixture."""
    if SIMPLIFICATION_FILE.exists():
        return SIMPLIFICATION_FILE, True
    if FIXTURE_FILE.exists():
        return FIXTURE_FILE, False
    return None, False


def test_simplification_pairs_count():
    """Assert the dataset contains the expected number of unique clauses."""
    target_file, is_full = get_target_file()
    if target_file is None:
        import pytest
        pytest.skip("Neither full simplification file nor fixture found.")

    unique_uids = set()
    with target_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "_uid" in record:
                unique_uids.add(record["_uid"])

    if is_full:
        assert len(unique_uids) in [3419, 3420], f"Expected 3420 unique clauses, but found {len(unique_uids)}"
    else:
        assert len(unique_uids) >= 23, f"Expected at least 23 unique clauses in fixture, found {len(unique_uids)}"


def test_zero_redaction_placeholders():
    """Assert zero clauses contain redaction placeholders in the cleaned output."""
    target_file, _ = get_target_file()
    if target_file is None:
        import pytest
        pytest.skip("Neither full simplification file nor fixture found.")

    redaction_regex = re.compile(r'\[\s*\*{1,}\s*\]|\[\s*REDACTED\s*\]|\[\s*OMITTED\s*\]|\[\s*Date\s*\]', re.IGNORECASE)

    with target_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            clause_text = record.get("original_clause", "")
            if redaction_regex.search(clause_text):
                assert False, f"Found redaction placeholder in line {i+1}: {clause_text}"


def test_zero_label_hallucinations():
    """Assert zero clauses are 'label hallucinations' (clause text < 5 words)."""
    target_file, _ = get_target_file()
    if target_file is None:
        import pytest
        pytest.skip("Neither full simplification file nor fixture found.")

    with target_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            clause_text = record.get("original_clause", "")
            words = clause_text.split()
            if len(words) < 5:
                assert False, f"Found short clause (potential label hallucination) in line {i+1}: {clause_text}"


def test_adversarial_holdout_extraction():
    """
    Assert the 23-clause adversarial holdout UID list contains EXACTLY 23 entries,
    and that extracting rows matching those UIDs via strict == returns exactly 23 rows.
    """
    if not HOLDOUT_FILE.exists():
        import pytest
        pytest.skip(f"{HOLDOUT_FILE} not found - skipping test.")

    target_file, _ = get_target_file()
    if target_file is None:
        import pytest
        pytest.skip("Neither full simplification file nor fixture found.")

    # Read UIDs
    with HOLDOUT_FILE.open("r", encoding="utf-8") as f:
        holdout_uids = [line.strip() for line in f if line.strip()]

    assert len(holdout_uids) == 23, f"Expected exactly 23 UIDs in {HOLDOUT_FILE.name}, found {len(holdout_uids)}"

    # Extract matches via strict ==
    matched_uids = set()
    with target_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            uid = record.get("_uid")
            if uid in holdout_uids:
                matched_uids.add(uid)

    assert len(matched_uids) == 23, f"Expected exactly 23 matching unique UIDs, but extracted {len(matched_uids)} using strict equality."
