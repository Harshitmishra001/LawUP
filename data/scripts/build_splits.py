"""Create contract-level train / val / test splits from CUAD.

Loads the CUAD QA dataset saved by ``download_cuad.py``, applies the
taxonomy mapping from ``build_taxonomy.py``, performs contract-level
splitting, and writes JSONL splits plus summary statistics.

Usage:
    python -m data.scripts.build_splits [--raw-dir data/raw] \
                                        [--processed-dir data/processed] \
                                        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import datasets
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORY_REGEX: re.Pattern[str] = re.compile(r"""related to [\"']([^\"']+)[\"']""")
RANDOM_SEED: int = 42
TRAIN_FRAC: float = 0.85  # of CUAD-train contracts → 85 % train, 15 % val

# Singleton buckets to call out in stats
SINGLETON_BUCKETS: dict[str, str] = {
    "Renewal Term": "Auto-Renewal Trap",
    "Non-Disparagement": "Reputational Restriction",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_category(question: str) -> str | None:
    """Extract the CUAD category name from the question string."""
    m = CATEGORY_REGEX.search(question)
    return m.group(1) if m else None


def _example_to_record(
    example: dict[str, Any],
    cuad_to_lawup: dict[str, str],
) -> dict[str, Any] | None:
    """Convert a single HF example dict to our flat record dict.

    Returns ``None`` if the category cannot be parsed or is unknown.
    """
    cuad_category = _parse_category(example["question"])
    if cuad_category is None:
        logger.warning("Could not parse category from question: %s", example["question"][:120])
        return None

    # Case-insensitive lookup to handle casing differences between CSV and JSON
    lawup_risk_category = cuad_to_lawup.get(cuad_category)
    if lawup_risk_category is None:
        # Fallback: try case-insensitive
        lower_map = {k.lower(): v for k, v in cuad_to_lawup.items()}
        lawup_risk_category = lower_map.get(cuad_category.lower())
    if lawup_risk_category is None:
        logger.warning("Unknown CUAD category '%s' — skipping", cuad_category)
        return None

    answers: dict[str, Any] = example["answers"]
    answer_texts: list[str] = answers.get("text", [])
    answer_starts: list[int] = answers.get("answer_start", [])

    has_clause = len(answer_texts) > 0 and answer_texts[0] != ""

    return {
        "contract_name": example["title"],
        "cuad_category": cuad_category,
        "lawup_risk_category": lawup_risk_category,
        "has_clause": has_clause,
        "clause_text": answer_texts[0] if has_clause else None,
        "clause_start": answer_starts[0] if has_clause else None,
        "context": example["context"],
    }


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _compute_split_stats(
    split_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-split statistics."""
    contracts = {r["contract_name"] for r in records}
    cuad_pos: Counter[str] = Counter()
    cuad_neg: Counter[str] = Counter()
    lawup_pos: Counter[str] = Counter()
    lawup_neg: Counter[str] = Counter()

    for r in records:
        if r["has_clause"]:
            cuad_pos[r["cuad_category"]] += 1
            lawup_pos[r["lawup_risk_category"]] += 1
        else:
            cuad_neg[r["cuad_category"]] += 1
            lawup_neg[r["lawup_risk_category"]] += 1

    return {
        "split": split_name,
        "num_contracts": len(contracts),
        "num_examples": len(records),
        "cuad_category_positive": dict(sorted(cuad_pos.items())),
        "cuad_category_negative": dict(sorted(cuad_neg.items())),
        "lawup_risk_category_positive": dict(sorted(lawup_pos.items())),
        "lawup_risk_category_negative": dict(sorted(lawup_neg.items())),
    }


def _print_summary(all_stats: list[dict[str, Any]]) -> None:
    """Print a formatted summary table to the log."""
    logger.info("=" * 90)
    logger.info("SPLIT STATISTICS SUMMARY")
    logger.info("=" * 90)
    header = f"{'Split':<10} {'Contracts':>10} {'Examples':>10} {'Positives':>10} {'Negatives':>10}"
    logger.info(header)
    logger.info("-" * 90)
    for s in all_stats:
        n_pos = sum(s["cuad_category_positive"].values())
        n_neg = sum(s["cuad_category_negative"].values())
        logger.info(
            "%-10s %10d %10d %10d %10d",
            s["split"],
            s["num_contracts"],
            s["num_examples"],
            n_pos,
            n_neg,
        )
    logger.info("-" * 90)

    # Singleton bucket call-out
    logger.info("")
    logger.info("SINGLETON BUCKET COUNTS:")
    for cuad_cat, lawup_cat in SINGLETON_BUCKETS.items():
        logger.info("  %s → %s", cuad_cat, lawup_cat)
        for s in all_stats:
            pos = s["cuad_category_positive"].get(cuad_cat, 0)
            neg = s["cuad_category_negative"].get(cuad_cat, 0)
            logger.info(
                "    %s: pos=%d, neg=%d, total=%d",
                s["split"], pos, neg, pos + neg,
            )
    logger.info("=" * 90)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_splits(
    raw_dir: Path,
    processed_dir: Path,
    seed: int = RANDOM_SEED,
) -> None:
    """Load CUAD, split by contract, and write JSONL files + stats."""
    # 1. Load dataset -------------------------------------------------------
    cuad_path = raw_dir / "cuad_qa"
    if not cuad_path.exists():
        logger.error(
            "CUAD dataset not found at %s. Run download_cuad.py first.",
            cuad_path,
        )
        sys.exit(1)

    logger.info("Loading dataset from %s …", cuad_path)
    ds = datasets.load_from_disk(str(cuad_path))

    # 2. Load taxonomy mapping -----------------------------------------------
    taxonomy_path = processed_dir / "taxonomy_mapping.json"
    if not taxonomy_path.exists():
        logger.error(
            "taxonomy_mapping.json not found at %s. Run build_taxonomy.py first.",
            taxonomy_path,
        )
        sys.exit(1)

    with taxonomy_path.open("r", encoding="utf-8") as fh:
        taxonomy = json.load(fh)
    cuad_to_lawup: dict[str, str] = taxonomy["cuad_to_lawup"]

    # 3. Convert all examples ------------------------------------------------
    hf_train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for example in ds["train"]:
        rec = _example_to_record(example, cuad_to_lawup)
        if rec is not None:
            hf_train_records.append(rec)

    for example in ds["test"]:
        rec = _example_to_record(example, cuad_to_lawup)
        if rec is not None:
            test_records.append(rec)

    logger.info(
        "Parsed %d train + %d test records from CUAD.",
        len(hf_train_records),
        len(test_records),
    )

    # 4. Contract-level train/val split --------------------------------------
    train_contracts = sorted({r["contract_name"] for r in hf_train_records})
    test_contracts = {r["contract_name"] for r in test_records}

    # Group array for GroupShuffleSplit — one entry per record
    groups = [r["contract_name"] for r in hf_train_records]

    gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - TRAIN_FRAC, random_state=seed)
    train_idx, val_idx = next(gss.split(hf_train_records, groups=groups))

    train_records = [hf_train_records[i] for i in train_idx]
    val_records = [hf_train_records[i] for i in val_idx]

    # 5. Verify no contract leakage -----------------------------------------
    train_contract_set = {r["contract_name"] for r in train_records}
    val_contract_set = {r["contract_name"] for r in val_records}

    assert train_contract_set.isdisjoint(val_contract_set), (
        "Contract leakage between train and val: "
        f"{train_contract_set & val_contract_set}"
    )
    assert train_contract_set.isdisjoint(test_contracts), (
        "Contract leakage between train and test: "
        f"{train_contract_set & test_contracts}"
    )
    assert val_contract_set.isdisjoint(test_contracts), (
        "Contract leakage between val and test: "
        f"{val_contract_set & test_contracts}"
    )
    logger.info("✓ No contract appears in multiple splits.")

    # 6. Write JSONL --------------------------------------------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_records, processed_dir / "train.jsonl")
    _write_jsonl(val_records, processed_dir / "val.jsonl")
    _write_jsonl(test_records, processed_dir / "test.jsonl")

    # 7. Stats ---------------------------------------------------------------
    train_stats = _compute_split_stats("train", train_records)
    val_stats = _compute_split_stats("val", val_records)
    test_stats = _compute_split_stats("test", test_records)
    all_stats = [train_stats, val_stats, test_stats]

    # Add singleton bucket info at top level
    singleton_counts: dict[str, dict[str, dict[str, int]]] = {}
    for cuad_cat, lawup_cat in SINGLETON_BUCKETS.items():
        singleton_counts[cuad_cat] = {}
        for s in all_stats:
            pos = s["cuad_category_positive"].get(cuad_cat, 0)
            neg = s["cuad_category_negative"].get(cuad_cat, 0)
            singleton_counts[cuad_cat][s["split"]] = {
                "positive": pos,
                "negative": neg,
            }

    stats_payload: dict[str, Any] = {
        "splits": all_stats,
        "singleton_bucket_counts": singleton_counts,
    }

    stats_path = processed_dir / "split_stats.json"
    stats_path.write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Split stats saved to %s", stats_path)

    # 8. Print summary -------------------------------------------------------
    _print_summary(all_stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create contract-level train/val/test splits from CUAD.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CUAD data (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed outputs (default: data/processed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for train/val splitting (default: {RANDOM_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry-point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = parse_args(argv)
    try:
        build_splits(args.raw_dir, args.processed_dir, seed=args.seed)
    except Exception:
        logger.exception("Split building failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
