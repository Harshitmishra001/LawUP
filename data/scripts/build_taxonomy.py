"""Validate CUAD categories and produce the canonical taxonomy mapping JSON.

Reads category_descriptions.csv from data/raw/, validates against the
hard-coded TAXONOMY_MAPPING, and writes the structured taxonomy JSON to
data/processed/taxonomy_mapping.json.

Usage:
    python -m data.scripts.build_taxonomy [--raw-dir data/raw] [--output-dir data/processed]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical taxonomy mapping — every CUAD category → LawUP risk bucket
# ---------------------------------------------------------------------------
# Keys match real CUAD SQuAD JSON casing (Title Case prepositions)
TAXONOMY_MAPPING: dict[str, str] = {
    # Termination Risk
    "Termination For Convenience": "Termination Risk",
    "Notice Period To Terminate Renewal": "Termination Risk",
    "Post-Termination Services": "Termination Risk",
    # Auto-Renewal Trap
    "Renewal Term": "Auto-Renewal Trap",
    # Non-Compete / Exclusivity Overreach
    "Non-Compete": "Non-Compete / Exclusivity Overreach",
    "Exclusivity": "Non-Compete / Exclusivity Overreach",
    "Competitive Restriction Exception": "Non-Compete / Exclusivity Overreach",
    "No-Solicit Of Customers": "Non-Compete / Exclusivity Overreach",
    "No-Solicit Of Employees": "Non-Compete / Exclusivity Overreach",
    # IP Ownership Risk
    "Ip Ownership Assignment": "IP Ownership Risk",
    "License Grant": "IP Ownership Risk",
    "Affiliate License-Licensee": "IP Ownership Risk",
    "Affiliate License-Licensor": "IP Ownership Risk",
    "Joint Ip Ownership": "IP Ownership Risk",
    "Irrevocable Or Perpetual License": "IP Ownership Risk",
    "Non-Transferable License": "IP Ownership Risk",
    "Unlimited/All-You-Can-Eat-License": "IP Ownership Risk",
    "Source Code Escrow": "IP Ownership Risk",
    # Liability Exposure
    "Cap On Liability": "Liability Exposure",
    "Uncapped Liability": "Liability Exposure",
    "Liquidated Damages": "Liability Exposure",
    "Warranty Duration": "Liability Exposure",
    # Assignment/Control Risk
    "Anti-Assignment": "Assignment/Control Risk",
    "Change Of Control": "Assignment/Control Risk",
    "Third Party Beneficiary": "Assignment/Control Risk",
    # Dispute / Governing Law
    "Governing Law": "Dispute / Governing Law",
    "Covenant Not To Sue": "Dispute / Governing Law",
    # Audit/Insurance Burden
    "Audit Rights": "Audit/Insurance Burden",
    "Insurance": "Audit/Insurance Burden",
    # Reputational Restriction
    "Non-Disparagement": "Reputational Restriction",
    # Preferential/Exclusivity Terms
    "Most Favored Nation": "Preferential/Exclusivity Terms",
    "Rofr/Rofo/Rofn": "Preferential/Exclusivity Terms",
    # Commercial Commitment Constraints
    "Price Restrictions": "Commercial Commitment Constraints",
    "Revenue/Profit Sharing": "Commercial Commitment Constraints",
    "Volume Restriction": "Commercial Commitment Constraints",
    "Minimum Commitment": "Commercial Commitment Constraints",
    # Metadata (extraction, not risk)
    "Document Name": "_metadata",
    "Parties": "_metadata",
    "Agreement Date": "_metadata",
    "Effective Date": "_metadata",
    "Expiration Date": "_metadata",
}

# Build a case-insensitive lookup for matching across sources
_TAXONOMY_LOWER: dict[str, str] = {k.lower(): v for k, v in TAXONOMY_MAPPING.items()}

def lookup_taxonomy(category: str) -> str | None:
    """Case-insensitive taxonomy lookup."""
    return _TAXONOMY_LOWER.get(category.lower())

# Phase 1 coverage gaps — categories with no CUAD training data
PHASE_1_COVERAGE_GAPS: list[dict[str, str]] = [
    {
        "category": "Confidentiality Scope",
        "reason": (
            "No CUAD annotation category exists. Phase 1 placeholder only."
        ),
    },
    {
        "category": "Indemnification",
        "reason": (
            "No CUAD annotation category exists. Listed under Liability "
            "Exposure conceptually but not trainable in Phase 1."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invert_mapping(mapping: dict[str, str]) -> dict[str, dict[str, object]]:
    """Build the ``risk_categories`` section from the flat mapping.

    Returns a dict keyed by LawUP risk category, each with:
      - cuad_categories: sorted list of CUAD category names
      - is_risk_flag: False only for ``_metadata``
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for cuad_cat, lawup_cat in mapping.items():
        grouped[lawup_cat].append(cuad_cat)

    risk_categories: dict[str, dict[str, object]] = {}
    for lawup_cat in sorted(grouped):
        risk_categories[lawup_cat] = {
            "cuad_categories": sorted(grouped[lawup_cat]),
            "is_risk_flag": lawup_cat != "_metadata",
        }
    return risk_categories


def _print_coverage_table(
    risk_categories: dict[str, dict[str, object]],
) -> None:
    """Print a human-readable coverage report table to the log."""
    header = f"{'LawUP Risk Category':<40} {'CUAD Sub-Categories'}"
    logger.info("=" * 100)
    logger.info("TAXONOMY COVERAGE REPORT")
    logger.info("=" * 100)
    logger.info(header)
    logger.info("-" * 100)
    for lawup_cat, info in sorted(risk_categories.items()):
        cuad_cats = info["cuad_categories"]
        # First sub-category on the same line
        logger.info(
            "%-40s %s", lawup_cat, cuad_cats[0] if cuad_cats else "(none)"
        )
        for sub in cuad_cats[1:]:
            logger.info("%-40s %s", "", sub)
    logger.info("=" * 100)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_taxonomy(raw_dir: Path, output_dir: Path) -> None:
    """Validate categories and write taxonomy_mapping.json."""
    # 1. Load category_descriptions.csv ------------------------------------
    csv_path = raw_dir / "category_descriptions.csv"
    if not csv_path.exists():
        logger.error(
            "category_descriptions.csv not found at %s. "
            "Run download_cuad.py first.",
            csv_path,
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # The actual CSV header is "Category (incl. context and answer)" and the values start with "Category: "
    csv_col = "Category (incl. context and answer)"
    if csv_col not in df.columns and "Category" in df.columns:
        csv_col = "Category" # fallback if we ever hit the mock
    raw_categories = df[csv_col].tolist()
    csv_categories: set[str] = {c.replace("Category: ", "").strip() for c in raw_categories}
    logger.info(
        "Loaded %d categories from %s", len(csv_categories), csv_path
    )

    # 2. Validate (case-insensitive) -----------------------------------------
    csv_lower = {c.lower() for c in csv_categories}
    mapping_lower = {k.lower() for k in TAXONOMY_MAPPING.keys()}
    missing_from_mapping = csv_lower - mapping_lower
    extra_in_mapping = mapping_lower - csv_lower

    if missing_from_mapping or extra_in_mapping:
        if missing_from_mapping:
            logger.error(
                "Categories in CSV but NOT in TAXONOMY_MAPPING (case-insensitive): %s",
                sorted(missing_from_mapping),
            )
        if extra_in_mapping:
            logger.error(
                "Categories in TAXONOMY_MAPPING but NOT in CSV (case-insensitive): %s",
                sorted(extra_in_mapping),
            )
        sys.exit(1)

    logger.info("✓ All %d CSV categories match TAXONOMY_MAPPING (case-insensitive).", len(csv_categories))

    # 3. Build output JSON -------------------------------------------------
    risk_categories = _invert_mapping(TAXONOMY_MAPPING)

    taxonomy: dict[str, object] = {
        "version": "1.0",
        "confirmed_date": "2026-07-11",
        "source": "CUAD v1 category_descriptions.csv",
        "phase_1_coverage_gaps": PHASE_1_COVERAGE_GAPS,
        "risk_categories": risk_categories,
        "cuad_to_lawup": dict(sorted(TAXONOMY_MAPPING.items())),
    }

    # 4. Save --------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "taxonomy_mapping.json"
    out_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Taxonomy mapping saved to %s", out_path)

    # 5. Coverage report ---------------------------------------------------
    _print_coverage_table(risk_categories)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate CUAD categories and produce the canonical "
            "taxonomy mapping JSON."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CUAD downloads (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed outputs (default: data/processed).",
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
        build_taxonomy(args.raw_dir, args.output_dir)
    except Exception:
        logger.exception("Taxonomy build failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
