"""Download/extract all CUAD data sources to data/raw/.

Supports:
  1. Local ZIP extraction (if CUAD_v1.zip already exists in output_dir)
  2. Zenodo download with retry+backoff
  3. HuggingFace parquet mirror fallback

Usage:
    python -m data.scripts.download_cuad [--output-dir data/raw]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
import zipfile
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_URL: str = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
HF_DATASET_NAME: str = "whpthomas/cuad-parquet"

GITHUB_RAW_BASE: str = (
    "https://raw.githubusercontent.com/TheAtticusProject/cuad/master"
)
CATEGORY_DESC_URL: str = f"{GITHUB_RAW_BASE}/category_descriptions.csv"

EXPECTED_CATEGORY_COUNT: int = 41

CATEGORY_REGEX: re.Pattern[str] = re.compile(
    r"related to \"([^\"]+)\""
)

MAX_RETRIES: int = 3
BACKOFF_BASE: float = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_with_retry(url: str, timeout: int = 300) -> requests.Response:
    """Download with exponential backoff retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Attempt %d/%d: %s", attempt, MAX_RETRIES, url)
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE ** attempt
            logger.warning("Attempt %d failed (%s). Retrying in %.0fs...", attempt, e, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def _download_file(url: str, dest: Path) -> None:
    """Download a file from *url* and write it to *dest*."""
    logger.info("Downloading %s -> %s", url, dest)
    resp = _download_with_retry(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    logger.info("Saved %s (%d bytes)", dest, len(resp.content))


def _extract_categories(dataset_dict: DatasetDict) -> set[str]:
    """Return the set of unique category names parsed from questions."""
    categories: set[str] = set()
    for split_name in dataset_dict:
        for example in dataset_dict[split_name]:
            match = CATEGORY_REGEX.search(example["question"])
            if match:
                categories.add(match.group(1))
    return categories


def _squad_json_to_datasetdict(cuad_data: dict) -> DatasetDict:
    """Convert SQuAD 2.0 JSON to flat HF DatasetDict with train/test splits.
    
    Uses the CUAD original 80/20 split by contract index.
    """
    articles = cuad_data.get("data", [])
    logger.info("Found %d articles in SQuAD JSON", len(articles))

    split_idx = int(len(articles) * 0.8)  # 408 train, 102 test

    train_records: dict[str, list] = {"id": [], "title": [], "context": [], "question": [], "answers": []}
    test_records: dict[str, list] = {"id": [], "title": [], "context": [], "question": [], "answers": []}

    for idx, article in enumerate(articles):
        target = train_records if idx < split_idx else test_records
        title = article.get("title", f"contract_{idx}")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = qa.get("id", f"{title}_{len(target['id'])}")
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                ans_texts = [a["text"] for a in answers]
                ans_starts = [a["answer_start"] for a in answers]

                target["id"].append(qid)
                target["title"].append(title)
                target["context"].append(context)
                target["question"].append(question)
                target["answers"].append({"text": ans_texts, "answer_start": ans_starts})

    ds = DatasetDict({
        "train": Dataset.from_dict(train_records),
        "test": Dataset.from_dict(test_records),
    })
    logger.info("Converted to DatasetDict: train=%d, test=%d", len(ds["train"]), len(ds["test"]))
    return ds


def _load_from_local_zip(zip_path: Path, output_dir: Path) -> DatasetDict | None:
    """Extract CUAD data from a local ZIP file."""
    try:
        logger.info("Extracting from local ZIP: %s", zip_path)
        zf = zipfile.ZipFile(zip_path)

        # Find SQuAD JSON
        json_candidates = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_candidates:
            logger.error("No JSON file found in ZIP.")
            return None

        json_path = json_candidates[0]
        logger.info("Loading SQuAD JSON: %s", json_path)
        with zf.open(json_path) as f:
            cuad_data = json.load(f)

        # Extract master_clauses.csv if present
        csv_candidates = [n for n in zf.namelist() if "master_clauses" in n.lower()]
        if csv_candidates:
            master_dest = output_dir / "master_clauses.csv"
            with zf.open(csv_candidates[0]) as src:
                master_dest.write_bytes(src.read())
            logger.info("Extracted master_clauses.csv")

        return _squad_json_to_datasetdict(cuad_data)

    except Exception as e:
        logger.warning("Local ZIP extraction failed: %s", e)
        return None


def _load_from_zenodo(output_dir: Path) -> DatasetDict | None:
    """Download CUAD_v1.zip from Zenodo."""
    try:
        logger.info("Downloading CUAD from Zenodo: %s", ZENODO_URL)
        resp = _download_with_retry(ZENODO_URL, timeout=600)
        zip_path = output_dir / "CUAD_v1.zip"
        zip_path.write_bytes(resp.content)
        logger.info("Saved ZIP (%d bytes)", len(resp.content))
        return _load_from_local_zip(zip_path, output_dir)
    except Exception as e:
        logger.warning("Zenodo download failed: %s", e)
        return None


def _load_from_hf() -> DatasetDict | None:
    """Fallback: load from HuggingFace."""
    try:
        from datasets import load_dataset
        logger.info("Fallback: Loading from HuggingFace '%s'...", HF_DATASET_NAME)
        ds: DatasetDict = load_dataset(HF_DATASET_NAME)  # type: ignore[assignment]
        return ds
    except Exception as e:
        logger.warning("HuggingFace fallback also failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def download_cuad(output_dir: Path) -> None:
    """Download all CUAD data sources and print a summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try sources in order: local ZIP → Zenodo → HuggingFace
    zip_path = output_dir / "CUAD_v1.zip"
    ds = None
    if zip_path.exists():
        logger.info("Found local ZIP at %s — extracting...", zip_path)
        ds = _load_from_local_zip(zip_path, output_dir)

    if ds is None:
        ds = _load_from_zenodo(output_dir)
    if ds is None:
        ds = _load_from_hf()
    if ds is None:
        logger.error("FATAL: Could not load CUAD from any source.")
        sys.exit(1)

    cuad_dir = output_dir / "cuad_qa"
    ds.save_to_disk(str(cuad_dir))
    logger.info("Saved dataset to %s", cuad_dir)

    # category_descriptions.csv
    cat_desc_path = output_dir / "category_descriptions.csv"
    if not cat_desc_path.exists():
        _download_file(CATEGORY_DESC_URL, cat_desc_path)
    else:
        logger.info("category_descriptions.csv already exists, skipping download.")

    # Summary
    for split_name in sorted(ds.keys()):
        logger.info("Split '%s': %d examples", split_name, len(ds[split_name]))

    all_titles: set[str] = set()
    for split_name in ds:
        all_titles.update(ds[split_name]["title"])
    logger.info("Unique contracts (by title): %d", len(all_titles))

    categories = _extract_categories(ds)
    logger.info("Extracted %d unique categories:", len(categories))
    for cat in sorted(categories):
        logger.info("  • %s", cat)

    # Validate
    assert len(categories) == EXPECTED_CATEGORY_COUNT, (
        f"Expected {EXPECTED_CATEGORY_COUNT} categories, found {len(categories)}"
    )
    logger.info(
        "✓ Validation passed — exactly %d categories found.", EXPECTED_CATEGORY_COUNT
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CUAD data sources to data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save raw data (default: data/raw).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = parse_args(argv)
    try:
        download_cuad(args.output_dir)
    except Exception:
        logger.exception("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
