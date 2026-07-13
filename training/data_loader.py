"""Data loading utilities for LawUP training.

Handles loading processed CUAD splits and formatting them
for multi-task LoRA training (classification + simplification).
"""


def load_splits(config: dict) -> dict:
    """Load train/val/test splits from processed JSONL files."""
    raise NotImplementedError("Implementation pending — see M2 milestone.")


def format_for_classification(examples: list) -> list:
    """Format examples for clause risk classification task."""
    raise NotImplementedError("Implementation pending — see M2 milestone.")


def format_for_simplification(examples: list) -> list:
    """Format examples for plain-language simplification task."""
    raise NotImplementedError("Implementation pending — see M2 milestone.")
