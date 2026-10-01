"""Ingestion Agent — parses contract text and extracts clause-level text units."""

import re
from typing import List


# Patterns that signal a new clause boundary in a raw contract
_SECTION_HEADING_PATTERN = re.compile(
    r"""
    ^\s*                    # optional leading whitespace
    (?:
        \d+(?:\.\d+)*\.?    # numbered: 1, 1.2, 1.2.3, 1.
        |[A-Z]{1,3}\.       # lettered: A. B. XII.
        |(?:ARTICLE|SECTION|CLAUSE|PART|EXHIBIT|SCHEDULE)\s+\S+  # keyword headings
    )
    \s+                     # at least one space after the header
    [A-Z]                   # heading starts with a capital letter
    """,
    re.VERBOSE | re.MULTILINE,
)

# Minimum word count for a text block to be considered a real clause
_MIN_CLAUSE_WORDS = 8


class IngestionAgent:
    """Ingestion Agent — parses raw contract text into discrete clause-level units.

    Pipeline step: 1 (reference Section 5 of PLAN.md)

    This agent handles the baseline case of plain-text contract input.
    PDF and DOCX parsing is handled upstream by the API layer (future work).

    Splitting strategy (in priority order):
      1. Explicit numbered or lettered section headings (e.g. "1.", "1.2", "A.", "SECTION 4")
      2. Double newlines (paragraph breaks)
      3. Fallback: single sentences if no structure is detected

    Each output clause is stripped of leading/trailing whitespace and must
    contain at least 8 words to be returned (filters out headings-only chunks).
    """

    def __init__(self, min_clause_words: int = _MIN_CLAUSE_WORDS):
        self.min_clause_words = min_clause_words

    def run(self, contract_text: str) -> List[str]:
        """Split a raw contract string into a list of clause text units.

        Args:
            contract_text: The full raw text of the contract or agreement.

        Returns:
            List of clause strings, each with at least min_clause_words words.
            Returns an empty list if the input is blank.

        Raises:
            ValueError: If contract_text is None.
        """
        if contract_text is None:
            raise ValueError("contract_text cannot be None.")

        text = contract_text.strip()
        if not text:
            return []

        # --- Strategy 1: Try splitting on numbered/lettered section headings ---
        clauses = self._split_by_section_headings(text)
        if len(clauses) >= 2:
            return self._filter_clauses(clauses)

        # --- Strategy 2: Split on double newlines (paragraph breaks) ---
        clauses = [p.strip() for p in re.split(r"\n\s*\n", text)]
        clauses = self._filter_clauses(clauses)
        if len(clauses) >= 2:
            return clauses

        # --- Strategy 3: Fallback — treat the whole text as one clause ---
        # (The caller can further split or pass it as-is to the pipeline)
        filtered = self._filter_clauses([text])
        return filtered if filtered else [text]

    def _split_by_section_headings(self, text: str) -> List[str]:
        """Split text at section heading boundaries."""
        matches = list(_SECTION_HEADING_PATTERN.finditer(text))
        if not matches:
            return []

        segments = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            segments.append(text[start:end].strip())

        # Prepend any preamble text that comes before the first heading
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                segments.insert(0, preamble)

        return segments

    def _filter_clauses(self, clauses: List[str]) -> List[str]:
        """Remove empty or too-short segments."""
        return [
            c.strip()
            for c in clauses
            if c.strip() and len(c.strip().split()) >= self.min_clause_words
        ]


if __name__ == "__main__":
    sample_contract = """
    CONSULTING AGREEMENT

    This Consulting Agreement is entered into as of January 1, 2024.

    1. SERVICES
    Consultant shall provide software development services as reasonably requested by Client.

    2. COMPENSATION
    Client shall pay Consultant a monthly fee of $5,000 payable within 30 days of invoice.

    3. CONFIDENTIALITY
    Consultant agrees to keep all Client information strictly confidential and shall not
    disclose any proprietary information to third parties without prior written consent.

    4. TERMINATION
    Either party may terminate this Agreement upon 30 days written notice to the other party.
    Upon termination, all outstanding invoices shall become immediately due and payable.
    """

    agent = IngestionAgent()
    clauses = agent.run(sample_contract)

    print(f"Extracted {len(clauses)} clauses:\n")
    for i, clause in enumerate(clauses, 1):
        print(f"[{i}] {clause[:120]}...")
        print()
