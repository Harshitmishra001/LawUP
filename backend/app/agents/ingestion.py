"""Ingestion Agent — parses raw contracts (Text, Markdown, PDF, DOCX) into structured clause units."""

import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

# Patterns that signal a clause or section boundary in commercial agreements
_SECTION_HEADING_PATTERN = re.compile(
    r"""
    ^\s*                        # optional leading whitespace
    (?:
        \d+(?:\.\d+)*\.?        # numbered: 1, 1.2, 1.2.3, 1.
        |[A-Z]{1,3}\.           # lettered: A. B. XII.
        |\(\s*[0-9a-zA-Z]+\s*\) # parenthesized: (1), (a), (i)
        |(?:ARTICLE|SECTION|CLAUSE|PART|EXHIBIT|SCHEDULE)\s+[0-9A-Za-z\.\:\-]+ # keyword headings
    )
    \s+                         # whitespace after header
    [A-Za-z0-9]                 # heading or text starts
    """,
    re.VERBOSE | re.MULTILINE,
)

_ALL_CAPS_HEADING_PATTERN = re.compile(
    r"^\s*([A-Z0-9\s,\-\/\(\)]{4,60})\s*$",
    re.MULTILINE,
)

# Minimum word count for a text segment to be considered a substantive clause
_MIN_CLAUSE_WORDS = 8


@dataclass
class ClauseUnit:
    """Represents an extracted, structured clause unit within a contract."""

    index: int
    heading: str
    text: str
    word_count: int

    def to_dict(self):
        return asdict(self)


class IngestionAgent:
    """Ingestion Agent — parses raw contracts into discrete, structured clause units.

    Pipeline step: 1 (reference Section 5 of PLAN.md)

    Capabilities:
      - Plain-text and Markdown document parsing
      - PDF extraction via pypdf
      - Word Document (.docx) extraction via python-docx
      - In-memory bytes / file-like streams for direct API uploads
      - Multi-strategy clause segmentation with heading identification
      - Filtering of short non-substantive fragments (< 8 words)
    """

    def __init__(self, min_clause_words: int = _MIN_CLAUSE_WORDS):
        self.min_clause_words = min_clause_words

    def extract_text(
        self,
        input_data: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
    ) -> str:
        """Extract plain text from string, file path, bytes, or file-like stream.

        Supports .txt, .md, .pdf, and .docx.
        """
        if input_data is None:
            raise ValueError("input_data cannot be None.")

        # Case 1: Already a raw text string (and not a path on disk)
        if isinstance(input_data, str):
            path_candidate = Path(input_data)
            if not path_candidate.is_file():
                return input_data.strip()
            # If it is an existing file path string, treat as Path
            input_data = path_candidate

        # Case 2: Path object
        if isinstance(input_data, Path):
            filename = filename or input_data.name
            with open(input_data, "rb") as f:
                content_bytes = f.read()
            return self._extract_from_bytes(content_bytes, filename=filename)

        # Case 3: BinaryIO / BytesIO stream
        if hasattr(input_data, "read"):
            content_bytes = input_data.read()
            if isinstance(content_bytes, str):
                return content_bytes.strip()
            return self._extract_from_bytes(content_bytes, filename=filename)

        # Case 4: Raw bytes
        if isinstance(input_data, bytes):
            return self._extract_from_bytes(input_data, filename=filename)

        raise TypeError(f"Unsupported input type for extract_text: {type(input_data)}")

    def _extract_from_bytes(self, content: bytes, filename: Optional[str] = None) -> str:
        """Helper to extract text from raw file bytes based on extension or signature."""
        fname = (filename or "").lower()

        # PDF Detection (via filename or PDF magic bytes %PDF)
        if fname.endswith(".pdf") or content.startswith(b"%PDF"):
            try:
                import pypdf

                reader = pypdf.PdfReader(io.BytesIO(content))
                pages_text = [page.extract_text() or "" for page in reader.pages]
                extracted = "\n\n".join(p.strip() for p in pages_text if p.strip())
                if extracted.strip():
                    return extracted
            except Exception as e:
                # Fallback to pdfplumber/fitz if pypdf has parsing difficulty
                try:
                    import fitz  # PyMuPDF

                    doc = fitz.open(stream=content, filetype="pdf")
                    pages_text = [page.get_text() for page in doc]
                    return "\n\n".join(p.strip() for p in pages_text if p.strip())
                except Exception:
                    raise ValueError(f"Failed to extract text from PDF: {e}")

        # DOCX Detection (via filename or ZIP magic bytes PK\x03\x04)
        if fname.endswith(".docx") or (content.startswith(b"PK\x03\x04") and b"word/" in content[:2000]):
            try:
                import docx

                doc = docx.Document(io.BytesIO(content))
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                return "\n\n".join(paragraphs)
            except Exception as e:
                raise ValueError(f"Failed to extract text from DOCX: {e}")

        # Default fallback: Treat as UTF-8 / Latin-1 plain text
        try:
            return content.decode("utf-8").strip()
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="replace").strip()

    def ingest(
        self,
        input_data: Union[str, bytes, BinaryIO, Path],
        filename: Optional[str] = None,
    ) -> List[ClauseUnit]:
        """Parse raw input and return structured ClauseUnit instances.

        Args:
            input_data: Raw text, bytes, file path, or file stream.
            filename: Optional filename to identify file format (.pdf, .docx, .txt).

        Returns:
            List of ClauseUnit objects, each containing index, heading, text, and word count.
        """
        raw_text = self.extract_text(input_data, filename=filename)
        return self._segment_into_clause_units(raw_text)

    def run(self, contract_text: str) -> List[str]:
        """Backward-compatible entry point returning plain list of clause strings.

        Args:
            contract_text: Full raw text of the contract.

        Returns:
            List of clause text strings.
        """
        units = self._segment_into_clause_units(contract_text)
        return [unit.text for unit in units]

    def _segment_into_clause_units(self, text: str) -> List[ClauseUnit]:
        """Internal segmentation pipeline combining heading detection and fallback chunking."""
        cleaned = text.strip()
        if not cleaned:
            return []

        # Strategy 1: Explicit numbered/lettered section headings
        chunks = self._split_by_section_headings(cleaned)

        # Strategy 2: If < 2 clauses found, split by double newlines (paragraphs)
        if len(chunks) < 2:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned)]
            chunks = [p for p in paragraphs if p]

        # Strategy 3: Fallback — single clause
        if not chunks:
            chunks = [cleaned]

        # Convert raw text chunks into structured ClauseUnit instances
        clause_units: List[ClauseUnit] = []
        clause_idx = 1

        for chunk in chunks:
            words = chunk.split()
            if len(words) < self.min_clause_words:
                continue

            heading, body = self._extract_heading_and_body(chunk, default_idx=clause_idx)
            clause_units.append(
                ClauseUnit(
                    index=clause_idx,
                    heading=heading,
                    text=body,
                    word_count=len(body.split()),
                )
            )
            clause_idx += 1

        # If all chunks were filtered out due to length, return full text if non-empty
        if not clause_units and cleaned:
            clause_units.append(
                ClauseUnit(
                    index=1,
                    heading="General Agreement",
                    text=cleaned,
                    word_count=len(cleaned.split()),
                )
            )

        return clause_units

    def _split_by_section_headings(self, text: str) -> List[str]:
        """Split text at section heading boundaries while preserving order."""
        matches = list(_SECTION_HEADING_PATTERN.finditer(text))
        if not matches:
            return []

        segments = []
        # Any text before the first detected section (e.g., contract title, preamble)
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                segments.append(preamble)

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)

        return segments

    def _extract_heading_and_body(self, chunk: str, default_idx: int) -> tuple[str, str]:
        """Split a chunk into a clean heading and substantive body text."""
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            return f"Clause {default_idx}", chunk

        first_line = lines[0]

        # Check if first line matches standard section heading or is short (< 60 chars)
        is_heading = (
            bool(_SECTION_HEADING_PATTERN.match(first_line))
            or bool(_ALL_CAPS_HEADING_PATTERN.match(first_line))
            or (len(first_line) < 60 and len(lines) > 1 and not first_line.endswith("."))
        )

        if is_heading and len(lines) > 1:
            heading = first_line
            body = " ".join(lines[1:]).strip()
            # If body became empty, fallback to full chunk
            if not body:
                body = chunk
            return heading, body
        elif is_heading and len(lines) == 1:
            return first_line, first_line
        else:
            return f"Clause {default_idx}", " ".join(lines).strip()
