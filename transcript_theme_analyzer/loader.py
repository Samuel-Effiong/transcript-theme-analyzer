"""Detects a transcript input file's format and extracts plain text from it.

Supported formats: ``.txt`` (read as-is) and ``.docx`` (paragraph text via
python-docx). Rich formatting (bold/italic/underline, fonts, colors) is
intentionally dropped on extraction — nothing downstream (chunker.py's
timestamp/speaker regexes, the analysis prompts) reasons about typography,
only about the text content. Paragraph breaks ARE preserved (one paragraph
per line): chunk_transcript splits on "\\n" boundaries and SPEAKER_RE only
matches at the start of a line, so losing paragraph structure would break
both.
"""
from __future__ import annotations

import os
import zipfile

from docx import Document

SUPPORTED_EXTENSIONS = {".txt", ".docx"}


def detect_format(path: str) -> str:
    """Returns the lowercased extension (".txt" or ".docx"), or raises
    ValueError with a clear message if the file isn't a supported/valid
    transcript format."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported transcript file type {ext!r} for {path!r}; "
            f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if ext == ".docx" and not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path!r} has a .docx extension but isn't a valid .docx file (not a zip container)"
        )
    return ext


def _extract_docx_text(path: str) -> str:
    document = Document(path)
    lines = [p.text for p in document.paragraphs]

    # Collapse runs of 2+ blank paragraphs (common docx spacing) down to a
    # single blank line, so the chunker doesn't see artificially huge gaps.
    collapsed: list[str] = []
    for line in lines:
        if line == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(line)

    return "\n".join(collapsed).strip()


def load_transcript_text(path: str) -> str:
    ext = detect_format(path)
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return _extract_docx_text(path)
