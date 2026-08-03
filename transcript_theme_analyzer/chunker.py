"""Splits a transcript into overlapping chunks with positional metadata.

Token counts are approximated with a chars-per-token heuristic rather than a
real tokenizer, since this pipeline is model-agnostic (OpenAI tokenizers don't
apply to every provider/model it might run against anyway). The heuristic is
conservative (biased toward smaller chunks) so real token counts stay under
the configured budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 3.5

# Matches common transcript timestamp styles: [00:12:34], (00:12:34), 00:12:34.
TIMESTAMP_RE = re.compile(r"[\[\(]?\b\d{1,2}:\d{2}(?::\d{2})?\b[\]\)]?")
# Matches "Speaker Name:" / "SPEAKER 1:" at the start of a line.
SPEAKER_RE = re.compile(
    r"^\s*(?:[\[\(]?\d{1,2}:\d{2}(?::\d{2})?[\]\)]?\s*)?([A-Za-z][A-Za-z0-9 ._'-]{0,40}):\s",
    re.MULTILINE,
)


@dataclass
class Chunk:
    index: int
    char_start: int
    char_end: int
    text: str
    nearest_timestamp: str | None
    nearest_speaker: str | None


def _tokens_to_chars(n_tokens: int) -> int:
    return int(n_tokens * CHARS_PER_TOKEN)


def has_structure(transcript: str, sample_size: int = 20000) -> tuple[bool, bool]:
    """Detect whether the transcript carries timestamps and/or speaker labels."""
    sample = transcript[:sample_size]
    has_timestamps = bool(TIMESTAMP_RE.search(sample))
    has_speakers = bool(SPEAKER_RE.search(sample))
    return has_timestamps, has_speakers


def _nearest_match(pattern: re.Pattern, text: str, pos: int) -> str | None:
    """Find the match of `pattern` closest to (at or before) `pos`; else the first after it."""
    best_before = None
    for m in pattern.finditer(text[: pos + 1]):
        best_before = m
    if best_before is not None:
        return best_before.group(1) if pattern is SPEAKER_RE else best_before.group(0)
    after = pattern.search(text, pos)
    if after is not None:
        return after.group(1) if pattern is SPEAKER_RE else after.group(0)
    return None


def chunk_transcript(
    transcript: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split `transcript` into overlapping chunks.

    Splits on paragraph/line boundaries near the target size where possible,
    so a chunk doesn't cut mid-sentence any more than necessary.
    """
    chunk_chars = _tokens_to_chars(chunk_size_tokens)
    overlap_chars = _tokens_to_chars(overlap_tokens)
    if overlap_chars >= chunk_chars:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    has_ts, has_sp = has_structure(transcript)
    n = len(transcript)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # Prefer to break at a paragraph or line boundary within the last 20%
            search_from = max(start, end - int(chunk_chars * 0.2))
            boundary = transcript.rfind("\n", search_from, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        text = transcript[start:end]
        nearest_ts = _nearest_match(TIMESTAMP_RE, transcript, start) if has_ts else None
        nearest_sp = _nearest_match(SPEAKER_RE, transcript, start) if has_sp else None

        chunks.append(
            Chunk(
                index=index,
                char_start=start,
                char_end=end,
                text=text,
                nearest_timestamp=nearest_ts,
                nearest_speaker=nearest_sp,
            )
        )

        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)
        index += 1

    return chunks


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)
