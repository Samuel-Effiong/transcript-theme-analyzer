"""Deterministically extracts a full passage from source text given a short
start/end boundary marker, instead of asking the model to reproduce the
passage verbatim -- cheaper in completion tokens, and avoids the model
truncating or paraphrasing a long quote.
"""
from __future__ import annotations

import re

DEFAULT_MAX_FALLBACK_SPAN = 6000


def _find_marker(text: str, marker: str, start_pos: int = 0) -> tuple[int, int] | None:
    """Locates `marker` in `text` starting from `start_pos`, tolerating
    whitespace (newlines vs. spaces) and case differences -- models don't
    always reproduce a marker phrase with byte-exact spacing. Searches the
    original text directly (not a normalized copy) so the returned offsets
    are correct without a position-remapping step. Returns (start, end)
    char offsets in `text`, or None if not found.
    """
    marker = marker.strip()
    if not marker:
        return None
    tokens = [re.escape(t) for t in marker.split()]
    if not tokens:
        return None
    pattern = r"\s+".join(tokens)
    match = re.search(pattern, text[start_pos:], re.IGNORECASE)
    if match is None:
        return None
    return start_pos + match.start(), start_pos + match.end()


def extract_passage(
    text: str,
    start_marker: str,
    end_marker: str,
    max_fallback_span: int = DEFAULT_MAX_FALLBACK_SPAN,
) -> str | None:
    """Extracts the full passage between `start_marker` and `end_marker`.

    If `start_marker` can't be located at all, there's nothing to anchor
    on, so this returns None (the caller drops the location, same policy
    as a missing excerpt). If `end_marker` can't be located (searched after
    the start position), falls back to a bounded window from the start
    rather than dropping the passage entirely -- a partial capture is still
    useful.
    """
    start_span = _find_marker(text, start_marker)
    if start_span is None:
        return None
    start_idx, _ = start_span

    end_span = _find_marker(text, end_marker, start_pos=start_idx)
    if end_span is not None:
        _, end_idx = end_span
    else:
        end_idx = min(len(text), start_idx + max_fallback_span)

    passage = text[start_idx:end_idx].strip()
    return passage or None
