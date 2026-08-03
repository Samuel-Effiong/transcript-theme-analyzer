"""Reduce step: merges chunk-level analyses into one final AnalysisResult.

The final relevance_score is an explicit, tunable heuristic (not a naive
max/average of chunk scores): it combines
  - peak intensity   (the single most relevant chunk -- catches a strong,
                       concentrated discussion that would get diluted by
                       averaging over a long transcript)
  - average intensity (weighted by chunk length -- rewards sustained
                       discussion over one spike)
  - coverage breadth (what fraction of the transcript, by length, discusses
                       the theme at all, weighted double for "explicit" vs
                       "tangential" chunks)

Weights are keyword arguments so this can be tuned, or swapped for an LLM
aggregation pass later, without touching the map step.
"""
from __future__ import annotations

from .schema import ChunkAnalysis, Location

DEFAULT_WEIGHTS = {
    "peak": 0.30,
    "average": 0.40,
    "coverage": 0.30,
}

EXPLICITNESS_COVERAGE_WEIGHT = {
    "explicit": 1.0,
    "tangential": 0.5,
    "absent": 0.0,
}


def compute_aggregate_score(
    chunk_results: list[ChunkAnalysis],
    chunk_lengths: list[int],
    weights: dict[str, float] | None = None,
) -> int:
    if not chunk_results:
        return 0
    weights = weights or DEFAULT_WEIGHTS
    total_length = sum(chunk_lengths) or 1

    peak = max(c.relevance_score for c in chunk_results)

    weighted_sum = sum(c.relevance_score * length for c, length in zip(chunk_results, chunk_lengths))
    average = weighted_sum / total_length

    coverage_weighted_length = sum(
        length * EXPLICITNESS_COVERAGE_WEIGHT.get(c.explicitness, 0.0)
        for c, length in zip(chunk_results, chunk_lengths)
    )
    coverage = (coverage_weighted_length / total_length) * 100

    final = weights["peak"] * peak + weights["average"] * average + weights["coverage"] * coverage
    return max(0, min(100, round(final)))


def merge_and_dedupe_locations(
    all_locations: list[Location], overlap_threshold: float = 0.5
) -> list[Location]:
    """Merge locations from overlapping chunk boundaries.

    Locations are already translated to full-transcript char offsets by the
    caller. Two locations are treated as duplicates of the same mention if
    their char ranges overlap by more than `overlap_threshold` of the smaller
    range's length -- this is what happens when the same passage falls in the
    overlap region of two adjacent chunks and both flag it.
    """
    if not all_locations:
        return []

    ordered = sorted(all_locations, key=lambda loc: (loc.char_start, loc.char_end))
    kept: list[Location] = [ordered[0]]

    for loc in ordered[1:]:
        prev = kept[-1]
        overlap = min(loc.char_end, prev.char_end) - max(loc.char_start, prev.char_start)
        smaller_len = min(loc.char_end - loc.char_start, prev.char_end - prev.char_start) or 1
        if overlap > 0 and (overlap / smaller_len) > overlap_threshold:
            # Keep the one with the longer excerpt (usually the more complete capture).
            if len(loc.excerpt) > len(prev.excerpt):
                kept[-1] = loc
            continue
        kept.append(loc)

    return kept
