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


def _normalize_excerpt(excerpt: str) -> str:
    return " ".join(excerpt.split()).lower()


def merge_and_dedupe_locations(all_locations: list[Location]) -> list[Location]:
    """Merge locations from overlapping chunk boundaries.

    There are no character offsets to compare (removed entirely -- models
    were unreliable about reporting them). Instead, two locations are
    treated as duplicates of the same mention if their excerpts are
    identical after normalizing whitespace/case -- this is what happens when
    the same passage falls in the overlap region of two adjacent chunks and
    both flag it, since it's literally the same source text either way.
    """
    if not all_locations:
        return []

    kept: list[Location] = []
    index_by_norm: dict[str, int] = {}

    for loc in all_locations:
        norm = _normalize_excerpt(loc.excerpt)
        if norm in index_by_norm:
            idx = index_by_norm[norm]
            # Keep the one with the longer excerpt (usually the more complete capture).
            if len(loc.excerpt) > len(kept[idx].excerpt):
                kept[idx] = loc
            continue
        index_by_norm[norm] = len(kept)
        kept.append(loc)

    return kept
