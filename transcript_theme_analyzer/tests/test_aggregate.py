from transcript_theme_analyzer.aggregate import compute_aggregate_score, merge_and_dedupe_locations
from transcript_theme_analyzer.schema import ChunkAnalysis, Location


def make_chunk(score, explicitness):
    return ChunkAnalysis(relevance_score=score, explicitness=explicitness, reasoning="r", locations=[])


def test_all_absent_scores_low():
    chunks = [make_chunk(0, "absent") for _ in range(5)]
    lengths = [100] * 5
    score = compute_aggregate_score(chunks, lengths)
    assert score == 0


def test_one_strong_explicit_chunk_among_absent_scores_moderate_not_max():
    chunks = [make_chunk(90, "explicit")] + [make_chunk(0, "absent") for _ in range(9)]
    lengths = [100] * 10
    score = compute_aggregate_score(chunks, lengths)
    # Peak intensity pulls it up, but low coverage/average keep it well under 90.
    assert 0 < score < 90


def test_sustained_explicit_discussion_scores_high():
    chunks = [make_chunk(80, "explicit") for _ in range(10)]
    lengths = [100] * 10
    score = compute_aggregate_score(chunks, lengths)
    assert score >= 75


def test_dedupe_merges_overlapping_locations():
    locs = [
        Location(excerpt="short", context_summary="a", char_start=100, char_end=120),
        Location(excerpt="a much longer overlapping excerpt", context_summary="a", char_start=105, char_end=130),
        Location(excerpt="distinct elsewhere", context_summary="b", char_start=5000, char_end=5020),
    ]
    merged = merge_and_dedupe_locations(locs)
    assert len(merged) == 2
    assert any(loc.excerpt == "a much longer overlapping excerpt" for loc in merged)
