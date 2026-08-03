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


def test_dedupe_merges_identical_excerpts_from_overlapping_chunks():
    locs = [
        Location(excerpt="the grief comes in waves", context_summary="a"),
        Location(excerpt="the grief comes in waves", context_summary="a"),
        Location(excerpt="distinct elsewhere", context_summary="b"),
    ]
    merged = merge_and_dedupe_locations(locs)
    assert len(merged) == 2
    assert any(loc.excerpt == "the grief comes in waves" for loc in merged)


def test_dedupe_is_whitespace_and_case_insensitive():
    locs = [
        Location(excerpt="The Grief Comes In Waves"),
        Location(excerpt="  the   grief comes in waves  "),
    ]
    merged = merge_and_dedupe_locations(locs)
    assert len(merged) == 1


def test_dedupe_keeps_the_longer_excerpt_on_match():
    # Same passage, normalized-equal, but the raw strings differ in length
    # (extra surrounding whitespace) -- the longer raw string is kept as the
    # more complete capture.
    locs = [
        Location(excerpt="the grief comes in waves"),
        Location(excerpt="  the grief comes in waves  "),
    ]
    merged = merge_and_dedupe_locations(locs)
    assert len(merged) == 1
    assert merged[0].excerpt == "  the grief comes in waves  "


def test_dedupe_leaves_distinct_excerpts_unmerged():
    locs = [
        Location(excerpt="first distinct passage"),
        Location(excerpt="second distinct passage"),
        Location(excerpt="third distinct passage"),
    ]
    merged = merge_and_dedupe_locations(locs)
    assert len(merged) == 3
