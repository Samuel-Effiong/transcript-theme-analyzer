from transcript_theme_analyzer.analyzer import _fill_location_defaults, _repair_parsed_json
from transcript_theme_analyzer.chunker import Chunk
from transcript_theme_analyzer.schema import AnalysisResult, ChunkAnalysis, Location


def make_chunk(char_start=1000, char_end=1200, timestamp="[10:00]", speaker="Host"):
    return Chunk(
        index=0,
        char_start=char_start,
        char_end=char_end,
        text="...",
        nearest_timestamp=timestamp,
        nearest_speaker=speaker,
    )


def test_fill_location_defaults_fills_timestamp_speaker_from_chunk_when_missing():
    loc = Location(excerpt="e")
    result = _fill_location_defaults(loc, make_chunk(timestamp="[05:00]", speaker="Guest"))
    assert result.timestamp == "[05:00]"
    assert result.speaker == "Guest"


def test_fill_location_defaults_keeps_locations_own_timestamp_speaker_when_present():
    loc = Location(excerpt="e", timestamp="[01:00]", speaker="Host")
    result = _fill_location_defaults(loc, make_chunk(timestamp="[05:00]", speaker="Guest"))
    assert result.timestamp == "[01:00]"
    assert result.speaker == "Host"


def test_fill_location_defaults_preserves_context_summary_and_excerpt():
    loc = Location(excerpt="the exact quote", context_summary="a summary")
    result = _fill_location_defaults(loc, make_chunk())
    assert result.excerpt == "the exact quote"
    assert result.context_summary == "a summary"


def test_repair_fills_missing_required_reasoning_with_empty_string():
    parsed = {"relevance_score": 5, "explicitness": "tangential", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert result.reasoning == ""
    assert result.relevance_score == 5


def test_repair_rounds_and_clamps_a_fractional_relevance_score():
    parsed = {"relevance_score": 0.3, "explicitness": "tangential", "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    assert repaired["relevance_score"] == 0

    parsed = {"relevance_score": 87.6, "explicitness": "explicit", "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    assert repaired["relevance_score"] == 88


def test_repair_drops_location_entries_missing_excerpt_instead_of_failing():
    parsed = {
        "relevance_score": 40,
        "explicitness": "explicit",
        "reasoning": "r",
        "locations": [
            {"excerpt": "a real quote"},
            {"context_summary": "no excerpt here, should be dropped"},
        ],
    }
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert len(result.locations) == 1
    assert result.locations[0].excerpt == "a real quote"


def test_repair_does_not_touch_a_fully_populated_response():
    parsed = {"relevance_score": 42, "explicitness": "explicit", "reasoning": "solid reasoning", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    assert repaired == parsed


def test_repair_defaults_missing_explicitness_to_absent():
    parsed = {"relevance_score": 5, "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert result.explicitness == "absent"


def test_repair_coerces_out_of_enum_explicitness_to_absent():
    parsed = {"relevance_score": 5, "explicitness": "somewhat", "reasoning": "r", "locations": []}
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert result.explicitness == "absent"


def test_repair_keeps_valid_explicitness_values_unchanged():
    for value in ("explicit", "tangential", "absent"):
        parsed = {"relevance_score": 5, "explicitness": value, "reasoning": "r", "locations": []}
        repaired = _repair_parsed_json(parsed, ChunkAnalysis)
        assert repaired["explicitness"] == value


def test_repair_fills_missing_reasoning_on_analysis_result_too():
    parsed = {
        "theme": "t",
        "relevance_score": 10,
        "locations": [],
        "model_used": "m1",
        "chunked": False,
    }
    repaired = _repair_parsed_json(parsed, AnalysisResult)
    result = AnalysisResult.model_validate(repaired)
    assert result.reasoning == ""
