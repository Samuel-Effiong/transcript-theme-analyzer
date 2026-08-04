from transcript_theme_analyzer.analyzer import _extract_and_fill_location, _repair_parsed_json
from transcript_theme_analyzer.chunker import Chunk
from transcript_theme_analyzer.schema import AnalysisResult, ChunkAnalysis, LocationMarker


def make_chunk(text="...", char_start=1000, char_end=1200, timestamp="[10:00]", speaker="Host"):
    return Chunk(
        index=0,
        char_start=char_start,
        char_end=char_end,
        text=text,
        nearest_timestamp=timestamp,
        nearest_speaker=speaker,
    )


def test_extract_and_fill_location_extracts_the_passage_between_markers():
    chunk = make_chunk(text="Intro. The passage begins here and keeps going until it ends right here. Outro.")
    marker = LocationMarker(start_marker="The passage begins here", end_marker="it ends right here")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.excerpt == "The passage begins here and keeps going until it ends right here"


def test_extract_and_fill_location_fills_timestamp_speaker_from_chunk_when_missing():
    chunk = make_chunk(text="Some text with a start point and an end point in it.", timestamp="[05:00]", speaker="Guest")
    marker = LocationMarker(start_marker="start point", end_marker="end point")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.timestamp == "[05:00]"
    assert result.speaker == "Guest"


def test_extract_and_fill_location_keeps_markers_own_timestamp_speaker_when_present():
    chunk = make_chunk(text="Some text with a start point and an end point in it.", timestamp="[05:00]", speaker="Guest")
    marker = LocationMarker(start_marker="start point", end_marker="end point", timestamp="[01:00]", speaker="Host")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.timestamp == "[01:00]"
    assert result.speaker == "Host"


def test_extract_and_fill_location_preserves_title():
    chunk = make_chunk(text="Some text with a start point and an end point in it.")
    marker = LocationMarker(start_marker="start point", end_marker="end point", title="A Title")
    result = _extract_and_fill_location(marker, chunk)
    assert result is not None
    assert result.title == "A Title"


def test_extract_and_fill_location_returns_none_when_marker_not_found():
    chunk = make_chunk(text="This text does not contain the marker phrase at all.")
    marker = LocationMarker(start_marker="nonexistent phrase", end_marker="also nonexistent")
    assert _extract_and_fill_location(marker, chunk) is None


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


def test_repair_drops_location_entries_missing_start_or_end_marker():
    parsed = {
        "relevance_score": 40,
        "explicitness": "explicit",
        "reasoning": "r",
        "locations": [
            {"start_marker": "a start", "end_marker": "an end"},
            {"start_marker": "only a start, should be dropped"},
            {"title": "no markers at all, should be dropped"},
        ],
    }
    repaired = _repair_parsed_json(parsed, ChunkAnalysis)
    result = ChunkAnalysis.model_validate(repaired)
    assert len(result.locations) == 1
    assert result.locations[0].start_marker == "a start"


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
