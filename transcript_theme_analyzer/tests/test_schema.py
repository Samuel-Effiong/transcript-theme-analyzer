import pytest
from pydantic import ValidationError

from transcript_theme_analyzer.schema import ChunkAnalysis, Location, LocationMarker


def test_location_only_requires_excerpt():
    loc = Location(excerpt="some quote")
    assert loc.title is None
    assert loc.timestamp is None
    assert loc.speaker is None
    assert not hasattr(loc, "char_start")
    assert not hasattr(loc, "char_end")


def test_location_marker_requires_start_and_end_markers():
    marker = LocationMarker(start_marker="it begins here", end_marker="it ends here")
    assert marker.title is None
    assert marker.timestamp is None
    assert marker.speaker is None

    with pytest.raises(ValidationError):
        LocationMarker(start_marker="only a start")

    with pytest.raises(ValidationError):
        LocationMarker(end_marker="only an end")


def test_chunk_analysis_parses_location_markers_missing_optional_fields():
    """Reproduces the real failure mode: a model response where location
    markers omit title must still validate."""
    data = {
        "relevance_score": 42,
        "explicitness": "tangential",
        "reasoning": "r",
        "locations": [
            {"start_marker": "a start", "end_marker": "an end", "timestamp": "[01:00]"},
            {"start_marker": "another start", "end_marker": "another end"},
        ],
    }
    result = ChunkAnalysis.model_validate(data)
    assert result.locations[0].start_marker == "a start"
    assert result.locations[0].title is None
    assert result.locations[1].title is None


def test_chunk_analysis_explicitness_must_be_a_valid_enum_value():
    data = {
        "relevance_score": 10,
        "explicitness": "somewhat",
        "reasoning": "r",
        "locations": [],
    }
    with pytest.raises(ValidationError):
        ChunkAnalysis.model_validate(data)
