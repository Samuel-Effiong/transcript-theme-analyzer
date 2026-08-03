from transcript_theme_analyzer.analyzer import _fill_location_defaults
from transcript_theme_analyzer.chunker import Chunk
from transcript_theme_analyzer.schema import Location


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
