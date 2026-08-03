from transcript_theme_analyzer.schema import ChunkAnalysis, Location


def test_location_only_requires_excerpt():
    loc = Location(excerpt="some quote")
    assert loc.context_summary is None
    assert loc.timestamp is None
    assert loc.speaker is None
    assert not hasattr(loc, "char_start")
    assert not hasattr(loc, "char_end")


def test_chunk_analysis_parses_locations_missing_optional_fields():
    """Reproduces the real failure mode: a model response where locations
    omit context_summary must still validate."""
    data = {
        "relevance_score": 42,
        "explicitness": "tangential",
        "reasoning": "r",
        "locations": [
            {"excerpt": "some quote", "timestamp": "[01:00]"},
            {"excerpt": "another quote", "explicitness": "tangential"},
        ],
    }
    result = ChunkAnalysis.model_validate(data)
    assert result.locations[0].excerpt == "some quote"
    assert result.locations[0].context_summary is None
    assert result.locations[1].context_summary is None
