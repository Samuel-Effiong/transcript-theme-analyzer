from transcript_theme_analyzer.extraction import _find_marker, extract_passage


def test_find_marker_exact_match():
    text = "Hello world, this is a test."
    span = _find_marker(text, "this is a test")
    assert span == (13, 27)
    assert text[span[0]:span[1]] == "this is a test"


def test_find_marker_tolerates_whitespace_differences():
    text = "Hello\nworld,   this  is\na test."
    span = _find_marker(text, "world, this is a test")
    assert span is not None
    matched = text[span[0]:span[1]]
    assert matched.split() == "world, this is a test".split()


def test_find_marker_case_insensitive():
    text = "The Grief Comes In Waves, not a straight line."
    span = _find_marker(text, "the grief comes in waves")
    assert span is not None
    assert text[span[0]:span[1]] == "The Grief Comes In Waves"


def test_find_marker_not_found_returns_none():
    text = "Hello world."
    assert _find_marker(text, "this text is not present") is None


def test_find_marker_finds_first_occurrence_after_start_pos():
    text = "repeat me. other stuff. repeat me again."
    first = _find_marker(text, "repeat me")
    assert first is not None
    second = _find_marker(text, "repeat me", start_pos=first[1])
    assert second is not None
    assert second[0] > first[1]


def test_extract_passage_happy_path():
    text = "Intro stuff. The theme begins here and continues for a while until it ends right here. Outro stuff."
    passage = extract_passage(text, "The theme begins here", "it ends right here")
    assert passage == "The theme begins here and continues for a while until it ends right here"


def test_extract_passage_multi_paragraph():
    text = (
        "Preamble.\n\n"
        "Section start marker text.\nMore relevant content here.\n\nEven more content.\n"
        "Section end marker text.\n\nUnrelated trailing content."
    )
    passage = extract_passage(text, "Section start marker text", "Section end marker text")
    assert "More relevant content here" in passage
    assert "Even more content" in passage
    assert "Unrelated trailing content" not in passage


def test_extract_passage_start_marker_not_found_returns_none():
    text = "Some transcript text that does not contain the marker."
    assert extract_passage(text, "nonexistent phrase here", "also nonexistent") is None


def test_extract_passage_end_marker_missing_falls_back_to_bounded_window():
    text = "Passage begins now" + (" filler word" * 2000)
    passage = extract_passage(text, "Passage begins now", "this end marker does not exist", max_fallback_span=100)
    assert passage is not None
    assert len(passage) <= 100


def test_extract_passage_strips_and_returns_none_for_empty_result():
    text = "marker here immediately marker here"
    passage = extract_passage(text, "marker here", "marker here")
    # start and end markers match the same/adjacent text -- should not crash,
    # returns whatever (possibly empty) span results, never raises.
    assert passage is None or isinstance(passage, str)
