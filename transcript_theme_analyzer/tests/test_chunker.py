from transcript_theme_analyzer.chunker import chunk_transcript, has_structure


def test_short_transcript_single_chunk():
    text = "hello world " * 20
    chunks = chunk_transcript(text, chunk_size_tokens=1000, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


def test_long_transcript_multiple_overlapping_chunks():
    text = ("This is a sentence about various topics.\n" * 500)
    chunks = chunk_transcript(text, chunk_size_tokens=200, overlap_tokens=50)
    assert len(chunks) > 1
    # Verify overlap: each chunk (after the first) starts before the previous one's end.
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.char_start < prev.char_end
        assert cur.char_start > prev.char_start
    # Full coverage: last chunk reaches the end of the transcript.
    assert chunks[-1].char_end == len(text)


def test_detects_timestamps_and_speakers():
    text = "[00:01:23] Alice: Hello there.\n[00:01:30] Bob: Hi Alice.\n"
    has_ts, has_sp = has_structure(text)
    assert has_ts
    assert has_sp


def test_plain_text_has_no_structure():
    text = "just some plain unstructured words with no markers at all"
    has_ts, has_sp = has_structure(text)
    assert not has_ts
    assert not has_sp
