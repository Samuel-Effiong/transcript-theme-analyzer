import os

from transcript_theme_analyzer.cache import CACHE_FILENAME, read_cache, write_cache


def make_success(score):
    return {
        "theme": "t",
        "relevance_score": score,
        "reasoning": "r",
        "locations": [],
        "model_used": "m1",
        "chunked": False,
        "_meta": {"elapsed_seconds": 1.0, "usage": {"total_tokens": 100}},
    }


def test_write_cache_creates_dot_prefixed_file(tmp_path):
    path = write_cache("theme", {"ep1": [("m1", make_success(50))]}, str(tmp_path))
    assert os.path.basename(path) == CACHE_FILENAME
    assert os.path.isfile(path)


def test_round_trip_preserves_theme_and_success_payloads(tmp_path):
    transcripts = {
        "ep1": [("m1", make_success(80)), ("m2", make_success(40))],
        "ep2": [("m1", make_success(10))],
    }
    path = write_cache("the glory of God", transcripts, str(tmp_path))
    theme, loaded = read_cache(path)

    assert theme == "the glory of God"
    assert set(loaded.keys()) == {"ep1", "ep2"}
    assert loaded["ep1"] == [("m1", make_success(80)), ("m2", make_success(40))]
    assert loaded["ep2"] == [("m1", make_success(10))]


def test_round_trip_preserves_error_payloads(tmp_path):
    transcripts = {"ep1": [("m1", {"model": "m1", "error": "timeout"})]}
    path = write_cache("theme", transcripts, str(tmp_path))
    theme, loaded = read_cache(path)

    assert loaded["ep1"] == [("m1", {"model": "m1", "error": "timeout"})]


def test_read_cache_missing_theme_key_defaults_gracefully(tmp_path):
    import json

    path = os.path.join(tmp_path, CACHE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": {}}, f)

    theme, loaded = read_cache(path)
    assert theme == "(unknown theme)"
    assert loaded == {}
