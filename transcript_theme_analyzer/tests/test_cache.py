import json
import os

import pytest

from transcript_theme_analyzer import run_report, word_report
from transcript_theme_analyzer.cache import CACHE_FILENAME, OLD_FORMAT_MESSAGE, read_cache, write_cache
from transcript_theme_analyzer.themes import Theme

THEMES = [Theme("glory", "the glory of God", "His majesty"), Theme("hope", "Hope")]


def make_success(score):
    entry = {"theme": "the glory of God", "relevance_score": score, "explicitness": "explicit",
             "reasoning": "r", "locations": []}
    return {
        "model_used": "m1",
        "chunked": False,
        "themes": {"glory": entry, "hope": dict(entry, theme="Hope")},
        "_meta": {"elapsed_seconds": 1.0, "usage": {"total_tokens": 100}, "theme_usage": {}},
    }


def test_write_cache_creates_dot_prefixed_file(tmp_path):
    path = write_cache(THEMES, {"ep1": [("m1", make_success(50))]}, str(tmp_path))
    assert os.path.basename(path) == CACHE_FILENAME
    assert os.path.isfile(path)


def test_round_trip_preserves_themes_and_success_payloads(tmp_path):
    transcripts = {
        "ep1": [("m1", make_success(80)), ("m2", make_success(40))],
        "ep2": [("m1", make_success(10))],
    }
    path = write_cache(THEMES, transcripts, str(tmp_path))
    themes, loaded = read_cache(path)

    assert themes == THEMES
    assert list(loaded.keys()) == ["ep1", "ep2"]
    assert loaded["ep1"] == [("m1", make_success(80)), ("m2", make_success(40))]
    assert loaded["ep2"] == [("m1", make_success(10))]


def test_round_trip_preserves_error_payloads(tmp_path):
    transcripts = {"ep1": [("m1", {"model": "m1", "error": "timeout"})]}
    path = write_cache(THEMES, transcripts, str(tmp_path))
    _, loaded = read_cache(path)

    assert loaded["ep1"] == [("m1", {"model": "m1", "error": "timeout"})]


def _old_cache(tmp_path):
    path = os.path.join(tmp_path, CACHE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"theme": "the glory of God", "results": {}}, f)
    return path


def test_old_single_theme_cache_is_rejected(tmp_path):
    with pytest.raises(ValueError) as exc:
        read_cache(_old_cache(tmp_path))
    assert str(exc.value) == (
        "This cache was written by the single-theme version; re-run the analysis."
    )
    assert OLD_FORMAT_MESSAGE == str(exc.value)


@pytest.mark.parametrize("module", [run_report, word_report])
def test_regen_mains_exit_1_on_old_cache(tmp_path, capsys, module):
    with pytest.raises(SystemExit) as exc:
        module.main(["--cache", _old_cache(tmp_path)])
    assert exc.value.code == 1
    assert "single-theme version" in capsys.readouterr().err


@pytest.mark.parametrize("module", [run_report, word_report])
def test_regen_mains_exit_1_on_missing_cache(tmp_path, capsys, module):
    with pytest.raises(SystemExit) as exc:
        module.main(["--cache", str(tmp_path / "nope.json")])
    assert exc.value.code == 1
    assert "nope.json" in capsys.readouterr().err


def test_run_report_main_regenerates_html(tmp_path, capsys):
    path = write_cache(THEMES, {"ep1": [("m1", make_success(70))]}, str(tmp_path))
    run_report.main(["--cache", path])
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "the glory of God" in html and "Hope" in html
    assert "report.html" in capsys.readouterr().out

    other = tmp_path / "elsewhere" / "r.html"
    run_report.main(["--cache", path, "--out", str(other)])
    assert other.exists()


def test_word_report_main_regenerates_every_theme_docx(tmp_path, capsys):
    path = write_cache(THEMES, {"ep1": [("m1", make_success(70))]}, str(tmp_path))
    word_report.main(["--cache", path])
    assert (tmp_path / "docx" / "glory.docx").exists()
    assert (tmp_path / "docx" / "hope.docx").exists()
    assert "hope.docx" in capsys.readouterr().out

    word_report.main(["--cache", path, "--out-dir", str(tmp_path / "other")])
    assert (tmp_path / "other" / "docx" / "glory.docx").exists()


def _with_excerpt(payload, words):
    """make_success() payload whose 'glory' entry carries one excerpt of that many words."""
    payload["themes"]["glory"]["locations"] = [{"excerpt": " ".join(["w"] * words), "title": "t"}]
    return payload


def test_word_report_main_honours_docx_volume_words(tmp_path, capsys, monkeypatch):
    transcripts = {f"ep{i}": [("m1", _with_excerpt(make_success(90 - i), 700))] for i in range(2)}
    path = write_cache(THEMES, transcripts, str(tmp_path))

    monkeypatch.setenv("DOCX_VOLUME_WORDS", "1000")
    word_report.main(["--cache", path])
    assert (tmp_path / "docx" / "glory" / "glory-vol-02.docx").exists()
    assert "glory-index.docx" in capsys.readouterr().out

    word_report.main(["--cache", path, "--docx-volume-words", "5000"])
    assert (tmp_path / "docx" / "glory.docx").exists()
    assert not (tmp_path / "docx" / "glory").exists()

    # below the floor clamps to 1000, so the two 700-word transcripts still split
    word_report.main(["--cache", path, "--docx-volume-words", "1"])
    assert (tmp_path / "docx" / "glory" / "glory-vol-02.docx").exists()
