from transcript_theme_analyzer import report
from transcript_theme_analyzer.report import build_report


def make_success(score, reasoning="r", locations=None, model_used="m1", chunked=False):
    return {
        "theme": "t",
        "relevance_score": score,
        "reasoning": reasoning,
        "locations": locations or [],
        "model_used": model_used,
        "chunked": chunked,
        "_meta": {"elapsed_seconds": 1.0, "usage": {"total_tokens": 100}},
    }


def make_error(model, error="boom"):
    return {"model": model, "error": error}


def test_build_report_ranks_by_best_score_descending():
    transcripts = {
        "low": [("m1", make_success(10))],
        "high": [("m1", make_success(90))],
        "mid": [("m1", make_success(50))],
    }
    report = build_report("theme", transcripts)
    assert [t.name for t in report.transcripts] == ["high", "mid", "low"]
    assert report.transcripts[0].best_score == 90


def test_build_report_puts_failed_transcripts_last():
    transcripts = {
        "ok": [("m1", make_success(5))],
        "failed": [("m1", make_error("m1"))],
    }
    report = build_report("theme", transcripts)
    assert [t.name for t in report.transcripts] == ["ok", "failed"]
    assert report.transcripts[1].best_score is None
    assert report.transcripts[1].best_model is None


def test_build_report_best_model_picks_max_across_models_ignoring_errors():
    transcripts = {
        "t1": [("m1", make_success(30)), ("m2", make_success(70)), ("m3", make_error("m3"))],
    }
    report = build_report("theme", transcripts)
    t = report.transcripts[0]
    assert t.best_score == 70
    assert t.best_model.model == "m2"


def test_single_theme_page_is_gone():
    """report.html is now the run-wide page from run_report; the old
    single-theme renderer must not linger as a second way to write it."""
    for name in ("render_html", "write_html", "write_html_from_data", "main"):
        assert not hasattr(report, name), name
