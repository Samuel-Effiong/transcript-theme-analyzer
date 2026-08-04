import os

from transcript_theme_analyzer.report import (
    build_report,
    render_html,
    write_html_from_data,
)


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


def test_render_html_contains_score_and_reasoning():
    transcripts = {"t1": [("m1", make_success(77, reasoning="deep discussion of the theme"))]}
    report = build_report("theme", transcripts)
    html = render_html(report)
    assert "77" in html
    assert "deep discussion of the theme" in html
    assert "t1" in html


def test_render_html_escapes_hostile_content():
    hostile = "<script>alert(1)</script>"
    transcripts = {
        hostile: [("m1", make_success(50, reasoning=hostile, locations=[
            {"excerpt": hostile, "title": hostile},
        ]))],
    }
    report = build_report("theme", transcripts)
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_html_handles_all_failed_transcript_without_crashing():
    transcripts = {"t1": [("m1", make_error("m1", "provider down"))]}
    report = build_report("theme", transcripts)
    html = render_html(report)
    assert "provider down" in html
    assert "FAILED" in html


def test_render_html_has_filter_input():
    report = build_report("theme", {"t1": [("m1", make_success(50))]})
    html = render_html(report)
    assert 'id="transcript-filter"' in html
    assert "type=\"search\"" in html


def test_render_html_rows_have_data_labels_for_mobile_layout():
    report = build_report("theme", {"t1": [("m1", make_success(50))]})
    html = render_html(report)
    assert 'data-label="Transcript"' in html
    assert 'data-label="Score"' in html
    assert 'data-label="Relevance"' in html


def test_render_html_has_mobile_media_query():
    report = build_report("theme", {"t1": [("m1", make_success(50))]})
    html = render_html(report)
    assert "@media (max-width: 640px)" in html


def test_render_html_transcript_name_links_to_its_detail_section():
    report = build_report("theme", {"t1": [("m1", make_success(50))]})
    html = render_html(report)
    assert 'href="#detail-1"' in html
    assert 'id="detail-1"' in html


def test_render_html_handles_location_with_none_title():
    """title is optional on Location, so a real payload can have it
    explicitly None (not just absent) -- .get(key, "") would NOT catch that
    (the key is present with value None), so this must not raise TypeError
    from html.escape(None)."""
    transcripts = {
        "t1": [("m1", make_success(50, locations=[
            {"excerpt": "some quote", "title": None},
        ]))],
    }
    report = build_report("theme", transcripts)
    html = render_html(report)
    assert "some quote" in html


def test_render_html_preserves_paragraph_breaks_in_multi_paragraph_excerpt():
    excerpt = "First paragraph of the passage.\n\nSecond paragraph of the passage."
    transcripts = {
        "t1": [("m1", make_success(50, locations=[
            {"excerpt": excerpt, "title": "A Title"},
        ]))],
    }
    report = build_report("theme", transcripts)
    html = render_html(report)
    assert "<p>First paragraph of the passage.</p>" in html
    assert "<p>Second paragraph of the passage.</p>" in html
    assert "A Title" in html


def test_write_html_from_data_writes_a_file(tmp_path):
    transcripts = {"t1": [("m1", make_success(60))]}
    out_path = str(tmp_path / "report.html")
    result_path = write_html_from_data("theme", transcripts, out_path)
    assert result_path == out_path
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
