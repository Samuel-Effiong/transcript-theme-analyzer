import json
import re

from transcript_theme_analyzer.run_report import (
    build_run_report,
    render_run_html,
    write_run_html,
)
from transcript_theme_analyzer.themes import Theme

A = Theme(id="a", name="Alpha")
B = Theme(id="b", name="Beta")


def entry(score, explicitness="explicit", reasoning="why", locations=None):
    return {
        "theme": "x",
        "relevance_score": score,
        "explicitness": explicitness,
        "reasoning": reasoning,
        "locations": locations or [],
    }


def usage(cost=0.0123, source="openrouter", discount=None, prompt=100, completion=50, cached=10):
    return {
        "calls": 1,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "total_tokens": prompt + completion,
        "cost": cost,
        "cost_source": source,
        "discount": discount,
        "by_call": [],
    }


def payload(themes, use=None, theme_usage=None):
    return {
        "model_used": "m",
        "chunked": False,
        "themes": themes,
        "_meta": {
            "elapsed_seconds": 1.0,
            "usage": use if use is not None else usage(),
            "theme_usage": theme_usage
            if theme_usage is not None
            else {t: {"tokens": 10, "cost": 0.001} for t in themes},
        },
    }


def test_cost_summary_shows_reported_cost_and_source():
    tr = {"t1": [("m1", payload({"a": entry(50), "b": entry(0, "absent")}))]}
    html = render_run_html(build_run_report([A, B], tr))
    assert "$0.0123" in html
    assert "Actual amounts charged by OpenRouter" in html


def test_discount_labelled_as_estimate():
    tr = {"t1": [("m1", payload({"a": entry(50)}, use=usage(discount=0.0123)))]}
    html = render_run_html(build_run_report([A], tr))
    assert "Provider discount (est., from list prices): $0.0123" in html


def test_unknown_cost_is_na_not_zero():
    tr = {"t1": [("m1", payload({"a": entry(50)}, use=usage(cost=None, source="openrouter"),
                                theme_usage={"a": {"tokens": 5, "cost": None}}))]}
    html = render_run_html(build_run_report([A], tr))
    assert "n/a" in html
    assert "$0.0000" not in html
    assert "Actual amounts charged" not in html


def test_total_cost_none_if_any_contributor_unknown():
    tr = {
        "t1": [("m1", payload({"a": entry(5)}, use=usage(cost=0.01)))],
        "t2": [("m1", payload({"a": entry(5)}, use=usage(cost=None, source=None)))],
    }
    rep = build_run_report([A], tr)
    assert rep.costs.cost is None
    assert rep.costs.discount is None


def test_prices_and_mixed_source_lines():
    tr = {"t1": [("m1", payload({"a": entry(5)}, use=usage(source="prices")))]}
    html = render_run_html(build_run_report([A], tr))
    assert "Estimated from prices in settings (LLM_PRICE_*) or the OpenRouter price list." in html
    tr = {
        "t1": [("m1", payload({"a": entry(5)}, use=usage(source="prices")))],
        "t2": [("m1", payload({"a": entry(5)}, use=usage(source="openrouter")))],
    }
    rep = build_run_report([A], tr)
    assert rep.costs.cost_source == "mixed"
    html = render_run_html(rep)
    assert "Actual amounts charged by OpenRouter" in html
    assert "Estimated from prices" in html


def test_per_theme_headers_say_est():
    tr = {"t1": [("m1", payload({"a": entry(50), "b": entry(0, "absent")}))]}
    html = render_run_html(build_run_report([A, B], tr))
    assert "Tokens (est.)" in html
    assert "Cost (est.)" in html
    rep = build_run_report([A, B], tr)
    by = {c.theme.id: c for c in rep.costs.per_theme}
    assert by["a"].found == 1 and by["b"].found == 0
    assert by["a"].tokens == 10


def test_one_cell_per_transcript_and_theme_with_absent_and_failed():
    tr = {
        "t1": [("m1", payload({"a": entry(50), "b": entry(0, "absent")}))],
        "t2": [("m1", {"model": "m1", "error": "boom"})],
    }
    html = render_run_html(build_run_report([A, B], tr))
    assert len(re.findall(r'<td class="cell', html)) == 4
    assert html.count(">FAILED<") >= 2
    assert "–" in html


def test_tangential_cell_class():
    tr = {"t1": [("m1", payload({"a": entry(30, "tangential")}))]}
    html = render_run_html(build_run_report([A], tr))
    assert "cell-tangential" in html


def test_failed_transcript_contributes_no_usage():
    tr = {
        "ok": [("m1", payload({"a": entry(5)}, use=usage(cost=0.01)))],
        "bad": [("m1", {"model": "m1", "error": "boom"})],
    }
    rep = build_run_report([A], tr)
    assert rep.costs.cost == 0.01
    assert rep.costs.total_tokens == 150


def test_theme_name_is_escaped():
    evil = Theme(id="x", name="<script>x</script>")
    tr = {"t1": [("m1", payload({"x": entry(50)}))]}
    html = render_run_html(build_run_report([evil], tr))
    assert "&lt;script&gt;x&lt;/script&gt;" in html
    assert "<script>x</script>" not in html


def test_best_model_per_theme_wins_when_models_disagree():
    tr = {
        "t1": [
            ("m1", payload({"a": entry(20), "b": entry(90)})),
            ("m2", payload({"a": entry(80), "b": entry(10)})),
        ]
    }
    rep = build_run_report([A, B], tr)
    cells = rep.rows[0].cells
    assert cells["a"].score == 80 and cells["a"].model == "m2"
    assert cells["b"].score == 90 and cells["b"].model == "m1"


def _detail_data(html):
    m = re.search(r'<script type="application/json" id="detail-data">(.*?)</script>', html, re.S)
    assert m, "detail data blob missing"
    return json.loads(m.group(1))


def test_details_and_picker_present(tmp_path):
    tr = {"t1": [("m1", payload({"a": entry(50, reasoning="because")}))]}
    out = write_run_html([A], tr, str(tmp_path / "sub" / "report.html"))
    html = open(out, encoding="utf-8").read()
    assert 'id="detail"' in html
    assert "<details" not in html  # panels are built on click, not up front
    data = _detail_data(html)
    assert data["rows"][0]["cells"]["a"]["reasoning"] == "because"
    assert re.search(r'<td class="cell"[^>]*data-theme="a"[^>]*data-row="1"', html)
    assert "All themes" in html
    assert "<select" in html
    assert "src=" not in html and "href=\"http" not in html


def test_only_found_cells_get_detail_data():
    tr = {
        "t1": [("m1", payload({"a": entry(50), "b": entry(0, "absent")}))],
        "t2": [("m1", {"model": "m1", "error": "boom"})],
    }
    html = render_run_html(build_run_report([A, B], tr))
    data = _detail_data(html)
    assert list(data["rows"][0]["cells"]) == ["a"]
    assert data["rows"][1]["cells"] == {}
    assert html.count("data-row=") == 1


def test_detail_json_cannot_break_out_of_its_script_tag():
    evil = "</script><script>alert(1)</script>"
    loc = {"excerpt": evil, "title": evil, "timestamp": None, "speaker": None}
    tr = {evil: [("m1", payload({"a": entry(50, reasoning=evil, locations=[loc])}))]}
    html = render_run_html(build_run_report([A], tr))
    # Exactly the two closing tags the page itself has: data blob and JS.
    assert html.count("</script>") == 2
    assert "<script>alert(1)" not in html
    data = _detail_data(html)
    row = data["rows"][0]
    assert row["name"] == evil
    assert row["cells"]["a"]["reasoning"] == evil
    assert row["locations"][0]["excerpt"] == evil


def test_shared_passages_stored_once_and_capped_per_cell():
    locs = [{"excerpt": f"p{i}", "title": None, "timestamp": None, "speaker": None} for i in range(7)]
    tr = {"t1": [("m1", payload({"a": entry(50, locations=locs), "b": entry(40, locations=locs[:2])}))]}
    data = _detail_data(render_run_html(build_run_report([A, B], tr), top_locations=5))
    row = data["rows"][0]
    assert [l["excerpt"] for l in row["locations"]] == ["p0", "p1", "p2", "p3", "p4"]
    assert row["cells"]["a"]["locations"] == [0, 1, 2, 3, 4]
    assert row["cells"]["a"]["more"] == 2
    assert row["cells"]["b"]["locations"] == [0, 1]
    assert row["cells"]["b"]["more"] == 0


def test_other_models_listed_in_detail():
    tr = {
        "t1": [
            ("m1", payload({"a": entry(20)})),
            ("m2", payload({"a": entry(80)})),
            ("m3", {"model": "m3", "error": "boom"}),
        ]
    }
    data = _detail_data(render_run_html(build_run_report([A], tr)))
    cell = data["rows"][0]["cells"]["a"]
    assert cell["model"] == "m2"
    assert cell["others"] == [["m1", 20], ["m3", None]]


def test_single_model_has_no_others():
    tr = {"t1": [("m1", payload({"a": entry(20)}))]}
    data = _detail_data(render_run_html(build_run_report([A], tr)))
    assert data["rows"][0]["cells"]["a"]["others"] == []


def test_filter_is_debounced_and_detail_uses_text_content():
    html = render_run_html(build_run_report([A], {"t1": [("m1", payload({"a": entry(5)}))]}))
    assert "setTimeout(apply, 150)" in html
    assert "innerHTML" not in html
