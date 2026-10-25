import os

from docx import Document

from transcript_theme_analyzer.report import build_report
from transcript_theme_analyzer.word_report import build_docx, write_docx, write_docx_from_data


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


def _headings(document, level):
    return [p.text for p in document.paragraphs if p.style.name == f"Heading {level}"]


def test_document_title_is_the_theme():
    transcripts = {"ep1": [("m1", make_success(80, locations=[
        {"excerpt": "e1", "title": "First topic"},
    ]))]}
    report = build_report("the glory of God", transcripts)
    document = build_docx(report)
    assert _headings(document, 1) == ["the glory of God"]


def test_one_section_per_transcript_named_after_transcript():
    transcripts = {
        "Sunday Service": [("m1", make_success(80, locations=[
            {"excerpt": "e1", "title": "Topic A"},
        ]))],
        "Leadership Conference": [("m1", make_success(70, locations=[
            {"excerpt": "e2", "title": "Topic B"},
        ]))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report)
    headings2 = _headings(document, 2)
    assert "Transcript: Sunday Service" in headings2
    assert "Transcript: Leadership Conference" in headings2


def test_excerpt_heading_uses_context_not_the_raw_keyword():
    transcripts = {
        "ep1": [("m1", make_success(80, locations=[
            {"excerpt": "e1", "title": "God's Glory Revealed Through Obedience."},
            {"excerpt": "e2", "title": "The Purpose of Living for God's Glory."},
        ]))],
    }
    report = build_report("the glory of God", transcripts)
    document = build_docx(report)
    headings3 = _headings(document, 3)
    assert "God's Glory Revealed Through Obedience" in headings3
    assert "The Purpose of Living for God's Glory" in headings3
    # Distinct excerpts get distinct headings, not a repeat of the theme.
    assert len(set(headings3)) == 2


def test_multiple_excerpts_under_same_transcript_all_included():
    transcripts = {
        "ep1": [("m1", make_success(80, locations=[
            {"excerpt": "e1", "title": "Topic A"},
            {"excerpt": "e2", "title": "Topic B"},
            {"excerpt": "e3", "title": "Topic C"},
        ]))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report)
    body_text = "\n".join(p.text for p in document.paragraphs)
    assert "e1" in body_text and "e2" in body_text and "e3" in body_text
    assert len(_headings(document, 3)) == 3


def test_transcripts_with_no_located_excerpts_are_skipped():
    transcripts = {
        "has_locations": [("m1", make_success(80, locations=[
            {"excerpt": "e1", "title": "Topic A"},
        ]))],
        "no_locations": [("m1", make_success(50, locations=[]))],
        "all_failed": [("m1", make_error("m1"))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report)
    headings2 = _headings(document, 2)
    assert headings2 == ["Transcript: has_locations"]


def test_min_score_filters_out_low_scoring_transcripts():
    transcripts = {
        "high": [("m1", make_success(90, locations=[
            {"excerpt": "e1", "title": "Topic A"},
        ]))],
        "low": [("m1", make_success(10, locations=[
            {"excerpt": "e2", "title": "Topic B"},
        ]))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report, min_score=50)
    assert _headings(document, 2) == ["Transcript: high"]


def test_no_matching_transcripts_produces_a_document_without_crashing():
    transcripts = {"ep1": [("m1", make_success(80, locations=[]))]}
    report = build_report("theme", transcripts)
    document = build_docx(report)
    assert _headings(document, 2) == []
    assert any("No transcripts" in p.text for p in document.paragraphs)


def test_multi_paragraph_excerpt_gets_open_quote_only_on_first_paragraph_and_close_on_last():
    excerpt = "First paragraph of the passage.\n\nSecond paragraph.\n\nThird and final paragraph."
    transcripts = {
        "ep1": [("m1", make_success(80, locations=[
            {"excerpt": excerpt, "title": "A Long Passage"},
        ]))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report)
    body_paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    assert "“First paragraph of the passage." in body_paragraphs
    assert "Second paragraph." in body_paragraphs
    assert "Third and final paragraph.”" in body_paragraphs
    # The middle paragraph carries neither a leading nor trailing quote mark.
    assert "“Second paragraph.”" not in body_paragraphs
    assert "“Second paragraph." not in body_paragraphs


def test_write_docx_produces_a_valid_openable_file(tmp_path):
    transcripts = {"ep1": [("m1", make_success(80, locations=[
        {"excerpt": "e1", "title": "Topic A", "timestamp": "[01:00]", "speaker": "Host"},
    ]))]}
    out_path = str(tmp_path / "out.docx")
    write_docx_from_data("theme", transcripts, out_path)
    assert os.path.isfile(out_path)
    reopened = Document(out_path)
    assert reopened.paragraphs[0].text == "theme"


def test_hostile_content_is_stored_literally_not_executed(tmp_path):
    hostile = "=cmd|' /C calc'!A1"  # CSV-injection-shaped string; docx just stores text
    transcripts = {
        "ep1": [("m1", make_success(80, locations=[
            {"excerpt": hostile, "title": hostile},
        ]))],
    }
    out_path = str(tmp_path / "out.docx")
    write_docx_from_data("theme", transcripts, out_path)
    reopened = Document(out_path)
    all_text = "\n".join(p.text for p in reopened.paragraphs)
    assert hostile in all_text


def test_overview_table_lists_every_transcript_with_its_score_in_ranked_order():
    transcripts = {
        "high": [("m1", make_success(90, locations=[
            {"excerpt": "e1", "title": "Topic A"},
        ]))],
        "no_excerpts": [("m1", make_success(50, locations=[]))],
        "failed": [("m1", make_error("m1"))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report)
    assert len(document.tables) == 1
    table = document.tables[0]

    # header + one row per transcript
    assert len(table.rows) == 1 + len(transcripts)
    header_texts = [c.text for c in table.rows[0].cells]
    assert header_texts == ["Transcript", "Relevance Score"]

    body_rows = [(row.cells[0].text, row.cells[1].text) for row in table.rows[1:]]
    assert body_rows[0] == ("high", "90")
    assert body_rows[1][0].startswith("no_excerpts")
    assert body_rows[1][1] == "50"
    assert body_rows[2][0].startswith("failed")
    assert body_rows[2][1] == "FAILED"


def test_overview_table_links_only_transcripts_with_a_detail_section():
    transcripts = {
        "has_section": [("m1", make_success(80, locations=[
            {"excerpt": "e1", "title": "Topic A"},
        ]))],
        "no_section": [("m1", make_success(20, locations=[]))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report)
    table = document.tables[0]

    linked_row_xml = table.rows[1].cells[0].paragraphs[0]._p.xml
    unlinked_row_xml = table.rows[2].cells[0].paragraphs[0]._p.xml

    assert "w:hyperlink" in linked_row_xml
    assert 'w:anchor="t0"' in linked_row_xml
    assert "w:hyperlink" not in unlinked_row_xml


def test_transcript_heading_carries_the_matching_bookmark():
    transcripts = {
        "only_one": [("m1", make_success(80, locations=[
            {"excerpt": "e1", "title": "Topic A"},
        ]))],
    }
    report = build_report("theme", transcripts)
    document = build_docx(report)

    heading_paragraph = next(
        p for p in document.paragraphs if p.style.name == "Heading 2"
    )
    assert 'w:name="t0"' in heading_paragraph._p.xml


# --- one Word report per theme -------------------------------------------

def _multi_payload(themes, theme_usage=None, model_used="m1"):
    return {
        "model_used": model_used,
        "chunked": False,
        "themes": themes,
        "_meta": {
            "elapsed_seconds": 2.0,
            "usage": {"total_tokens": 500},
            "theme_usage": theme_usage or {},
        },
    }


def _grace_entry():
    return {
        "theme": "Grace",
        "relevance_score": 80,
        "explicitness": "explicit",
        "reasoning": "r",
        "locations": [{"excerpt": "e1", "title": "First topic"}],
    }


def test_write_theme_docxs_one_file_per_theme(tmp_path):
    from transcript_theme_analyzer.themes import Theme
    from transcript_theme_analyzer.word_report import write_theme_docxs

    themes = [Theme(id="grace", name="Grace"), Theme(id="forgiveness", name="Forgiveness")]
    transcripts = {
        "ep1": [("m1", _multi_payload(
            {"grace": _grace_entry(),
             "forgiveness": {"theme": "Forgiveness", "relevance_score": 0, "explicitness": "none",
                             "reasoning": "", "locations": []}},
            {"grace": {"tokens": 120.6, "cost": None}, "forgiveness": {"tokens": 80, "cost": None}},
        ))],
        "ep2": [("m1", make_error("m1"))],
    }
    out_dir = str(tmp_path)
    paths = write_theme_docxs(themes, transcripts, out_dir)
    assert paths == [
        os.path.join(out_dir, "docx", "grace.docx"),
        os.path.join(out_dir, "docx", "forgiveness.docx"),
    ]

    grace = Document(paths[0])
    assert _headings(grace, 1) == ["Grace"]
    assert _headings(grace, 2) == ["Transcript: ep1"]

    forgiveness = Document(paths[1])
    assert _headings(forgiveness, 1) == ["Forgiveness"]
    assert _headings(forgiveness, 2) == []
    rows = [[c.text for c in r.cells] for r in forgiveness.tables[0].rows]
    assert ["ep1", "0"] in rows
    assert any(r[0].startswith("ep2") and r[1] == "FAILED" for r in rows)


def test_project_theme_maps_to_single_theme_shape():
    from transcript_theme_analyzer.report import project_theme
    from transcript_theme_analyzer.themes import Theme

    payload = _multi_payload({"grace": _grace_entry()}, {"grace": {"tokens": 120.6, "cost": 0.01}})
    out = project_theme({"ep1": [("m1", payload)]}, Theme(id="grace", name="Grace"))
    projected = out["ep1"][0][1]
    assert out["ep1"][0][0] == "m1"
    assert projected["relevance_score"] == 80
    assert projected["reasoning"] == "r"
    assert projected["locations"] == [{"excerpt": "e1", "title": "First topic"}]
    assert projected["model_used"] == "m1"
    assert projected["chunked"] is False
    assert projected["_meta"]["elapsed_seconds"] == 2.0
    assert projected["_meta"]["usage"]["total_tokens"] == 121


def test_project_theme_missing_theme_and_errors():
    from transcript_theme_analyzer.report import project_theme
    from transcript_theme_analyzer.themes import Theme

    err = make_error("m1")
    payload = {"model_used": "m1", "chunked": False, "themes": {}}  # no _meta at all
    out = project_theme({"ep1": [("m1", payload)], "ep2": [("m1", err)]},
                        Theme(id="grace", name="Grace"))
    assert out["ep1"][0][1] == {"model": "m1", "error": "no result for theme grace"}
    assert out["ep2"][0][1] == err

    ok = {"model_used": "m1", "chunked": False, "themes": {"grace": _grace_entry()}}
    projected = project_theme({"ep1": [("m1", ok)]}, Theme(id="grace", name="Grace"))["ep1"][0][1]
    assert projected["_meta"]["usage"].get("total_tokens") is None


def test_project_theme_payload_without_themes_key_is_an_error():
    from transcript_theme_analyzer.report import project_theme
    from transcript_theme_analyzer.themes import Theme

    legacy = make_success(80, locations=[{"excerpt": "e", "title": "t"}])
    out = project_theme({"ep1": [("m1", legacy)]}, Theme(id="grace", name="Grace"))
    assert out["ep1"][0][1] == {"model": "m1", "error": "no result for theme grace"}


def _found_rule_transcripts():
    """One shared case set for the 'found' rule: score > 0 and not absent."""
    loc = [{"excerpt": "e", "title": "t"}]

    def e(score, explicitness, locations=loc):
        return {"theme": "Grace", "relevance_score": score, "explicitness": explicitness,
                "reasoning": "r", "locations": locations}

    return {
        "found": [("m1", _multi_payload({"grace": e(40, "explicit")}, {}))],
        "absent-with-passage": [("m1", _multi_payload({"grace": e(30, "absent")}, {}))],
        "zero-with-passage": [("m1", _multi_payload({"grace": e(0, "tangential")}, {}))],
        "found-no-passage": [("m1", _multi_payload({"grace": e(50, "tangential", [])}, {}))],
    }


def test_found_rule_matches_between_docx_and_html(tmp_path):
    from transcript_theme_analyzer.run_report import build_run_report, render_run_html
    from transcript_theme_analyzer.themes import Theme
    from transcript_theme_analyzer.word_report import write_theme_docxs

    themes = [Theme(id="grace", name="Grace")]
    transcripts = _found_rule_transcripts()

    (path,) = write_theme_docxs(themes, transcripts, str(tmp_path))
    # docx section <=> found and at least one passage
    assert _headings(Document(path), 2) == ["Transcript: found"]

    rep = build_run_report(themes, transcripts)
    assert rep.costs.per_theme[0].found == 2  # "found" and "found-no-passage"
    html = render_run_html(rep)
    assert html.count('class="cell cell-absent"') == 2


def test_stale_theme_docx_removed_on_rerun(tmp_path):
    from transcript_theme_analyzer.themes import Theme
    from transcript_theme_analyzer.word_report import write_theme_docxs

    docx_dir = tmp_path / "docx"
    docx_dir.mkdir()
    (docx_dir / "old-theme.docx").write_bytes(b"stale")
    (docx_dir / "notes.txt").write_text("keep me")
    (docx_dir / "nested").mkdir()
    (docx_dir / "nested" / "other.docx").write_bytes(b"keep")

    transcripts = {"ep1": [("m1", _multi_payload({"grace": _grace_entry()}, {}))]}
    write_theme_docxs([Theme(id="grace", name="Grace")], transcripts, str(tmp_path))

    assert sorted(p.name for p in docx_dir.iterdir()) == ["grace.docx", "nested", "notes.txt"]
    assert (docx_dir / "nested" / "other.docx").exists()


# --- volume planning -------------------------------------------------------

from transcript_theme_analyzer.word_report import _excerpt_words, _matching_transcripts, _plan_volumes  # noqa: E402


def _words(n):
    return " ".join(["w"] * n)


def _ranked_report(sizes):
    """One transcript per entry (ep0, ep1, ...), scores 90, 89, ... so
    build_report keeps the given order; each has one excerpt of that many words."""
    transcripts = {
        f"ep{i}": [("m1", make_success(90 - i, locations=[{"excerpt": _words(n), "title": "t"}]))]
        for i, n in enumerate(sizes)
    }
    return build_report("Grace", transcripts)


def _ranked(sizes):
    return _matching_transcripts(_ranked_report(sizes), None)


def test_excerpt_words_counts_only_excerpt_text_across_all_passages():
    (t,) = _ranked([0])
    t.best_model.locations = [
        {"excerpt": "one two three", "title": "a long title here", "speaker": "Pastor", "timestamp": "00:01"},
        {"excerpt": "four\n\nfive", "title": "t"},
    ]
    assert _excerpt_words(t) == 5


def test_plan_volumes_fills_greedily_in_rank_order():
    matched = _ranked([60, 30, 20, 50, 10])
    plan = _plan_volumes(matched, budget=100)
    assert [[t.name for t in vol] for vol in plan] == [["ep0", "ep1"], ["ep2", "ep3", "ep4"]]


def test_plan_volumes_exact_fit_stays_in_volume():
    plan = _plan_volumes(_ranked([50, 50, 1]), budget=100)
    assert [len(v) for v in plan] == [2, 1]


def test_plan_volumes_oversized_transcript_gets_its_own_volume_unsplit():
    plan = _plan_volumes(_ranked([10, 500, 10]), budget=100)
    assert [[t.name for t in vol] for vol in plan] == [["ep0"], ["ep1"], ["ep2"]]
    assert _excerpt_words(plan[1][0]) == 500


def test_plan_volumes_empty_input_is_empty_plan():
    assert _plan_volumes([], budget=100) == []


# --- volume and index documents ------------------------------------------

from transcript_theme_analyzer.word_report import build_index_docx, build_volume_docx  # noqa: E402


def _rows(document, table_index=0):
    return [[c.text for c in r.cells] for r in document.tables[table_index].rows]


def test_volume_docx_has_subtitle_own_overview_and_only_its_transcripts():
    report = _ranked_report([10, 10, 10])
    matched = _matching_transcripts(report, None)
    doc = build_volume_docx(report, matched[1:], volume_no=2, volume_count=2)
    assert _headings(doc, 1) == ["Grace"]
    assert doc.paragraphs[1].text == "Volume 2 of 2"
    assert doc.paragraphs[2].text.startswith("Generated ")
    assert _rows(doc) == [["Transcript", "Relevance Score"], ["ep1", "89"], ["ep2", "88"]]
    assert _headings(doc, 2) == ["Transcript: ep1", "Transcript: ep2"]


def test_volume_docx_bookmarks_use_global_rank_and_links_match():
    report = _ranked_report([10, 10, 10])
    matched = _matching_transcripts(report, None)
    doc = build_volume_docx(report, matched[2:], volume_no=2, volume_count=2)
    xml = doc.element.xml
    assert 'w:name="t2"' in xml and 'w:anchor="t2"' in xml
    assert 'w:name="t0"' not in xml


def test_index_docx_lists_every_transcript_with_passages_and_volume():
    transcripts = {
        "strong": [("m1", make_success(90, locations=[{"excerpt": "a b", "title": "t"}, {"excerpt": "c", "title": "t"}]))],
        "weak": [("m1", make_success(40, locations=[{"excerpt": "d", "title": "t"}]))],
        "nothing": [("m1", make_success(0, locations=[]))],
        "broken": [("m1", make_error("m1"))],
    }
    report = build_report("Grace", transcripts)
    by_name = {t.name: t for t in report.transcripts}
    doc = build_index_docx(report, {id(by_name["strong"]): 1, id(by_name["weak"]): 2})
    assert _headings(doc, 1) == ["Grace"]
    assert doc.paragraphs[1].text == "Index of 2 volumes"
    assert _rows(doc) == [
        ["Transcript", "Relevance Score", "Passages", "Volume"],
        ["strong", "90", "2", "1"],
        ["weak", "40", "1", "2"],
        ["nothing", "0", "0", "—"],
        ["broken (failed)", "FAILED", "—", "—"],
    ]
    assert "w:anchor" not in doc.element.xml


# --- file layout and housekeeping ---------------------------------------

from transcript_theme_analyzer.themes import Theme  # noqa: E402
from transcript_theme_analyzer.word_report import write_theme_docxs  # noqa: E402


def _big_run(sizes, theme_id="grace"):
    """Multi-theme payloads: one transcript per size, each with one excerpt of
    that many words under the given theme."""
    return {
        f"ep{i}": [("m1", _multi_payload({theme_id: {
            "theme": "Grace", "relevance_score": 90 - i, "explicitness": "explicit",
            "reasoning": "r", "locations": [{"excerpt": _words(n), "title": "t"}]}}, {}))]
        for i, n in enumerate(sizes)
    }


def test_theme_within_budget_keeps_flat_file(tmp_path):
    paths = write_theme_docxs([Theme(id="grace", name="Grace")], _big_run([400, 400]), str(tmp_path), volume_words=1000)
    assert paths == [str(tmp_path / "docx" / "grace.docx")]
    assert not (tmp_path / "docx" / "grace").exists()


def test_theme_over_budget_writes_index_and_zero_padded_volumes(tmp_path):
    paths = write_theme_docxs([Theme(id="grace", name="Grace")], _big_run([700, 700, 700]), str(tmp_path), volume_words=1000)
    d = tmp_path / "docx" / "grace"
    assert paths == [str(d / "grace-index.docx"), str(d / "grace-vol-01.docx"),
                     str(d / "grace-vol-02.docx"), str(d / "grace-vol-03.docx")]
    assert not (tmp_path / "docx" / "grace.docx").exists()
    index = Document(paths[0])
    assert _rows(index)[1:] == [["ep0", "90", "1", "1"], ["ep1", "89", "1", "2"], ["ep2", "88", "1", "3"]]
    vol2 = Document(paths[2])
    assert vol2.paragraphs[1].text == "Volume 2 of 3"
    assert _headings(vol2, 2) == ["Transcript: ep1"]


def test_every_passage_is_printed_verbatim_across_volumes(tmp_path):
    run = _big_run([700, 700])
    paths = write_theme_docxs([Theme(id="grace", name="Grace")], run, str(tmp_path), volume_words=1000)
    # Every excerpt word is the token "w" (the first and last carry the quote
    # marks); no heading or caption is that token.
    printed = " ".join(p.text for path in paths[1:] for p in Document(path).paragraphs)
    assert sum(tok.strip("\u201c\u201d") == "w" for tok in printed.split()) == 1400


def test_theme_with_no_matches_is_a_flat_file_not_a_directory(tmp_path):
    run = {"ep1": [("m1", _multi_payload({"grace": {"theme": "Grace", "relevance_score": 0,
           "explicitness": "none", "reasoning": "", "locations": []}}, {}))]}
    paths = write_theme_docxs([Theme(id="grace", name="Grace")], run, str(tmp_path), volume_words=1000)
    assert paths == [str(tmp_path / "docx" / "grace.docx")]
    assert "No transcripts contained relevant excerpts" in "\n".join(p.text for p in Document(paths[0]).paragraphs)


def test_switching_layouts_leaves_one_layout_on_disk(tmp_path):
    theme = [Theme(id="grace", name="Grace")]
    write_theme_docxs(theme, _big_run([700, 700, 700]), str(tmp_path), volume_words=1000)
    write_theme_docxs(theme, _big_run([100]), str(tmp_path), volume_words=1000)
    assert sorted(p.name for p in (tmp_path / "docx").iterdir()) == ["grace.docx"]
    write_theme_docxs(theme, _big_run([700, 700, 700]), str(tmp_path), volume_words=1000)
    assert sorted(p.name for p in (tmp_path / "docx").iterdir()) == ["grace"]


def test_fewer_volumes_on_rerun_removes_higher_numbers_and_keeps_lock_files(tmp_path):
    theme = [Theme(id="grace", name="Grace")]
    write_theme_docxs(theme, _big_run([700, 700, 700]), str(tmp_path), volume_words=1000)
    d = tmp_path / "docx" / "grace"
    (d / ".~lock.grace-vol-01.docx#").write_text("libreoffice lock")
    (d / "~$grace-vol-02.docx").write_bytes(b"word lock")
    write_theme_docxs(theme, _big_run([700, 700]), str(tmp_path), volume_words=1000)
    assert sorted(p.name for p in d.iterdir()) == [
        ".~lock.grace-vol-01.docx#", "grace-index.docx", "grace-vol-01.docx", "grace-vol-02.docx", "~$grace-vol-02.docx"]


def test_stale_volume_directory_of_dropped_theme_is_removed_but_foreign_dirs_stay(tmp_path):
    write_theme_docxs([Theme(id="old", name="Old")], _big_run([700, 700], theme_id="old"), str(tmp_path), volume_words=1000)
    (tmp_path / "docx" / "mine").mkdir()
    (tmp_path / "docx" / "mine" / "mine-notes.docx").write_bytes(b"keep")
    write_theme_docxs([Theme(id="grace", name="Grace")], _big_run([100]), str(tmp_path), volume_words=1000)
    assert sorted(p.name for p in (tmp_path / "docx").iterdir()) == ["grace.docx", "mine"]
    assert (tmp_path / "docx" / "mine" / "mine-notes.docx").exists()


def test_volume_directory_with_foreign_file_loses_only_tool_files(tmp_path):
    write_theme_docxs([Theme(id="old", name="Old")], _big_run([700, 700], theme_id="old"), str(tmp_path), volume_words=1000)
    (tmp_path / "docx" / "old" / "my-notes.txt").write_text("keep")
    write_theme_docxs([Theme(id="grace", name="Grace")], _big_run([100]), str(tmp_path), volume_words=1000)
    assert sorted(p.name for p in (tmp_path / "docx" / "old").iterdir()) == ["my-notes.txt"]
