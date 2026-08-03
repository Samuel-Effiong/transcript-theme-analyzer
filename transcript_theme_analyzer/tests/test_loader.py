import os

import pytest
from docx import Document

from transcript_theme_analyzer.chunker import SPEAKER_RE, has_structure
from transcript_theme_analyzer.loader import detect_format, load_transcript_text


def _make_docx(tmp_path, paragraphs):
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    path = str(tmp_path / "transcript.docx")
    doc.save(path)
    return path


def test_detect_format_recognizes_txt_and_docx(tmp_path):
    txt_path = str(tmp_path / "a.txt")
    open(txt_path, "w").close()
    assert detect_format(txt_path) == ".txt"

    docx_path = _make_docx(tmp_path, ["hello"])
    assert detect_format(docx_path) == ".docx"


def test_detect_format_rejects_unsupported_extension(tmp_path):
    path = str(tmp_path / "a.pdf")
    open(path, "w").close()
    with pytest.raises(ValueError, match="Unsupported transcript file type"):
        detect_format(path)


def test_detect_format_rejects_fake_docx(tmp_path):
    path = str(tmp_path / "fake.docx")
    with open(path, "w", encoding="utf-8") as f:
        f.write("not actually a zip/docx file")
    with pytest.raises(ValueError, match="not a zip container"):
        detect_format(path)


def test_load_transcript_text_txt_reads_raw_content(tmp_path):
    path = str(tmp_path / "t.txt")
    content = "Host: Hello there.\nGuest: Hi!\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    assert load_transcript_text(path) == content


def test_load_transcript_text_docx_extracts_paragraphs_joined_by_newline(tmp_path):
    path = _make_docx(tmp_path, ["First paragraph.", "Second paragraph."])
    text = load_transcript_text(path)
    assert text == "First paragraph.\nSecond paragraph."


def test_load_transcript_text_docx_collapses_multiple_blank_paragraphs(tmp_path):
    path = _make_docx(tmp_path, ["Para one.", "", "", "", "Para two."])
    text = load_transcript_text(path)
    assert text == "Para one.\n\nPara two."


def test_load_transcript_text_docx_preserves_speaker_labels_for_chunker(tmp_path):
    path = _make_docx(tmp_path, [
        "[00:12] Host: Let's talk about grief and loss today.",
        "Guest: Sure, happy to.",
    ])
    text = load_transcript_text(path)

    has_ts, has_sp = has_structure(text)
    assert has_ts is True
    assert has_sp is True
    match = SPEAKER_RE.search(text)
    assert match is not None
    assert match.group(1) == "Host"
