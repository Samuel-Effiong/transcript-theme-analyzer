"""Generates one consolidated Word (.docx) report per theme with every
relevant excerpt from every transcript that matched that theme. A theme that
fits ``DOCX_VOLUME_WORDS`` of excerpt text is one file, ``docx/<theme_id>.docx``;
a larger theme is a folder ``docx/<theme_id>/`` with an index document and
numbered volumes (``<theme_id>-index.docx``, ``<theme_id>-vol-01.docx``, ...),
a transcript never being split across volumes. Each document holds:

- Document title = the search theme.
- An overview table (transcript × relevance score, in ranked order) right
  after the title, with clickable links that jump to each transcript's
  section further down.
- One section per transcript where the theme was found (``report.is_found``)
  with at least one located excerpt, headed
  ``Transcript: <name>`` so an excerpt can always be traced back to its
  source. One sub-heading per excerpt, auto-generated from that excerpt's
  own context rather than just the search keyword.

Usage (regenerate later from the internal cache, without re-running analysis):
    python -m transcript_theme_analyzer.word_report --cache results/.raw_results.json
"""
from __future__ import annotations

import argparse
import os
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from .cache import read_cache
from .config import load_config
from .report import BatchReport, TranscriptResult, build_report, is_found, project_theme
from .themes import Theme

_LINK_COLOR = RGBColor(0x25, 0x6A, 0xBF)  # same blue used for html links/bars
_FAILED_COLOR = RGBColor(0xD0, 0x3B, 0x3B)  # same red used for the html FAILED badge


def _add_horizontal_rule(document: Document) -> None:
    """A bottom-bordered empty paragraph, python-docx's standard recipe for
    a horizontal rule (Word has no native <hr> element)."""
    p = document.add_paragraph()
    p_pr = p._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "auto")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def _add_bookmark(paragraph, name: str, bookmark_id: int) -> None:
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(bookmark_id))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(bookmark_id))
    paragraph._p.insert(0, start)
    paragraph._p.append(end)


def _add_internal_link(paragraph, text: str, bookmark_name: str) -> None:
    """Appends a clickable link to ``paragraph`` that jumps to ``bookmark_name``
    elsewhere in the same document (an anchor-only hyperlink — no external
    relationship, so python-docx's public hyperlink-reading API won't see it,
    but Word/LibreOffice render and follow it like any other link)."""
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("w:anchor"), bookmark_name)

    run_element = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "256ABF")
    r_pr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.append(underline)
    run_element.append(r_pr)

    text_element = OxmlElement("w:t")
    text_element.text = text
    run_element.append(text_element)

    hyperlink.append(run_element)
    paragraph._p.append(hyperlink)


def _add_excerpt(document: Document, location: dict) -> None:
    # The heading is generated from the location's own title (produced by
    # the analysis pass for that specific passage), not from the search
    # theme, so headings differ across excerpts even within the same
    # transcript.
    heading_text = (location.get("title") or "").strip().rstrip(".")
    document.add_heading(heading_text or "Relevant Passage", level=3)

    meta_bits = [b for b in (location.get("timestamp"), location.get("speaker")) if b]
    if meta_bits:
        meta_p = document.add_paragraph(" · ".join(meta_bits))
        for run in meta_p.runs:
            run.italic = True
            run.font.size = Pt(9)

    excerpt = (location.get("excerpt") or "").strip()
    paragraphs = [p.strip() for p in excerpt.split("\n\n") if p.strip()] or [excerpt]
    for i, para_text in enumerate(paragraphs):
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Pt(18)
        prefix = "“" if i == 0 else ""
        suffix = "”" if i == len(paragraphs) - 1 else ""
        p.add_run(f"{prefix}{para_text}{suffix}").italic = True


def _matching_transcripts(report: BatchReport, min_score: int | None) -> list[TranscriptResult]:
    return [
        t for t in report.transcripts
        if t.best_model is not None
        and is_found(t.best_model.score, t.best_model.explicitness)
        and t.best_model.locations
        and (min_score is None or (t.best_score or 0) >= min_score)
    ]


DEFAULT_VOLUME_WORDS = 150_000


def _excerpt_words(t: TranscriptResult) -> int:
    """Whitespace-split word count of the transcript's excerpt text only;
    headings, titles, timestamps and speakers do not count."""
    return sum(len((loc.get("excerpt") or "").split()) for loc in t.best_model.locations)


def _plan_volumes(matched: list[TranscriptResult], budget: int) -> list[list[TranscriptResult]]:
    """Packs the ranked transcripts greedily into volumes of at most ``budget``
    excerpt words. A transcript is never split: one larger than the budget
    fills a volume on its own."""
    volumes: list[list[TranscriptResult]] = []
    current: list[TranscriptResult] = []
    current_words = 0
    for t in matched:
        words = _excerpt_words(t)
        if current and current_words + words > budget:
            volumes.append(current)
            current, current_words = [], 0
        current.append(t)
        current_words += words
    if current:
        volumes.append(current)
    return volumes


def _add_table_heading(document: Document, text: str) -> None:
    heading = document.add_paragraph()
    run = heading.add_run(text)
    run.bold = True
    run.font.size = Pt(13)


def _new_table(document: Document, columns: tuple[str, ...]):
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Light Grid Accent 1"
    for cell, text in zip(table.rows[0].cells, columns):
        cell.paragraphs[0].add_run(text).bold = True
    return table


def _add_name_cell(cell, t: TranscriptResult, link_to: str | None) -> None:
    if link_to is not None:
        _add_internal_link(cell.paragraphs[0], t.name, link_to)
    else:
        suffix = "" if t.best_model is not None else " (failed)"
        cell.paragraphs[0].add_run(f"{t.name}{suffix}")


def _add_score_cell(cell, t: TranscriptResult) -> None:
    score_run = cell.paragraphs[0].add_run(
        str(t.best_score) if t.best_score is not None else "FAILED"
    )
    if t.best_score is None:
        score_run.font.color.rgb = _FAILED_COLOR


def _add_overview_table(
    document: Document,
    report: BatchReport,
    matched_ids: set[int],
    transcripts: list[TranscriptResult] | None = None,
) -> None:
    """Transcript x score table. ``transcripts`` restricts the rows (a volume
    lists only its own); bookmarks always use the global rank in ``report``."""
    _add_table_heading(document, "Overview")
    table = _new_table(document, ("Transcript", "Relevance Score"))
    for t in report.transcripts if transcripts is None else transcripts:
        name_cell, score_cell = table.add_row().cells
        rank = report.transcripts.index(t)
        _add_name_cell(name_cell, t, f"t{rank}" if id(t) in matched_ids else None)
        _add_score_cell(score_cell, t)


def _add_index_table(document: Document, report: BatchReport, volume_of: dict[int, int]) -> None:
    _add_table_heading(document, "Index")
    table = _new_table(document, ("Transcript", "Relevance Score", "Passages", "Volume"))
    for t in report.transcripts:
        name_cell, score_cell, passages_cell, volume_cell = table.add_row().cells
        _add_name_cell(name_cell, t, None)
        _add_score_cell(score_cell, t)
        passages_cell.paragraphs[0].add_run(
            str(len(t.best_model.locations)) if t.best_model is not None else "\u2014"
        )
        volume = volume_of.get(id(t))
        volume_cell.paragraphs[0].add_run(str(volume) if volume is not None else "\u2014")


def _add_transcript_sections(
    document: Document, report: BatchReport, transcripts: list[TranscriptResult]
) -> None:
    """One ``Transcript: <name>`` section per entry, bookmarked by its global
    rank in ``report`` so overview links resolve."""
    for n, t in enumerate(transcripts):
        if n:
            _add_horizontal_rule(document)
        rank = report.transcripts.index(t)
        heading = document.add_heading(f"Transcript: {t.name}", level=2)
        _add_bookmark(heading, f"t{rank}", rank)
        for location in t.best_model.locations:
            _add_excerpt(document, location)


def _add_theme_title(document: Document, report: BatchReport) -> None:
    """Heading = the theme's label; the full theme text follows as a paragraph
    when it is longer than the label, so each file records what was asked."""
    document.add_heading(report.theme, level=1)
    if report.theme_text and report.theme_text != report.theme:
        document.add_paragraph(report.theme_text)


def build_docx(report: BatchReport, min_score: int | None = None) -> Document:
    document = Document()
    _add_theme_title(document, report)
    document.add_paragraph(f"Generated {report.generated_at}")

    matched = _matching_transcripts(report, min_score)
    matched_ids = {id(t) for t in matched}
    _add_overview_table(document, report, matched_ids)

    if not matched:
        document.add_paragraph("No transcripts contained relevant excerpts for this theme.")
        return document

    _add_transcript_sections(document, report, matched)
    return document


def build_volume_docx(
    report: BatchReport,
    volume: list[TranscriptResult],
    volume_no: int,
    volume_count: int,
) -> Document:
    """One volume of a theme split across several files: the theme title, a
    ``Volume N of M`` line, an overview of only this volume's transcripts
    (all linked) and their full sections."""
    document = Document()
    _add_theme_title(document, report)
    document.add_paragraph(f"Volume {volume_no} of {volume_count}")
    document.add_paragraph(f"Generated {report.generated_at}")
    _add_overview_table(document, report, {id(t) for t in volume}, transcripts=volume)
    _add_transcript_sections(document, report, volume)
    return document


def build_index_docx(report: BatchReport, volume_of: dict[int, int]) -> Document:
    """The index of a multi-volume theme: every transcript in the run with its
    score, passage count and the volume that holds it. ``volume_of`` maps
    ``id(transcript)`` to a 1-based volume number for the matched ones."""
    document = Document()
    _add_theme_title(document, report)
    document.add_paragraph(f"Index of {max(volume_of.values(), default=0)} volumes")
    document.add_paragraph(f"Generated {report.generated_at}")
    _add_index_table(document, report, volume_of)
    return document


def write_docx(report: BatchReport, out_path: str, min_score: int | None = None) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    build_docx(report, min_score=min_score).save(out_path)
    return out_path


def write_docx_from_data(
    theme: str,
    transcripts: dict[str, list[tuple[str, dict]]],
    out_path: str,
    min_score: int | None = None,
) -> str:
    """Build and save the Word document directly from in-memory run payloads
    (single-theme shape, see ``report.build_report``) — no disk round-trip."""
    report = build_report(theme, transcripts)
    return write_docx(report, out_path, min_score=min_score)


def _remove_stale(docx_dir: str, keep: set[str]) -> None:
    """Removes this tool's output that the current run did not write, and
    nothing else. ``keep`` holds the absolute path of every file written.

    Directly in ``docx_dir``: a ``*.docx`` not in ``keep`` goes (``~$`` Word
    lock files stay). A subdirectory is recognised as a theme's volume folder
    only by containing ``<dir>-index.docx``; inside it every ``<dir>-*.docx``
    not in ``keep`` goes, then the folder itself if that left it empty. Other
    subdirectories, non-docx files and LibreOffice ``.~lock.*#`` files are
    never touched."""
    if not os.path.isdir(docx_dir):
        return

    def is_tool_docx(name: str, prefix: str = "") -> bool:
        return (name.endswith(".docx") and name.startswith(prefix)
                and not name.startswith("~$"))

    def remove_unkept(folder: str, prefix: str) -> None:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if (is_tool_docx(name, prefix) and os.path.isfile(path)
                    and os.path.abspath(path) not in keep):
                os.remove(path)

    remove_unkept(docx_dir, "")
    for name in os.listdir(docx_dir):
        folder = os.path.join(docx_dir, name)
        if os.path.isdir(folder) and os.path.isfile(os.path.join(folder, f"{name}-index.docx")):
            remove_unkept(folder, f"{name}-")
            try:
                os.rmdir(folder)
            except OSError:
                pass  # something that is not ours is still inside


def write_theme_docxs(
    themes: list[Theme],
    transcripts: dict[str, list[tuple[str, dict]]],
    out_dir: str,
    volume_words: int = DEFAULT_VOLUME_WORDS,
) -> list[str]:
    """Write every theme's Word report(s) under ``out_dir/docx`` and return
    the written paths in theme order.

    A theme whose matched excerpts fit ``volume_words`` is one file,
    ``docx/<theme.id>.docx``. A larger theme becomes a folder
    ``docx/<theme.id>/`` holding ``<theme.id>-index.docx`` (returned first)
    and ``<theme.id>-vol-NN.docx`` for each volume, in order. Stale output of
    a previous run is removed (see ``_remove_stale``)."""
    docx_dir = os.path.join(out_dir, "docx")
    paths: list[str] = []
    for theme in themes:
        report = build_report(theme.name, project_theme(transcripts, theme), theme_text=theme.text)
        plan = _plan_volumes(_matching_transcripts(report, None), volume_words)
        if len(plan) <= 1:
            paths.append(write_docx(report, os.path.join(docx_dir, f"{theme.id}.docx")))
            continue
        folder = os.path.join(docx_dir, theme.id)
        os.makedirs(folder, exist_ok=True)
        volume_of = {id(t): k + 1 for k, volume in enumerate(plan) for t in volume}
        index_path = os.path.join(folder, f"{theme.id}-index.docx")
        build_index_docx(report, volume_of).save(index_path)
        paths.append(index_path)
        width = max(2, len(str(len(plan))))
        for k, volume in enumerate(plan, start=1):
            path = os.path.join(folder, f"{theme.id}-vol-{k:0{width}d}.docx")
            build_volume_docx(report, volume, k, len(plan)).save(path)
            paths.append(path)
    _remove_stale(docx_dir, {os.path.abspath(p) for p in paths})
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate every theme's Word report(s) under docx/ from the internal analysis cache"
    )
    parser.add_argument("--cache", required=True, help="Path to the .raw_results.json cache file")
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write docx/ into (defaults to the cache file's directory)",
    )
    parser.add_argument(
        "--docx-volume-words",
        type=int,
        default=None,
        help="Excerpt words per Word volume before a theme is split into numbered volumes "
        "(default: DOCX_VOLUME_WORDS or 150000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.dirname(args.cache) or "."
    config = load_config()
    if args.docx_volume_words is not None:
        config.docx_volume_words = args.docx_volume_words
    config = config.validated()
    try:
        themes, transcripts = read_cache(args.cache)
    except (ValueError, OSError) as exc:  # OSError covers FileNotFoundError
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    for path in write_theme_docxs(themes, transcripts, out_dir, volume_words=config.docx_volume_words):
        print(f"[report:docx] {path}")


if __name__ == "__main__":
    main()
