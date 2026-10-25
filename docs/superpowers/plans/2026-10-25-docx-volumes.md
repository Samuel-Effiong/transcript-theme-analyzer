# Per-Theme Word Reports in Volumes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split a theme's Word report into numbered volumes plus an index document when its excerpt text exceeds a configurable word budget, keeping every transcript and every passage verbatim, and stop passage markers matching inside words.

**Architecture:** `word_report.py` keeps the existing single-document builder untouched and adds three pure steps in front of it: count a transcript's excerpt words, pack the ranked matched transcripts greedily into volumes, and decide the layout (one flat file when the plan has at most one volume, otherwise a per-theme directory with an index document and one document per volume). Housekeeping removes only files this tool recognisably wrote. The budget travels as `Config.docx_volume_words` from the `DOCX_VOLUME_WORDS` env var, overridable by `--docx-volume-words` on both the analysis CLI and the `word_report --cache` regenerator. The marker fix is a regex change in `extraction._find_marker`.

**Tech Stack:** Python ≥3.10, pydantic 2, python-docx, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-docx-volumes-design.md`

All paths below are relative to `transcript_theme_analyzer/` unless they start with `docs/` or `.env`. Run tests from the repo root with `.venv/bin/python -m pytest transcript_theme_analyzer/tests`.

## Global Constraints

- Nothing is dropped or shortened: every transcript passing `report.is_found` with ≥1 located passage appears, every passage in full. No thresholds, caps or summaries.
- A transcript is never split across volumes. Ranking stays global (`build_report` order).
- Budget: `DOCX_VOLUME_WORDS`, default `150000` words of excerpt text per volume, clamped to ≥ `1000` in `Config.validated()`. CLI `--docx-volume-words` overrides it on `cli.py` and `word_report.py`.
- Layout: fits in one volume → `docx/<theme_id>.docx` exactly as today. Else `docx/<theme_id>/<theme_id>-index.docx` and `docx/<theme_id>/<theme_id>-vol-NN.docx`, and the flat file is not written.
- Housekeeping removes only this tool's output; other subdirectories, non-docx files, `~$…` and `.~lock.*#` lock files are never touched. `test_stale_theme_docx_removed_on_rerun` must keep passing unchanged.
- Marker matching: whole words only, using `(?<!\w)` and `(?!\w)` anchors (not `\b`). No checkpoint or cache invalidation.
- `report.html`, the notebook and `run_report.py` are not changed.

## Review Focus

1. **A theme with no matched transcripts** must still produce the flat `docx/<theme_id>.docx` with the "No transcripts contained relevant excerpts" paragraph, never an empty directory. Test in Task 4.
2. **One transcript larger than the budget** gets a volume of its own and is printed in full, not skipped or truncated. Test in Task 3.
3. **A theme that was multi-volume last run and fits one file now (or the reverse)** must leave exactly one layout on disk afterwards. Tests in Task 4.
4. **A marker that ends with punctuation** (`"waves,"`) or sits at the very start or end of the text must still match after the word-boundary change. Tests in Task 1.
5. **`DOCX_VOLUME_WORDS=0` or a negative `--docx-volume-words`** must clamp to 1000, not loop forever or put every transcript in its own volume. Test in Task 2.

---

### Task 1: Whole-word marker matching

**Files:**
- Modify: `extraction.py:24-31`
- Modify: `README.md` (the "Regenerating the reports" section, lines ~222-234)
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes: `_find_marker(text: str, marker: str, start_pos: int = 0) -> tuple[int, int] | None` (unchanged signature).
- Produces: same function; a match may no longer start or end inside a word.

- [ ] **Step 1: Write the failing tests**

```python
def test_find_marker_does_not_match_inside_a_word():
    text = "The pastor is in great danger, he said."
    # "r is in great danger" used to match inside "pastor"
    assert _find_marker(text, "r is in great danger") is None


def test_find_marker_does_not_match_a_prefix_of_a_longer_word():
    text = "He spoke of graceful living."
    assert _find_marker(text, "grace") is None


def test_find_marker_still_matches_at_text_start_end_and_next_to_punctuation():
    text = "Grace comes in waves, not lines"
    assert _find_marker(text, "Grace comes") == (0, 11)
    assert _find_marker(text, "waves,") == (15, 21)
    assert _find_marker(text, "not lines") == (22, 31)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_extraction.py -v`
Expected: the first two new tests FAIL (a span is returned instead of `None`); the third passes already.

- [ ] **Step 3: Anchor the pattern in `_find_marker`**

In `extraction.py`, build the pattern as `r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)"`. Do not use `\b`: a marker token may begin or end with punctuation, and `\b` would then demand a word character on the wrong side. Update the docstring with one sentence: the marker must start and end on a word boundary so it cannot match inside a word.

- [ ] **Step 4: Run the whole extraction test file**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_extraction.py -v`
Expected: all PASS, including the five pre-existing `_find_marker` tests.

- [ ] **Step 5: Note the reach in the README**

Under "Regenerating the reports" add one paragraph: passage boundaries are fixed at analysis time and only the extracted text is cached, so a change to passage extraction (such as the whole-word marker fix) shows up in transcripts analysed after it. To apply it to a finished run, delete that run's `.checkpoint.jsonl` and re-run, which pays the model cost again.

- [ ] **Step 6: Commit**

```bash
git add transcript_theme_analyzer/extraction.py transcript_theme_analyzer/tests/test_extraction.py transcript_theme_analyzer/README.md
git commit -m "fix: passage markers match whole words only

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `DOCX_VOLUME_WORDS` in `Config`

**Files:**
- Modify: `config.py:92-95` (add the field after `max_themes`), `config.py:122-132` (clamp in `validated()`)
- Modify: `.env.example:45-47` (add after `MAX_THEMES`)
- Modify: `README.md:66-67` (env block, add after `MAX_THEMES`)
- Test: `tests/test_config.py` (create)

**Interfaces:**
- Produces: `Config.docx_volume_words: int`, default `150000` from env `DOCX_VOLUME_WORDS`; `Config.validated()` clamps it to `max(1000, value)`. Later tasks read `config.docx_volume_words`.

- [ ] **Step 1: Write the failing tests**

```python
from transcript_theme_analyzer.config import Config


def test_docx_volume_words_defaults_to_150000(monkeypatch):
    monkeypatch.delenv("DOCX_VOLUME_WORDS", raising=False)
    assert Config().docx_volume_words == 150_000


def test_docx_volume_words_reads_env(monkeypatch):
    monkeypatch.setenv("DOCX_VOLUME_WORDS", "20000")
    assert Config().docx_volume_words == 20_000


def test_validated_clamps_docx_volume_words_to_1000():
    assert Config(docx_volume_words=0).validated().docx_volume_words == 1000
    assert Config(docx_volume_words=-5).validated().docx_volume_words == 1000
    assert Config(docx_volume_words=1000).validated().docx_volume_words == 1000
    assert Config(docx_volume_words=5000).validated().docx_volume_words == 5000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_config.py -v`
Expected: FAIL with `ValidationError` / `AttributeError` on `docx_volume_words`.

- [ ] **Step 3: Add the field and the clamp**

In `config.py` add, after `max_themes`, with a two-line comment ("Excerpt words per Word volume; a theme over this is split into numbered volumes plus an index"):

```python
docx_volume_words: int = Field(
    default_factory=lambda: int(os.environ.get("DOCX_VOLUME_WORDS", "150000"))
)
```

In `validated()` add `self.docx_volume_words = max(1000, self.docx_volume_words)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Document the variable**

`.env.example`, after the `MAX_THEMES` block:

```
# Words of excerpt text per Word volume. A theme whose excerpts exceed this is
# written as docx/<theme_id>/ with an index and numbered volumes instead of one
# file. --docx-volume-words overrides it for a single run. Minimum 1000.
DOCX_VOLUME_WORDS=150000
```

README env block (line ~67): add `DOCX_VOLUME_WORDS=150000` with a one-line comment "Excerpt words per Word volume (default 150000, about 300 pages)".

- [ ] **Step 6: Commit**

```bash
git add transcript_theme_analyzer/config.py transcript_theme_analyzer/tests/test_config.py .env.example transcript_theme_analyzer/README.md
git commit -m "feat: DOCX_VOLUME_WORDS budget for per-theme Word volumes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Volume planning (pure functions)

**Files:**
- Modify: `word_report.py` (add after `_matching_transcripts`, line ~120)
- Test: `tests/test_word_report.py`

**Interfaces:**
- Consumes: `TranscriptResult` from `report.py` (`best_model.locations` is a list of dicts with an `"excerpt"` key); `_matching_transcripts(report, min_score)` already returns the matched transcripts in ranked order.
- Produces:
  - `DEFAULT_VOLUME_WORDS: int = 150_000` (module constant).
  - `_excerpt_words(t: TranscriptResult) -> int`: whitespace-split word count of every `excerpt` in `t.best_model.locations`. Headings, titles, timestamps and speakers do not count.
  - `_plan_volumes(matched: list[TranscriptResult], budget: int) -> list[list[TranscriptResult]]`: greedy fill in the given order. A transcript joins the current volume if the volume's word total including it is ≤ `budget`; otherwise a new volume starts with it. An empty input returns `[]`. A transcript larger than the budget occupies a volume alone.

- [ ] **Step 1: Write the failing tests**

Use the existing `make_success`, `build_report` and `_matching_transcripts` helpers. `words(n)` below is `" ".join(["w"] * n)`.

```python
from transcript_theme_analyzer.word_report import _excerpt_words, _matching_transcripts, _plan_volumes


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_word_report.py -k "excerpt_words or plan_volumes" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `DEFAULT_VOLUME_WORDS`, `_excerpt_words` and `_plan_volumes` in `word_report.py`**

Signatures as in Interfaces. `_plan_volumes` is a single pass keeping `current: list`, `current_words: int`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_word_report.py -v`
Expected: PASS (new and existing).

- [ ] **Step 5: Commit**

```bash
git add transcript_theme_analyzer/word_report.py transcript_theme_analyzer/tests/test_word_report.py
git commit -m "feat: plan Word volumes from ranked excerpt word counts

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Volume and index documents, file layout, housekeeping

**Files:**
- Modify: `word_report.py:1-17` (module docstring), `word_report.py:123-178` (overview table and `build_docx`), `word_report.py:199-221` (`write_theme_docxs`)
- Test: `tests/test_word_report.py`

**Interfaces:**
- Consumes: Task 3's `_plan_volumes`, `_excerpt_words`, `DEFAULT_VOLUME_WORDS`; existing `build_docx`, `_add_overview_table`, `_add_bookmark`, `_add_internal_link`, `_add_excerpt`, `_add_horizontal_rule`.
- Produces:
  - `build_docx(report, min_score=None) -> Document`: unchanged behaviour and output (the single-file case).
  - `build_volume_docx(report: BatchReport, volume: list[TranscriptResult], volume_no: int, volume_count: int) -> Document`: Heading 1 = `report.theme`; paragraph `Volume {volume_no} of {volume_count}`; paragraph `Generated {report.generated_at}`; the two-column overview table (same `_add_overview_table` look) listing only `volume`'s transcripts, every name linked; then one section per transcript as in `build_docx`. Bookmark name and id are `t{i}` / `i` where `i = report.transcripts.index(t)` (the global rank), so names agree with the index numbering.
  - `build_index_docx(report: BatchReport, volume_of: dict[int, int]) -> Document` (`volume_of` maps `id(t)` → 1-based volume number for matched transcripts): Heading 1 = `report.theme`; paragraph `Index of {n} volumes`; paragraph `Generated {report.generated_at}`; a four-column table with header `Transcript`, `Relevance Score`, `Passages`, `Volume` and one row per `report.transcripts` entry in order. Names are plain text (no links, cross-file links are out of scope); failed transcripts get the ` (failed)` suffix and red `FAILED` score as today. `Passages` = `len(t.best_model.locations)` or `—` when failed. `Volume` = the number from `volume_of`, else `—`.
  - `write_theme_docxs(themes, transcripts, out_dir, volume_words: int = DEFAULT_VOLUME_WORDS) -> list[str]`: returns every written path, in theme order; within a multi-volume theme the index path first, then volumes in order. Layout and housekeeping per Global Constraints. Volume numbers are zero-padded to `max(2, len(str(volume_count)))` digits (so `-vol-01` … `-vol-12`, and `-vol-001` once past 99). *Plan decision*: the spec's example shows two digits, so two is the floor.
  - `_remove_stale(docx_dir: str, keep: set[str]) -> None` (`keep` holds `os.path.abspath` of every path written this run; compare with `os.path.abspath` of each candidate): for each entry directly in `docx_dir`: a `*.docx` file not starting with `~$` whose absolute path is not in `keep` is removed (today's rule); a directory `d` that contains `d/<d>-index.docx` has every `<d>-*.docx` file not in `keep` removed and is then removed if empty (`os.rmdir` inside `try/except OSError`); anything else is left alone.

- [ ] **Step 1: Write the failing tests for the two new builders**

```python
from transcript_theme_analyzer.word_report import build_index_docx, build_volume_docx


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_word_report.py -k "volume_docx or index_docx" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_volume_docx` and `build_index_docx`**

Factor the per-transcript section loop of `build_docx` (lines 165-176) into `_add_transcript_sections(document, report, transcripts: list[TranscriptResult]) -> None`, which looks up the global rank with `report.transcripts.index(t)` for the bookmark, and call it from both `build_docx` and `build_volume_docx`. Give `_add_overview_table` an optional `transcripts: list[TranscriptResult] | None = None` parameter defaulting to `report.transcripts`, so the volume table reuses it with the volume's list. Write `_add_index_table(document, report, volume_of)` for the four-column table. `build_docx`'s output must not change.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_word_report.py -v`
Expected: PASS, including every pre-existing `build_docx` and overview-table test unchanged.

- [ ] **Step 5: Write the failing tests for layout and housekeeping**

```python
from transcript_theme_analyzer.themes import Theme
from transcript_theme_analyzer.word_report import write_theme_docxs


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
    # Only excerpt text contains the letter "w" here ("Grace", "Volume", "Generated",
    # "Transcript: epN" and the title "t" do not), so the count is the printed word total.
    printed = " ".join(p.text for path in paths[1:] for p in Document(path).paragraphs)
    assert printed.count("w") == 1400


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
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_word_report.py -k "budget or verbatim or no_matches_is_a_flat or switching or fewer_volumes or stale_volume or foreign_file" -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'volume_words'`).

- [ ] **Step 7: Implement the layout in `write_theme_docxs` and `_remove_stale`**

Per theme: `report = build_report(theme.name, project_theme(transcripts, theme))`; `matched = _matching_transcripts(report, None)`; `plan = _plan_volumes(matched, volume_words)`. If `len(plan) <= 1`: write `build_docx(report)` to `docx/<theme.id>.docx`. Else: `volume_of = {id(t): k + 1 for k, vol in enumerate(plan) for t in vol}`; write the index and each volume into `docx/<theme.id>/` using the names in Interfaces. Collect absolute written paths into `keep`, then call `_remove_stale(docx_dir, keep)`. Return the paths. Update the module docstring (lines 1-17) to describe both layouts in two sentences.

- [ ] **Step 8: Run the whole suite**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`
Expected: all PASS, including `test_stale_theme_docx_removed_on_rerun`, `test_write_theme_docxs_one_file_per_theme` and `test_found_rule_matches_between_docx_and_html` unchanged.

- [ ] **Step 9: Commit**

```bash
git add transcript_theme_analyzer/word_report.py transcript_theme_analyzer/tests/test_word_report.py
git commit -m "feat: split oversized theme Word reports into an index and numbered volumes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `--docx-volume-words` on both CLIs, docs

**Files:**
- Modify: `cli.py:70-75` (add the flag after `--max-themes`), `cli.py:366-381` (`_write_outputs`), `cli.py:436-449` and `:537` (`run_batch`), `cli.py:541-558` (`write_report_from_results`), `cli.py:561-582` (`salvage_report`), `cli.py:602-612` (override in `main`), `cli.py:651` and `:674` (pass the value)
- Modify: `word_report.py:224-245` (`parse_args`, `main`)
- Modify: `README.md` (the `docx/<theme_id>.docx` bullet at ~184-199, the "Reusing an --out-dir" paragraph at ~201, "Regenerating the reports" at ~222-234)
- Test: `tests/test_cli.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: `Config.docx_volume_words` (Task 2); `write_theme_docxs(..., volume_words=)` (Task 4).
- Produces:
  - `cli.parse_args`: `--docx-volume-words` (`type=int`, `default=None`, help `Excerpt words per Word volume before a theme is split into numbered volumes (default: DOCX_VOLUME_WORDS or 150000)`).
  - `_write_outputs(themes, raw_by_transcript, out_dir, volume_words: int = DEFAULT_VOLUME_WORDS)`; `write_report_from_results(themes, items, results_by_key, out_dir, total, volume_words: int = DEFAULT_VOLUME_WORDS)`; `salvage_report(transcript_paths, themes, models, out_dir, base_dir, volume_words: int = DEFAULT_VOLUME_WORDS)`. `run_batch` passes `config.docx_volume_words`; `main` passes `config.docx_volume_words` to `salvage_report` and to the single-transcript `_write_outputs` call.
  - `word_report.parse_args`: same `--docx-volume-words` flag. `word_report.main` calls `load_config()`, applies the override when given, calls `validated()`, and passes `config.docx_volume_words` to `write_theme_docxs`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli.py`, extend `test_theme_is_repeatable_and_optional` or add:

```python
def test_docx_volume_words_flag_parses_and_defaults_to_none():
    assert parse_args(["--transcript", "t.txt", "--theme", "G"]).docx_volume_words is None
    assert parse_args(["--transcript", "t.txt", "--theme", "G", "--docx-volume-words", "2000"]).docx_volume_words == 2000
```

In `tests/test_cache.py`, after `test_word_report_main_regenerates_every_theme_docx`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_cli.py transcript_theme_analyzer/tests/test_cache.py -k "docx_volume_words" -v`
Expected: FAIL (`unrecognized arguments: --docx-volume-words` / `AttributeError`).

- [ ] **Step 3: Add the flag and thread the value through `cli.py`**

Signatures as in Interfaces. In `main`, after the `--max-themes` override: `if args.docx_volume_words is not None: config.docx_volume_words = args.docx_volume_words` (before `config.validated()`). Import `DEFAULT_VOLUME_WORDS` from `.word_report`.

- [ ] **Step 4: Add the flag to `word_report.parse_args` and the config load to `word_report.main`**

Import `load_config` from `.config`. Update the parser description to "Regenerate every theme's Word report(s) under docx/ from the internal analysis cache".

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`
Expected: all PASS.

- [ ] **Step 6: Update the README output section**

Rewrite the `docx/<theme_id>.docx` bullet (~184-199) so it says: one Word report per theme; when a theme's excerpt text exceeds `DOCX_VOLUME_WORDS` (default 150,000 words, about 300 pages) it is written instead as `docx/<theme_id>/<theme_id>-index.docx` plus `docx/<theme_id>/<theme_id>-vol-01.docx`, `-vol-02.docx`, …; the index lists every transcript with its score, passage count and volume number; each volume repeats the title, carries a "Volume N of M" line and its own linked overview table; transcripts are never split and ranking stays global; nothing is trimmed. Update the "Reusing an --out-dir" paragraph (~201) to say stale theme files and the tool's own volume folders are removed, and other files under `docx/` are left alone. In "Regenerating the reports" add `[--docx-volume-words N]` to the `word_report` command line and say it writes the same layout as the run.

- [ ] **Step 7: Manual check on the trial data (optional, no commit)**

If a `.raw_results.json` from the 20-transcript trial is at hand:

```bash
.venv/bin/python -m transcript_theme_analyzer.word_report --cache <path>/.raw_results.json --out-dir /tmp/volcheck --docx-volume-words 20000
```

Expected: "The will and plan of God" appears as a directory with an index and several volumes; a narrow theme stays a single file; every printed path opens in LibreOffice.

- [ ] **Step 8: Commit**

```bash
git add transcript_theme_analyzer/cli.py transcript_theme_analyzer/word_report.py transcript_theme_analyzer/tests/test_cli.py transcript_theme_analyzer/tests/test_cache.py transcript_theme_analyzer/README.md
git commit -m "feat: --docx-volume-words on the analysis CLI and the docx regenerator

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
