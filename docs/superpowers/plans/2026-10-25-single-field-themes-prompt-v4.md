# Single-Field Themes and Prompt v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a theme one field of any length, give every theme an automatic short label for file names and headings, and move the analysis prompt to v4 with score bands and teaching-point passages.

**Architecture:** `Theme` gains a `text` field (the full theme, what the model reads) and keeps `name` as the short label derived from the first six words; `description` is removed everywhere. The label is what reports show in headings, headers and file names, and the text appears once under it. `prompts.py` ships a rewritten system prompt under `PROMPT_VERSION = "multi-v4"`, which the existing checkpoint signature already treats as a new question. Old result caches, recognisable by a `description` key, are rejected with a one-line message.

**Tech Stack:** Python ≥3.10, pydantic 2, python-docx, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-25-single-field-themes-prompt-v4-design.md`

All paths below are relative to `transcript_theme_analyzer/` unless they start with `docs/`, `notebooks/` or `.env`. Run tests from the repo root with `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`.

## Global Constraints

- A theme is one text, entered once: `--theme "..."`, a one-column `theme` CSV, one string in the notebook's `THEMES` list. `description` is removed from `Theme`, the CSV reader, the prompt block, the checkpoint signature, the notebook text and the README.
- A CSV with a `description` column is rejected before any API call. Exact copy: `{path} has a 'description' column. Put the full theme text in the 'theme' column instead; a theme may be a phrase or a whole paragraph.`
- Theme text whitespace is collapsed to one line (existing behaviour).
- Label = first six words of the text, cut at a word boundary, trailing punctuation removed; a text of six words or fewer is its own label. Labels are unique within a run (` (2)`, ` (3)` suffix). The id is the slug of the unique label.
- The prompt receives the full text: theme block line `- id: <id> | <full text>`. Reports show the label, with the full text once beneath it (docx) or on hover plus once in the per-theme table (HTML). Checkpoint signature uses `[id, text]`.
- `PROMPT_VERSION = "multi-v4"`. The v4 prompt content is fixed by spec section 3; wording is the builder's.
- An old `.raw_results.json` (theme entries carrying `description`) is rejected by both regenerators with exit code 1 and the message `This cache was written before prompt v4; re-run the analysis.`
- Extraction, chunking, aggregation, docx volumes and the "found" rule are untouched.

## Review Focus

1. **Two paragraph-length themes that share their first six words** must get distinct labels, ids and Word files, not overwrite each other. Test in Task 1.
2. **A short theme (six words or fewer)** must not print its text twice: no paragraph under the docx heading, no hover text in the HTML. Tests in Task 3.
3. **A theme text containing quotes or angle brackets** must be escaped in the HTML `title` attribute and the per-theme table. Test in Task 3.
4. **Regenerating from a pre-v4 cache** must print the one-line message and exit 1, not a `TypeError` traceback. Test in Task 2.
5. **The model labels an entry by the theme's full text or label instead of its id.** The repair step must still map it to the right theme. Test in Task 2.

---

### Task 1: `Theme.text`, the derived label, one-column input

**Files:**
- Modify: `themes.py` (whole module: `Theme`, `slugify`, `build_themes`, `read_themes_file`)
- Modify: `cli.py:58-69` (`--theme` and `--themes-file` help), `eval/compare_models.py:30` (help)
- Test: `tests/test_themes.py`

**Interfaces:**
- Produces:
  - `Theme(id: str, name: str, text: str = "")`, frozen dataclass. `name` is the short label. `__post_init__` sets `text = name` when `text` is empty (so `Theme("a", "Alpha")` keeps working in every existing test and `text` is never empty). `description` no longer exists.
  - `derive_label(text: str, max_words: int = 6) -> str`: first `max_words` whitespace-split words joined by single spaces, then `rstrip` of `.,;:!?—–-"'` characters; a text of `max_words` words or fewer returns unchanged (after the same strip).
  - `build_themes(cli_texts: list[str], file_texts: list[str]) -> list[Theme]`: CLI texts first, then file texts. Each text is whitespace-collapsed; empty raises `ThemeInputError("A theme is empty.")`; duplicate texts (casefold) raise the existing "listed more than once" error. Label = `derive_label(text)`, made unique by appending ` (2)`, ` (3)`… when an earlier theme already has that label (casefold compare). Id = `slugify(label)`, made unique as today (`-2` suffix) in case two distinct labels slug the same.
  - `read_themes_file(path: str) -> list[str]`: the `theme` column's non-blank cells, in order. A `description` header (case-insensitive) raises `ThemeInputError` with the exact copy in Global Constraints. A missing `theme` column raises as today, message `{path} needs a 'theme' column.`

- [ ] **Step 1: Rewrite the description-era tests and add the label tests**

In `tests/test_themes.py` delete `test_cli_names_come_before_file_rows_and_descriptions_kept`, `test_read_themes_file_skips_blank_rows_and_keeps_descriptions`, `test_read_themes_file_without_description_column`; change `test_internal_whitespace_and_newlines_collapsed` to `build_themes(["the  glory\nof God"], ["Grace is unmerited\r\n\n favour\tgiven freely to all"])` asserting `themes[0].text == "the glory of God"`, `themes[1].text == "Grace is unmerited favour given freely to all"`, `themes[1].name == "Grace is unmerited favour given"`, and two prompt-block lines. Add:

```python
from transcript_theme_analyzer.themes import Theme, derive_label

LONG = ("Teachings about God's will for an individual's personal decisions, life direction, "
        "and calling. Exclude discussions of God's sovereign plan for historical events.")


def test_theme_text_defaults_to_name():
    t = Theme("a", "Alpha")
    assert t.text == "Alpha"
    assert not hasattr(t, "description")


def test_derive_label_first_six_words_without_trailing_punctuation():
    assert derive_label(LONG) == "Teachings about God's will for an"
    assert derive_label("will of God") == "will of God"
    assert derive_label("Grace, mercy, and peace to you all,") == "Grace, mercy, and peace to you"
    assert derive_label("Faith. Hope. Love.") == "Faith. Hope. Love"


def test_long_theme_gets_label_and_short_id():
    (t,) = build_themes([LONG], [])
    assert t.text == LONG
    assert t.name == "Teachings about God's will for an"
    assert t.id == "teachings-about-gods-will-for-an"


def test_themes_sharing_first_six_words_get_distinct_labels_and_ids():
    a, b = build_themes([LONG, LONG.replace("Exclude", "Include")], [])
    assert (a.name, b.name) == ("Teachings about God's will for an", "Teachings about God's will for an (2)")
    assert (a.id, b.id) == ("teachings-about-gods-will-for-an", "teachings-about-gods-will-for-an-2")


def test_cli_texts_come_before_file_texts():
    assert [t.text for t in build_themes(["Zeta"], ["Alpha"])] == ["Zeta", "Alpha"]


def test_read_themes_file_single_column_skips_blank_rows(tmp_path):
    p = tmp_path / "themes.csv"
    p.write_text('Theme\n"will of God"\n\n"Teachings about God\'s will,\nacross lines"\n', encoding="utf-8")
    assert read_themes_file(str(p)) == ["will of God", "Teachings about God's will,\nacross lines"]


def test_read_themes_file_rejects_description_column(tmp_path):
    p = tmp_path / "themes.csv"
    p.write_text("theme,description\nGrace,unmerited favour\n", encoding="utf-8")
    with pytest.raises(ThemeInputError) as exc:
        read_themes_file(str(p))
    assert str(exc.value) == (
        f"{p} has a 'description' column. Put the full theme text in the 'theme' column "
        "instead; a theme may be a phrase or a whole paragraph."
    )
```

- [ ] **Step 2: Run the file to verify the new tests fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_themes.py -q`
Expected: FAIL on `ImportError: derive_label` (collection error).

- [ ] **Step 3: Implement `Theme.text`, `derive_label`, `build_themes`, `read_themes_file` in `themes.py`**

Signatures as in Interfaces. Use `object.__setattr__(self, "text", self.name)` in `__post_init__` since the dataclass is frozen. Update the module docstring's first line to say a theme is one text with a derived label.

- [ ] **Step 4: Run the themes tests, then the full suite, and fix the callers the suite names**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_themes.py -q`
Expected: PASS.
Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q 2>&1 | tail -15`
Expected: failures only in tests that construct `Theme` with a third positional description, call `build_themes` with tuples, or write two-column CSVs. Fix each at its site, in this task, so the suite is green before commit:
- `tests/test_cache.py:10`: `Theme("glory", "the glory of God")`.
- `tests/test_batch.py:736-745` (`test_changed_description_invalidates_checkpoint`): rename to `test_changed_theme_text_invalidates_checkpoint`, use `Theme("a", "Alpha", "Alpha, first meaning")` then `Theme("a", "Alpha", "Alpha, second meaning")`, message `"a new theme text is a different question"`.
- `tests/test_batch.py:879-895` (`test_themes_file_and_flags_combine`): CSV becomes `"theme\nForgiveness is letting go of wrongs done to us\nHope\n"`; assert ids `["grace", "forgiveness-is-letting-go-of-wrongs", "hope"]` and `themes[1].text == "Forgiveness is letting go of wrongs done to us"`.
- `tests/test_checkpoint.py:122-137`: in `test_signature_ignores_order` (line ~122) use `Theme("a", "Alpha", "Alpha x")`; delete `test_signature_covers_description_name_id_and_prompt_version` outright (Task 2 writes its replacement).
- `tests/test_multi_theme_schema.py:27` and `:84`: `Theme("forgiveness", "Forgiveness", "Forgiveness — letting go")`; the expected block becomes `"- id: grace | Grace\n- id: forgiveness | Forgiveness — letting go"` (passes once Task 2 switches the block to `text`; until then mark the assertion `pytest.mark.xfail(strict=True, reason="Task 2 switches the theme block to text")`).

- [ ] **Step 5: Update the help strings**

`cli.py` `--theme`: `Theme to analyze for: a short label or a full description, in one string. Repeat for several themes`. `--themes-file`: `CSV with one 'theme' column (a phrase or a whole paragraph per row). Combines with --theme (those come first)`. Same text for `eval/compare_models.py:30`.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`
Expected: all PASS except the one xfail.

- [ ] **Step 7: Commit**

```bash
git add transcript_theme_analyzer/themes.py transcript_theme_analyzer/cli.py transcript_theme_analyzer/eval/compare_models.py transcript_theme_analyzer/tests/
git commit -m "feat: a theme is one text with a derived six-word label

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The full text reaches the model, the signature and the cache check

**Files:**
- Modify: `prompts.py:200-209` (`format_theme_block`), `analyzer.py:452-456` (`_repair_multi_theme` lookup), `analyzer.py:841` (synthesis section header), `checkpoint.py:40-52` (`_signature`), `cache.py` (`read_cache`, new message constant)
- Test: `tests/test_multi_theme_schema.py`, `tests/test_checkpoint.py`, `tests/test_cache.py`, `tests/test_analyzer.py`

**Interfaces:**
- Consumes: `Theme.text`, `Theme.name` (Task 1).
- Produces:
  - `format_theme_block(themes) -> str`: one line per theme, `- id: {theme.id} | {theme.text}`.
  - `_signature(themes, models)`: `"themes": sorted([t.id, t.text] for t in themes)`.
  - `cache.PRE_V4_MESSAGE = "This cache was written before prompt v4; re-run the analysis."`; `read_cache` raises `ValueError(PRE_V4_MESSAGE)` when any theme entry has a `description` key. Both regenerators already catch `ValueError` and exit 1.
  - `_repair_multi_theme`: the lookup also accepts `theme.text.casefold()` (keep `theme.name.casefold()`).

- [ ] **Step 1: Write the failing tests**

`tests/test_multi_theme_schema.py`: remove the xfail from Step 4 of Task 1 so the `format_theme_block` assertion is live. `tests/test_checkpoint.py`: replace `test_signature_covers_description_name_id_and_prompt_version` with:

```python
def test_signature_covers_text_id_and_prompt_version(monkeypatch):
    base = _signature([Theme("a", "Alpha", "Alpha x")], MODELS)
    assert base != _signature([Theme("a", "Alpha", "Alpha y")], MODELS)
    assert base != _signature([Theme("a2", "Alpha", "Alpha x")], MODELS)
    assert base == _signature([Theme("a", "Different label", "Alpha x")], MODELS)
    assert json.loads(base) == {
        "themes": [["a", "Alpha x"]],
        "models": sorted(MODELS),
        "prompt": prompts.PROMPT_VERSION,
    }
    monkeypatch.setattr(prompts, "PROMPT_VERSION", "other")
    assert base != _signature([Theme("a", "Alpha", "Alpha x")], MODELS)
```

`tests/test_cache.py`:

```python
@pytest.mark.parametrize("module", [run_report, word_report])
def test_regen_mains_exit_1_on_pre_v4_cache(tmp_path, capsys, module):
    path = tmp_path / ".raw_results.json"
    path.write_text(json.dumps({
        "themes": [{"id": "glory", "name": "the glory of God", "description": "His majesty"}],
        "results": {},
    }), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        module.main(["--cache", str(path)])
    assert exc.value.code == 1
    assert capsys.readouterr().err.strip() == "This cache was written before prompt v4; re-run the analysis."


def test_round_trip_keeps_theme_text(tmp_path):
    themes = [Theme("glory", "the glory of God", "the glory of God in creation and redemption")]
    path = write_cache(themes, {"ep1": [("m1", make_success(80))]}, str(tmp_path))
    loaded, _ = read_cache(path)
    assert loaded == themes
    assert loaded[0].text == "the glory of God in creation and redemption"
```

`tests/test_analyzer.py`: next to the existing repair tests, add a case where the model returns `"theme_id": "Alpha, the first letter"` for `Theme("a", "Alpha", "Alpha, the first letter")` and assert the entry is mapped to id `a`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_multi_theme_schema.py transcript_theme_analyzer/tests/test_checkpoint.py transcript_theme_analyzer/tests/test_cache.py transcript_theme_analyzer/tests/test_analyzer.py -q 2>&1 | tail -8`
Expected: FAIL on the block assertion, the signature shape, a `TypeError` from `Theme(**t)` on the pre-v4 cache, and the repair mapping.

- [ ] **Step 3: Implement the four changes**

As in Interfaces. In `analyzer.py:841` use `theme.text` in the synthesis section header. Update the docstring of `_signature` ("a changed theme text changes what the model is asked").

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`
Expected: all PASS, no xfail remaining.

- [ ] **Step 5: Commit**

```bash
git add transcript_theme_analyzer/prompts.py transcript_theme_analyzer/analyzer.py transcript_theme_analyzer/checkpoint.py transcript_theme_analyzer/cache.py transcript_theme_analyzer/tests/
git commit -m "feat: the model, checkpoint and cache use the full theme text

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Reports show the label and the full text once

**Files:**
- Modify: `report.py:55-58` (`BatchReport`), `report.py:79` (`build_report`), `word_report.py` (`build_docx`, `build_volume_docx`, `build_index_docx`, `write_theme_docxs`), `run_report.py:233-240` (per-theme table), `run_report.py:268` (grid header), `run_report.py:471-486` (title, subtitle)
- Test: `tests/test_word_report.py`, `tests/test_run_report.py`

**Interfaces:**
- Consumes: `Theme.name` (label), `Theme.text`.
- Produces:
  - `BatchReport(theme: str, transcripts, generated_at, theme_text: str = "")`; `build_report(theme: str, transcripts, theme_text: str = "") -> BatchReport`.
  - `_add_theme_title(document, report) -> None` in `word_report.py`: Heading 1 = `report.theme`; when `report.theme_text` is non-empty and differs from `report.theme`, one plain paragraph with the full text directly under the heading. Used by `build_docx`, `build_volume_docx` and `build_index_docx` (the volume subtitle and "Generated" lines follow it). `write_theme_docxs` calls `build_report(theme.name, project_theme(...), theme_text=theme.text)`.
  - HTML: grid header `<th data-theme="{id}" title="{text}">{label}</th>` with the `title` attribute only when text differs from the label; per-theme table cell `<td data-label="Theme">{label}<div class="theme-text">{text}</div></td>`, the div only when text differs; page `<title>` and subtitle use labels as today. Add a `.theme-text` rule (smaller, muted) to `EXTRA_CSS`.

- [ ] **Step 1: Write the failing docx tests**

```python
def test_long_theme_prints_label_heading_and_full_text_once(tmp_path):
    long_text = "Teachings about God's will for an individual's personal decisions and calling"
    theme = Theme(id="teachings-about-gods-will-for-an", name="Teachings about God's will for an", text=long_text)
    (path,) = write_theme_docxs([theme], _big_run([100], theme_id=theme.id), str(tmp_path))
    doc = Document(path)
    assert _headings(doc, 1) == ["Teachings about God's will for an"]
    assert doc.paragraphs[1].text == long_text
    assert sum(p.text == long_text for p in doc.paragraphs) == 1


def test_short_theme_has_no_text_paragraph(tmp_path):
    (path,) = write_theme_docxs([Theme(id="grace", name="Grace")], _big_run([100]), str(tmp_path))
    doc = Document(path)
    assert doc.paragraphs[1].text.startswith("Generated ")


def test_volume_and_index_carry_the_full_text(tmp_path):
    theme = Theme(id="grace", name="Grace", text="Grace as unmerited favour, not as a reward")
    paths = write_theme_docxs([theme], _big_run([700, 700]), str(tmp_path), volume_words=1000)
    for path in paths:
        doc = Document(path)
        assert _headings(doc, 1) == ["Grace"]
        assert doc.paragraphs[1].text == theme.text
```

- [ ] **Step 2: Write the failing HTML tests**

In `tests/test_run_report.py`, using the file's `payload` and `entry` helpers and the module-level `A`:

```python
def test_grid_header_shows_label_with_full_text_on_hover():
    long_theme = Theme(id="c", name="Calling and the will", text='Calling and the will of God for an individual\'s "life" <plan>')
    tr = {"t1": [("m1", payload({"a": entry(50), "c": entry(70)}))]}
    html = render_run_html(build_run_report([A, long_theme], tr))
    assert '<th data-theme="a">Alpha</th>' in html
    assert '<th data-theme="c" title="Calling and the will of God for an individual&#x27;s &quot;life&quot; &lt;plan&gt;">Calling and the will</th>' in html
    assert html.count('class="theme-text"') == 1
    assert "<title>Theme Analysis: Alpha, Calling and the will</title>" in html
    assert '"life" <plan>' not in html
```

The expected attribute text assumes `h()` is `html.escape(s, quote=True)` (which renders `'` as `&#x27;`); if `h()` differs, adjust the literal to its output, keeping the assertion that the raw quotes and angle brackets never appear unescaped.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_word_report.py transcript_theme_analyzer/tests/test_run_report.py -k "full_text or text_paragraph or hover" -q`
Expected: FAIL (`TypeError: unexpected keyword 'text'` is not possible after Task 1, so expect assertion failures on the missing paragraph and the missing `title` attribute).

- [ ] **Step 4: Implement**

As in Interfaces. Keep `build_docx`'s existing paragraph order for the short-theme case: heading, "Generated", overview.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add transcript_theme_analyzer/report.py transcript_theme_analyzer/word_report.py transcript_theme_analyzer/run_report.py transcript_theme_analyzer/tests/
git commit -m "feat: reports head each theme by its label and print the full text once

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Analysis prompt v4

**Files:**
- Modify: `prompts.py:6-137` (version and analysis system prompt), `prompts.py:139-179` (synthesis prompt constant name only), `analyzer.py:725,799,852` (constant names)
- Test: `tests/test_multi_theme_schema.py:97-100`, `tests/test_analyzer.py:177,187`, `tests/test_batch.py:102`

**Interfaces:**
- Produces: `PROMPT_VERSION = "multi-v4"`; constants renamed `MULTI_THEME_ANALYSIS_SYSTEM_PROMPT` and `MULTI_THEME_SYNTHESIS_SYSTEM_PROMPT` (the version lives in `PROMPT_VERSION`, not the name). Content of the analysis prompt per spec section 3.

- [ ] **Step 1: Write the failing tests**

Replace the version/content test in `tests/test_multi_theme_schema.py`:

```python
def test_prompt_v4_content():
    p = prompts.MULTI_THEME_ANALYSIS_SYSTEM_PROMPT
    assert prompts.PROMPT_VERSION == "multi-v4"
    # the theme text is the definition
    assert "inclusion" in p and "exclusion" in p
    # score bands
    for band in ("85", "60", "30", "1", "0"):
        assert band in p
    # teaching-point passages
    assert "teaching point" in p
    assert "entire segment" in p
    assert "Understanding the Will of God" in p
    # whole-word markers
    assert "complete words" in p
    # two scored examples plus the absent case
    assert p.count('"relevance_score":') == 3
    assert '"explicitness": "tangential"' in p
    # the format section states each rule once
    assert p.lower().count("never omit") <= 1
    assert '"themes"' in prompts.MULTI_THEME_SYNTHESIS_SYSTEM_PROMPT
```

Update the three test references to the old `_V2` names.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_multi_theme_schema.py -q`
Expected: FAIL with `AttributeError: MULTI_THEME_ANALYSIS_SYSTEM_PROMPT`.

- [ ] **Step 3: Write the v4 prompt**

Rename the constants and bump the version. Rewrite the analysis system prompt to contain, in this order: role (shortened), task with the theme-as-definition paragraph, the explicitness distinction, the score bands with the "most segments fall low" steer, the passage section (the six bullets of spec section 3 in prose, the six-title example, the "entire segment" warning, markers on complete words with exact punctuation), the output format with three examples (one explicit at a high score, one tangential at a low score, one absent at 0), and a rules list where each rule appears once. Keep the "THEMES list" and "more than one theme" phrases, since existing tests assert them. Target roughly two-thirds of the current length.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add transcript_theme_analyzer/prompts.py transcript_theme_analyzer/analyzer.py transcript_theme_analyzer/tests/
git commit -m "feat: analysis prompt v4 with score bands and teaching-point passages

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: README, notebook and env example

**Files:**
- Modify: `README.md:4,21,114-130,184-199,262-266`, `notebooks/colab_kaggle_runner.ipynb` (markdown cell lines ~147-162; code cell lines ~181-184 and ~201), `.env.example` (no change needed unless it mentions descriptions; check)
- Test: none beyond the notebook JSON still parsing.

- [ ] **Step 1: README**

In "Running": the CSV example becomes one column with a short theme and a paragraph theme; the paragraph after it says a theme is one text of any length, the first six words become its label (used for the Word file name, headings and the report header), and the full text is what the model reads, so a description with inclusions and exclusions is the main lever for a theme that matches too much or too little. State the `description` column is no longer accepted and what the error says. In "Outputs": the Word document heading is the label with the full text beneath when they differ. In "Resuming": "names and descriptions do" → "the theme texts do". Line 4: `MAX_THEMES` sentence unchanged. Add one line to "Regenerating the reports": caches from before prompt v4 are rejected with a message; re-run.

- [ ] **Step 2: Notebook**

Edit the `.ipynb` as JSON (do not hand-edit the raw file). Markdown cell: replace the sentence about `THEMES_FILE` with a `description` column with: a theme is one string, short or a full paragraph; `THEMES_FILE` is a one-column CSV. Code cell: the `THEMES` comment says `# <-- a short label or a full description, one string each`; the `THEMES_FILE` comment says `(one column: theme)`; the print line becomes `print(f"  - {_t.name}" + (f": {_t.text}" if _t.text != _t.name else ""))`.

Run: `.venv/bin/python -c "import json; json.load(open('notebooks/colab_kaggle_runner.ipynb'))" && echo ok`
Expected: `ok`.

- [ ] **Step 3: Full suite and commit**

Run: `.venv/bin/python -m pytest transcript_theme_analyzer/tests -q`
Expected: all PASS.

```bash
git add transcript_theme_analyzer/README.md notebooks/colab_kaggle_runner.ipynb .env.example
git commit -m "docs: one-field themes with derived labels; prompt v4 notes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Manual check (user-run, paid)**

After merge, re-run the 20-transcript trial with the same five themes and compare against `docx_reports/`: the score spread (expect fewer transcripts at 64+ for the broad theme), the number of titled passages per transcript (expect several per long discussion instead of one), and the log for the line `rejected structured output`, which would mean the trimmed format section needs more weight in a v5.
