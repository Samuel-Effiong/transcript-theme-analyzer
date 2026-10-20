# Multi-Theme Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate up to `MAX_THEMES` themes in one run, with every theme sent to the model in a single call per transcript (or per chunk), and produce one `.docx` per theme plus one HTML page for the run that includes token and cost reporting.

**Architecture:** A theme becomes a list of `Theme` objects throughout. The analyzer sends all of a run's themes in one call and gets back one entry per theme. Each entry is extracted, aggregated and deduplicated exactly as one theme is today. Usage capture is extended with cached tokens and OpenRouter's reported cost, and `pricing.py` turns calls into exact totals and per-theme estimates. Reports work from one payload shape: `report.project_theme` reduces it to today's single-theme shape, so the existing Word builder is reused per theme. A new `run_report.py` renders the run-wide HTML page.

**Tech Stack:** Python ≥3.10, openai SDK 2.x (`openai.NOT_GIVEN`, `extra_body`), pydantic 2, python-docx, httpx, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-20-multi-theme-design.md`

All paths below are relative to `transcript_theme_analyzer/` unless they start with `docs/`, `notebooks/` or `.env`. Run tests from the repo root with `.venv/bin/python -m pytest transcript_theme_analyzer/tests`.

## Global Constraints

- All themes in a run go in **one** call per transcript (single-pass) or per chunk (chunked). No internal grouping.
- `MAX_THEMES` env var, default `5`, clamped to ≥1. CLI `--max-themes` overrides it.
- Over-limit rejection happens **before any API call**. Exact copy: `You entered {n} themes. The limit is {cap} (MAX_THEMES). Remove {n - cap} theme(s) or raise MAX_THEMES.` Exit code `2`.
- No extraction threshold: explicit and tangential passages are both kept.
- No output token cap by default (already done: `max_output_tokens=None` → `max_tokens` omitted).
- Outputs in `--out-dir`: `report.html` (current run only), `docx/<theme_id>.docx` (one per theme), `.raw_results.json` and `.checkpoint.jsonl` (internal).
- Money: OpenRouter's reported `usage.cost` wins. Otherwise tokens × prices. Unknown → shown as `n/a`, **never** `$0.00`.
- Per-theme tokens and cost are labelled **"est."** everywhere they appear. Totals and per-transcript figures are exact.
- Add no new dependencies.

## Review Focus

1. **Model labels a theme by its name, or in different case, instead of its id.** It should still match the right theme, not drop it. Test in Task 2.
2. **Model omits one of the requested themes.** The call is retried. If the theme is still missing after retries, that model run is an error, never a silent score of 0. Tests in Task 2 and Task 6.
3. **Theme names that are unsafe or colliding as filenames** (`Faith/Works`, `God's Love`, `Love` vs `love!`) should give safe, unique `docx/` names. Test in Task 1.
4. **No cost information** (a non-OpenRouter provider with no prices set, or the OpenRouter catalogue unreachable offline) should show `n/a` and the run should continue. Tests in Task 3 and Task 5.
5. **Resuming with the same themes in a different order** should reuse the checkpoint. A changed theme description should invalidate it. Test in Task 7.

---

### Task 1: Theme input and the `MAX_THEMES` limit

**Files:**
- Create: `themes.py`
- Modify: `config.py` (add fields)
- Test: `tests/test_themes.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Theme: id: str; name: str; description: str = ""`
  - `class ThemeInputError(ValueError)`. Its message is the user-facing copy.
  - `def slugify(name: str) -> str`: lowercase, runs of non-`[a-z0-9]` → `-`, stripped of `-`, `"theme"` if empty.
  - `def build_themes(names: list[str], file_rows: list[tuple[str, str]]) -> list[Theme]`. Command-line names come first, then file rows. Ids come from `slugify`. Colliding ids get `-2`, `-3`… Raises `ThemeInputError` when there are no themes (`No themes given. Use --theme or --themes-file.`), when a name is empty, or when a name is duplicated case-insensitively (`Theme {name!r} is listed more than once.`).
  - `def read_themes_file(path: str) -> list[tuple[str, str]]`: a CSV with a header containing `theme` and an optional `description` column (case-insensitive). Blank rows are skipped. A missing `theme` column raises `ThemeInputError`.
  - `def check_theme_limit(themes: list[Theme], max_themes: int) -> None`: raises `ThemeInputError` with the exact Global Constraints copy.
  - `Config.max_themes: int` (env `MAX_THEMES`, default 5, clamped ≥1 in `validated()`).
  - `Config.price_input_per_m`, `Config.price_cached_input_per_m`, `Config.price_output_per_m`: `float | None` (env `LLM_PRICE_INPUT_PER_M`, `LLM_PRICE_CACHED_INPUT_PER_M`, `LLM_PRICE_OUTPUT_PER_M`; unset or empty → `None`). USD per 1M tokens.

- [ ] **Step 1: Write failing tests** in `tests/test_themes.py`:

```python
def test_limit_rejects_over_cap_with_exact_message():
    themes = build_themes([f"t{i}" for i in range(7)], [])
    with pytest.raises(ThemeInputError) as e:
        check_theme_limit(themes, 5)
    assert str(e.value) == ("You entered 7 themes. The limit is 5 (MAX_THEMES). "
                            "Remove 2 theme(s) or raise MAX_THEMES.")

def test_limit_allows_at_and_under_cap():
    check_theme_limit(build_themes(["a", "b", "c", "d", "e"], []), 5)
    check_theme_limit(build_themes(["a"], []), 5)

def test_unsafe_and_colliding_names_get_safe_unique_ids():
    ids = [t.id for t in build_themes(["Faith/Works", "God's Love", "Love", "love!"], [])]
    assert ids == ["faith-works", "god-s-love", "love", "love-2"]
```

Also cover: duplicate names (`"Grace"`, `"grace"`) raise `ThemeInputError`; no themes raises it; `read_themes_file` reads `Theme,Description` headers with a blank row and keeps descriptions; a file without a `theme` column raises; command-line names come before file rows; `MAX_THEMES=0` in env → `load_config().max_themes == 1`; `LLM_PRICE_INPUT_PER_M=""` → `None`.

- [ ] **Step 2:** Run `.venv/bin/python -m pytest transcript_theme_analyzer/tests/test_themes.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement `themes.py` and the `config.py` fields per the Interfaces block. Use `csv.DictReader` for the file.
- [ ] **Step 4:** Run the full suite. Expected: all pass.
- [ ] **Step 5:** Commit: `git commit -m "feat: theme list input and MAX_THEMES limit"`.

---

### Task 2: Multi-theme answer format, prompts and repair

**Files:**
- Modify: `schema.py`, `prompts.py`, `analyzer.py` (repair helpers and the `_call_with_retry` signature only)
- Test: `tests/test_multi_theme_schema.py`

**Interfaces:**
- Consumes: `Theme` (Task 1).
- Produces:
  - `schema.ThemeChunkAnalysis(ChunkAnalysis)`, which adds `theme_id: str`.
  - `schema.MultiThemeChunkAnalysis`, with `themes: list[ThemeChunkAnalysis]`.
  - `schema.ThemeSynthesis`, with `theme_id: str` and `reasoning: str`. `schema.MultiThemeSynthesis`, with `themes: list[ThemeSynthesis]`.
  - `schema.AnalysisResult` gains `theme_id: str` and `explicitness: Literal["explicit","tangential","absent"]`.
  - `prompts.PROMPT_VERSION = "multi-v2"`.
  - `prompts.MULTI_THEME_ANALYSIS_SYSTEM_PROMPT_V2` and `prompts.MULTI_THEME_SYNTHESIS_SYSTEM_PROMPT_V2`.
  - `prompts.format_theme_block(themes: list[Theme]) -> str`. Exactly one line per theme: `- id: {id} | {name}`, plus ` — {description}` when the description is non-empty.
  - `prompts.MULTI_SINGLE_PASS_USER_TEMPLATE` (fields `{theme_block}`, `{transcript}`) and `prompts.MULTI_CHUNK_USER_TEMPLATE` (`{theme_block}`, `{chunk_index}`, `{total_chunks}`, `{char_start}`, `{char_end}`, `{chunk_text}`). The **theme block comes before the transcript**. That makes the system prompt plus the theme block an identical opening for every call in the run, which providers cache.
  - `class MissingThemesError(ValueError)`.
  - `analyzer._repair_multi_theme(parsed, themes: list[Theme]) -> dict`. It applies `_repair_parsed_json(entry, ThemeChunkAnalysis)` to each entry. It maps `theme_id` to a theme by case-insensitive match on id **or** name. It drops unknown and duplicate entries (the first one wins). It returns the entries in input-theme order. It raises `MissingThemesError("missing themes: a, b")` when any theme is absent. The same function also handles `ThemeSynthesis` entries, which carry only `theme_id` and `reasoning`.
  - `_call_with_retry(..., repair: Callable[[Any], Any] | None = None, extra_body: dict | None = None)`. `repair` defaults to `lambda p: _repair_parsed_json(p, schema_cls)`. `extra_body` is passed through to both `create` calls (Task 3 uses it).

Prompt content: adapt `CHUNK_ANALYSIS_SYSTEM_PROMPT_V1` rather than rewriting it. Keep every rule (verbatim markers, titles, mandatory keys, integer score, honest "absent"). Change three things: the task is assessing **each theme in the THEMES list**; the output is `{"themes": [ {theme_id, relevance_score, explicitness, reasoning, locations} ]}` with **exactly one entry per listed id, in list order, absent themes included**; and add the sentence "The same passage may be reported under more than one theme when it genuinely discusses each." Keep the V1 constants until Task 7 deletes the single-theme path.

- [ ] **Step 1: Write failing tests:**

```python
THEMES = [Theme("grace", "Grace"), Theme("forgiveness", "Forgiveness", "letting go")]

def test_theme_matched_by_name_or_case():
    parsed = {"themes": [_entry("GRACE"), _entry("Forgiveness")]}
    out = _repair_multi_theme(parsed, THEMES)
    assert [e["theme_id"] for e in out["themes"]] == ["grace", "forgiveness"]

def test_missing_theme_raises():
    with pytest.raises(MissingThemesError, match="forgiveness"):
        _repair_multi_theme({"themes": [_entry("grace")]}, THEMES)

def test_unknown_and_duplicate_entries_dropped_and_order_restored():
    parsed = {"themes": [_entry("forgiveness"), _entry("hope"), _entry("grace"), _entry("grace", score=1)]}
    out = _repair_multi_theme(parsed, THEMES)
    assert [e["theme_id"] for e in out["themes"]] == ["grace", "forgiveness"]
    assert out["themes"][0]["relevance_score"] != 1

def test_theme_block_format():
    assert format_theme_block(THEMES) == "- id: grace | Grace\n- id: forgiveness | Forgiveness — letting go"

def test_theme_block_precedes_transcript():
    u = MULTI_SINGLE_PASS_USER_TEMPLATE.format(theme_block="TB", transcript="TX")
    assert u.index("TB") < u.index("TX")
```

Also cover: a per-entry fractional score is clamped and rounded, as in the existing repair tests; `_strictify(MultiThemeChunkAnalysis.model_json_schema())` sets `additionalProperties: false` on the nested entry object.

- [ ] **Step 2:** Run the new test file. Expected: FAIL.
- [ ] **Step 3:** Implement per the Interfaces block.
- [ ] **Step 4:** Run the full suite. Expected: all pass. The existing single-theme path is untouched.
- [ ] **Step 5:** Commit: `feat: multi-theme response schema, prompts and repair`.

---

### Task 3: Usage capture and pricing

**Files:**
- Create: `pricing.py`
- Modify: `analyzer.py` (`CallLog`, `RunStats`)
- Test: `tests/test_pricing.py`

**Interfaces:**
- Produces:
  - `CallLog` gains `cached_tokens: int = 0`, `reported_cost: float | None = None`, `theme_ids: list[str] = []` and `theme_output_chars: dict[str, int] = {}`, plus computed fields `cost: float | None = None`, `cost_source: str | None = None` and `discount: float | None = None`.
  - `RunStats.add(label, model, usage, theme_ids=(), theme_output_chars=None)`. `cached_tokens` is read from `usage.prompt_tokens_details.cached_tokens`, falling back to DeepSeek's `usage.prompt_cache_hit_tokens`, else 0. `reported_cost` is read from `usage.cost` (an attribute or a `model_extra` key), else `None`.
  - `RunStats.apply_prices(prices: Prices | None) -> None` fills each call's `cost`, `cost_source` and `discount`.
  - `RunStats.summary() -> dict` with keys `calls`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `total_tokens`, `cost`, `cost_source`, `discount` and `by_call`. `cost` is `None` if any call's cost is `None`. `cost_source` is `"openrouter"`, `"prices"`, `"mixed"` or `None`. `discount` is the sum of known values, or `None` if none are known.
  - `RunStats.theme_usage() -> dict[str, dict]`, giving `{theme_id: {"tokens": int, "cost": float | None}}` (estimates).
  - `@dataclass(frozen=True) class Prices: input_per_m: float; output_per_m: float; cached_input_per_m: float | None = None`.
  - `pricing.fetch_openrouter_prices(models: list[str], base_url: str, timeout: float = 10.0) -> dict[str, Prices]`. It calls `GET {base_url}/models` with httpx and converts `pricing.prompt`, `pricing.completion` and `pricing.input_cache_read` (USD per token, as strings) to per-1M floats. **Any** failure logs a warning and returns `{}`.
  - `pricing.resolve_prices(models: list[str], config: Config) -> dict[str, Prices | None]`. Env prices (when both input and output are set) apply to every model and win. Otherwise it uses OpenRouter catalogue prices when `config.is_openrouter`. Otherwise `None`.
  - `analyzer.openrouter_extra_body(config) -> dict | None` returns `{"usage": {"include": True}}` when `config.is_openrouter`, else `None`.

Rules (these are the decisions, write them as written):
- **Call cost:** if `reported_cost` is set, use it with source `"openrouter"`. Otherwise, with prices, it's `((prompt − cached)·input + cached·(cached_input or input) + completion·output) / 1e6` with source `"prices"`. Otherwise `None`.
- **Discount:** `cached · (input − cached_input) / 1e6` when prices and `cached_input_per_m` are known and `cached > 0`. It's `0.0` when prices are known and `cached == 0`. Otherwise `None`.
- **Per-theme attribution of one call:** split into an input part and an output part. With prices, use the price-based ratio scaled to the call cost. Without prices, use the token ratio (prompt vs completion). Each theme in `theme_ids` gets an **even** share of the input part, plus a share of the output part **in proportion** to `theme_output_chars`, with an even split when the chars are all zero. Tokens are attributed the same way. Shares are rounded only at display time.

- [ ] **Step 1: Write failing tests:**

```python
P = Prices(input_per_m=1.0, output_per_m=4.0, cached_input_per_m=0.1)

def test_reported_cost_wins():
    s = _stats(prompt=1000, completion=100, cached=0, reported_cost=0.5); s.apply_prices(P)
    assert s.summary()["cost"] == 0.5 and s.summary()["cost_source"] == "openrouter"

def test_price_cost_and_discount():
    s = _stats(prompt=1_000_000, completion=0, cached=500_000); s.apply_prices(P)
    assert s.summary()["cost"] == pytest.approx(0.55)
    assert s.summary()["discount"] == pytest.approx(0.45)

def test_no_prices_no_report_is_none_not_zero():
    s = _stats(prompt=10, completion=5, cached=0); s.apply_prices(None)
    assert s.summary()["cost"] is None and s.summary()["discount"] is None

def test_theme_attribution_sums_to_call_and_follows_output_share():
    s = _stats(prompt=900, completion=100, cached=0, theme_ids=["a", "b", "c"],
               theme_output_chars={"a": 300, "b": 100, "c": 0}); s.apply_prices(P)
    tu = s.theme_usage()
    assert sum(v["tokens"] for v in tu.values()) == pytest.approx(1000)
    assert sum(v["cost"] for v in tu.values()) == pytest.approx(s.summary()["cost"])
    assert tu["a"]["tokens"] > tu["b"]["tokens"] > tu["c"]["tokens"]
```

`_stats(**kw)` is a test helper: a `RunStats` with one `add()` call whose usage is a `SimpleNamespace` built from `kw` (`cost` as an attribute when `reported_cost` is given; `prompt_tokens_details.cached_tokens` from `cached`).

Also cover: `fetch_openrouter_prices` with an `httpx.MockTransport` catalogue (string per-token prices converted correctly); a transport that raises → `{}` with no exception; `resolve_prices` gives env precedence; `cached_tokens` is read from both usage shapes.

- [ ] **Step 2:** Run the file. Expected: FAIL.
- [ ] **Step 3:** Implement. Pass `extra_body=openrouter_extra_body(config)` at the existing `_call_with_retry` call sites in `analyzer.py`.
- [ ] **Step 4: Live check (needs the user's OK; costs a fraction of a cent).** Write a script in the scratchpad that makes one tiny OpenRouter call with `extra_body={"usage": {"include": True}}` against `deepseek/deepseek-v4-flash`, prints `resp.usage.model_dump()`, and runs `fetch_openrouter_prices` for that model. Confirm the field names `cost`, `prompt_tokens_details.cached_tokens` and `pricing.input_cache_read`. If they differ, fix the readers and add a test using the observed shape. If OpenRouter reports a cache-discount field directly, prefer it for `discount` when present.
- [ ] **Step 5:** Run the full suite. Expected: all pass. Commit: `feat: capture cached tokens and actual cost; pricing and per-theme attribution`.

---

### Task 4: One Word report per theme

**Files:**
- Modify: `report.py` (add the projection), `word_report.py` (add the multi-theme writer)
- Test: `tests/test_word_report.py` (append)

**Interfaces:**
- Consumes: `Theme`. The **multi-theme payload** below is defined here and produced by Task 6.
- Multi-theme payload (one per model per transcript; errors stay as `{"model": m, "error": str}`):

```python
{"model_used": str, "chunked": bool,
 "themes": {theme_id: {"theme": str, "relevance_score": int, "explicitness": str,
                       "reasoning": str, "locations": [Location dict]}},
 "_meta": {"elapsed_seconds": float, "usage": RunStats.summary(),
           "theme_usage": {theme_id: {"tokens": int, "cost": float | None}}}}
```

- Produces:
  - `report.project_theme(transcripts: dict[str, list[tuple[str, dict]]], theme: Theme) -> dict[str, list[tuple[str, dict]]]`. Converts each payload to today's single-theme shape (`relevance_score`, `reasoning`, `locations`, `model_used`, `chunked`, `_meta.elapsed_seconds`, `_meta.usage.total_tokens` = that theme's estimated tokens). Error payloads pass through. A success payload missing the theme becomes `{"model": m, "error": "no result for theme <id>"}`.
  - `word_report.write_theme_docxs(themes: list[Theme], transcripts, out_dir: str) -> list[str]`. Writes `out_dir/docx/{theme.id}.docx` via `build_report(theme.name, project_theme(...))` and the existing `write_docx`. Returns the paths in theme order.

- [ ] **Step 1: Write failing tests:** two themes and two transcripts (one with a passage for `grace` only, one with a model error). Assert that `write_theme_docxs` returns `[.../docx/grace.docx, .../docx/forgiveness.docx]`, that both files open with `Document(path)`, that `grace.docx`'s level-1 heading is `"Grace"` and it has a `Transcript: <name>` section, that `forgiveness.docx` has no transcript section but its overview table lists both transcripts (the errored one as `FAILED`), and that `project_theme` turns a payload missing the theme into an error entry.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Reuse `build_report` and `write_docx` unchanged.
- [ ] **Step 4:** Full suite passes. Commit: `feat: one Word report per theme`.

---

### Task 5: Run-wide HTML page with cost reporting

**Files:**
- Create: `run_report.py`
- Modify: `report.py` (move the inline CSS from `render_html` into a module constant `BASE_CSS`, unchanged; `render_html` uses it)
- Test: `tests/test_run_report.py`

**Interfaces:**
- Consumes: the multi-theme payload (Task 4) and `report._render_locations_html`, `report._bar_color_var`, `report.BASE_CSS`.
- Produces:
  - `build_run_report(themes: list[Theme], transcripts) -> RunReport`. A dataclass holding `themes`, `rows: list[RunRow]`, `costs: CostSummary` and `generated_at`. For each transcript and theme, the cell is the best successful model's result for that theme (highest score, as `TranscriptResult.best_model` does today). Row and total costs sum `_meta.usage` across all models. Per-theme costs sum `_meta.theme_usage`.
  - `render_run_html(report: RunReport, top_locations: int = 5) -> str`.
  - `write_run_html(themes, transcripts, out_path: str) -> str`.

Page content (top to bottom):
1. A title listing the run's themes and the generation time.
2. **Cost summary:** total tokens (with input, cached and output), total cost, provider discount, and a source line. For `"openrouter"`: "Actual amounts charged by OpenRouter." For `"prices"`: "Estimated from prices in settings (LLM_PRICE_*) or the OpenRouter price list." For `"mixed"`: both. Unknown values show `n/a`. Money is shown as `$0.0000` (4 decimals).
3. **Per-theme table:** theme, transcripts where found (score > 0), est. tokens, est. cost. The column headers say "est.".
4. **Grid:** rows are transcripts, columns are themes, plus a final "Tokens / Cost" column for the exact per-transcript figures. A cell shows the score with `_bar_color_var` background. Tangential-only cells get class `cell-tangential`. Absent cells show `–`. Failed cells show `FAILED`. The grid scrolls horizontally inside its own container; the page body never does.
5. A **theme picker** (`<select>`) with "All themes" plus each theme. Choosing a theme sorts rows by that theme's score and hides the other columns. There's also a transcript-name filter box.
6. **Details:** one `<details id="d-{row}-{theme_id}">` per non-empty cell, holding the reasoning and `_render_locations_html`. Clicking a cell opens and scrolls to it.

All text goes through `html.escape`. Keep light and dark styling from `BASE_CSS`. Inline JS only, with no external scripts.

- [ ] **Step 1: Write failing tests:** a cost summary with a reported cost renders `$0.0123` and "Actual amounts charged by OpenRouter"; a run with no cost info renders `n/a` and not `$0.0000`; per-theme headers contain `est.`; one cell per transcript × theme, with `–` for absent and `FAILED` for an errored model; a theme named `<script>x</script>` appears escaped; `build_run_report` picks the higher-scoring model per theme when two models disagree.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Full suite passes. Open the generated HTML from a fixture in a browser at desktop and at 375px wide to check the grid, picker and details work and the page doesn't scroll sideways. Commit: `feat: run-wide HTML report with token and cost breakdown`.

---

### Task 6: Multi-theme analyzer pipeline

**Files:**
- Modify: `analyzer.py`
- Test: `tests/test_analyzer.py` (append)

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces: `async def analyze(client, model: str, themes: list[Theme], transcript: str, config: Config, progress=None, gate=None) -> tuple[list[AnalysisResult], RunStats]`. One `AnalysisResult` per theme, in theme order. This **replaces** the single-theme `analyze`. Delete `analyze_single_pass`, `analyze_map_reduce`, `_analyze_one_chunk` and `_synthesize_reasoning` in their single-theme forms, and the V1 prompt constants once nothing references them.

Behaviour:
- **Single pass:** one `_call_with_retry` with `MultiThemeChunkAnalysis`, `repair=lambda p: _repair_multi_theme(p, themes)` and `MULTI_SINGLE_PASS_USER_TEMPLATE`. Each theme's markers are extracted with `extract_passage`, as today. `stats.add(..., theme_ids=[t.id ...], theme_output_chars={id: len(entry.model_dump_json())})`.
- **Chunked:** one call per chunk with all themes (`MULTI_CHUNK_USER_TEMPLATE`). Then, **per theme**, `compute_aggregate_score` over that theme's chunk entries, `_extract_and_fill_location` and `merge_and_dedupe_locations`, as today. The theme's `explicitness` is the strongest across chunks (explicit > tangential > absent). Then **one** synthesis call for all themes (`MultiThemeSynthesis`, the same repair function), whose user content lists, per theme, its computed score and the per-segment lines that the current `_synthesize_reasoning` builds.
- `MissingThemesError` is retried by the existing parse/validation branch. After retries are exhausted, it surfaces as the existing `RuntimeError`.

- [ ] **Step 1: Write failing tests** using a fake client that reads `- id: X` lines from the user message and answers with configured per-theme entries:
  - Single pass with 3 themes → `fake.calls == 1`, three results in theme order, and a passage extracted between markers for the theme that has one.
  - Chunked (`single_pass_token_limit` small, so there are 3 chunks) with 2 themes → `fake.calls == 3 + 1` (chunks plus one synthesis), and each theme's score equals `compute_aggregate_score` of its own chunk entries.
  - The fake omits theme `b` on every attempt → `analyze` raises `RuntimeError` mentioning `b`, after `max_retries` calls.
  - The fake omits `b` on the first attempt only → it succeeds on the retry.
  - `stats.theme_usage()` has a key for every theme.
- [ ] **Step 2:** Run `tests/test_analyzer.py`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `tests/test_analyzer.py tests/test_multi_theme_schema.py tests/test_pricing.py`. Expected: pass. **`tests/test_batch.py` is expected to fail until Task 7**, because `cli.py` still calls the old signature. Note this in the commit message.
- [ ] **Step 5:** Commit: `feat: analyze all themes in one call per transcript/chunk (cli wired in next commit)`.

---

### Task 7: CLI, checkpoint, cache and report wiring

**Files:**
- Modify: `cli.py`, `checkpoint.py`, `cache.py`, `report.py` (delete the single-theme page: `render_html`, `write_html*`, `main`, and their tests in `tests/test_report.py`; keep `build_report`, `project_theme`, the helpers and `BASE_CSS`), `word_report.py` (`main` regenerates every theme's docx from the cache), `run_report.py` (add `main`)
- Test: `tests/test_batch.py`, `tests/test_checkpoint.py`, `tests/test_cache.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `checkpoint._signature(themes: list[Theme], models: list[str])` returns JSON of `{"themes": sorted([id, name, description]), "models": sorted(models), "prompt": PROMPT_VERSION}`. `CheckpointWriter`, `read_checkpoint` and `compact` take `themes` where they took `theme`.
  - `cache.write_cache(themes: list[Theme], transcripts, out_dir)` stores `"themes": [asdict(t)]`. `read_cache(path) -> tuple[list[Theme], dict]`. A cache without `"themes"` raises `ValueError("This cache was written by the single-theme version; re-run the analysis.")`.
  - `cli.run_one_model(transcript, themes, model, config, client, label="", gate=None, prices=None) -> dict` returns the multi-theme payload. It calls `stats.apply_prices(prices)` before `summary()` and `theme_usage()`.
  - `run_all(transcript, themes, models, ...)`, `run_batch(transcript_paths, themes, models, out_dir, ...)` and `salvage_report(...)` take `themes`. `run_batch` and the single-file path call `resolve_prices(models, config)` **once** and pass `prices[model]` down.
  - `_write_outputs(themes, raw, out_dir)` writes the cache, `report.html` (`write_run_html`) and `docx/` (`write_theme_docxs`), printing each path.
  - CLI: `--theme` becomes `action="append"` (repeatable); new `--themes-file PATH` and `--max-themes N`. In `main`, right after config: build themes, then `check_theme_limit`. On `ThemeInputError`, print the message to stderr and `sys.exit(2)`, **before** discovery or any API call.
  - Regeneration: `python -m transcript_theme_analyzer.run_report --cache results/.raw_results.json [--out path]` and `python -m transcript_theme_analyzer.word_report --cache results/.raw_results.json [--out-dir dir]`.

- [ ] **Step 1: Update and add tests:**
  - Rework `FakeClient` in `tests/test_batch.py` to answer per `- id:` line (as in Task 6) and include `prompt_tokens_details.cached_tokens` in usage. Existing batch tests keep their assertions, using `[Theme("glory", THEME)]`.
  - New `test_over_limit_rejected_before_any_call`: `main([... 6 × --theme ..., "--max-themes", "5"])` raises `SystemExit(2)`, stderr has the exact copy, and `fake.calls == 0`.
  - New `test_themes_file_and_flags_combine`: CSV plus `--theme` → payload `themes` keys in order.
  - New `test_reordered_themes_reuse_checkpoint` and `test_changed_description_invalidates_checkpoint` (Review Focus 5).
  - New `test_outputs_written`: after a 2-theme batch, `report.html`, `docx/<id>.docx` for each theme and `.raw_results.json` exist, and the HTML contains both theme names.
  - New `test_one_call_per_transcript_for_all_themes`: 3 transcripts × 4 themes, single-pass → `fake.calls == 3`.
  - Cache round-trip with themes, and the old-format cache error, in `tests/test_cache.py`.
- [ ] **Step 2:** Run the full suite. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full suite. Expected: **all pass**.
- [ ] **Step 5:** Commit: `feat: multi-theme CLI, checkpoint and outputs`.

---

### Task 8: Comparison tool, notebook and docs

**Files:**
- Modify: `eval/compare_models.py`, `notebooks/colab_kaggle_runner.ipynb`, `README.md`, `.env.example`

- [ ] **Step 1:** `compare_models.py`: `--theme` is repeatable, plus `--themes-file`. Per model, print one table row per theme (score, locations) and the run's total tokens and cost. Apply `check_theme_limit`. Add a test in `tests/test_cli.py` that `compare_models.main` rejects an over-limit theme list with exit code 2.
- [ ] **Step 2:** Notebook: cell 15 takes `THEMES = [...]` (a list) or `THEMES_FILE`, and `MAX_THEMES`. It builds repeated `--theme` flags with proper quoting (`shlex.quote`). The download cell zips `OUT_DIR/docx/` and downloads the zip. Update the markdown cells that mention a single `report.docx` and "Changing `THEME`".
- [ ] **Step 3:** `README.md`: replace the single-theme run examples and the Outputs section with multi-theme equivalents. Document `MAX_THEMES` and the rejection message, the themes CSV format, the cost section (actual vs estimated, `n/a`, discounts) and the regeneration commands. Fix the stale default-model line. `.env.example`: add `MAX_THEMES=5` and commented `LLM_PRICE_*_PER_M` entries with one-line explanations.
- [ ] **Step 4:** Run the full suite. Then, with the user's OK, do one live smoke run: `--transcript transcript_theme_analyzer/samples/short_transcript.txt --theme "forgiveness" --theme "prayer" --out-dir <scratchpad>/smoke`. Open `report.html`, check the cost summary shows a real `$` figure from OpenRouter, and open both `.docx` files.
- [ ] **Step 5:** Commit: `docs: multi-theme usage, notebook and comparison tool`.
