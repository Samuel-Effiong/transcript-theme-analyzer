# Transcript Theme-Relevance Analyzer

Scores how thoroughly a transcript discusses one or more themes (up to
`MAX_THEMES` per run, default 5), with a per-theme `relevance_score` (0-100),
`reasoning`, and a list of `locations` in the source transcript — model-agnostic (OpenAI direct or any OpenAI-compatible
endpoint, e.g. OpenRouter), and safe for transcripts from 30 minutes to
10+ hours of audio.

## Structure

```
transcript_theme_analyzer/
├── config.py       # env-driven config: provider, base_url, api_key, model, chunk size/overlap
├── client.py       # OpenAI SDK client factory (OpenAI vs OpenRouter, model-agnostic)
├── chunker.py       # splits transcript into overlapping chunks w/ offset metadata
├── analyzer.py       # core pipeline: single-pass path + map-reduce path
├── schema.py         # pydantic models for the output schema
├── aggregate.py       # merges chunk-level results into final result
├── prompts.py         # the analysis/synthesis system prompts (versioned constants)
├── loader.py          # detects .txt/.docx input format, extracts plain text
├── themes.py          # theme list: --theme / CSV parsing, unique ids, the MAX_THEMES check
├── pricing.py         # per-call cost (reported or tokens x prices), provider discount, per-theme split
├── report.py          # shared results data model + per-theme projection of a run
├── run_report.py      # report.html: every transcript x every theme, plus the cost section
├── word_report.py     # docx/<theme_id>.docx: one Word report per theme
├── cli.py            # run one or many (transcript, themes, model) combos, save results
├── eval/
│   └── compare_models.py   # run same input across multiple models, side-by-side comparison
├── samples/           # sample transcripts + sample output JSON (see below)
└── tests/
```

This matches the requested layout with one addition: `samples/` for the
deliverable example transcripts/outputs.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r transcript_theme_analyzer/requirements.txt
```

Create a `.env` in your working directory (or export the equivalent env vars):

```
# Pick ONE provider setup:

# --- OpenRouter (default) ---
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-...
LLM_DEFAULT_MODEL=deepseek/deepseek-v4-flash

# --- OpenAI direct (swap by changing only these three lines) ---
# LLM_PROVIDER=openai
# LLM_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...
# LLM_DEFAULT_MODEL=gpt-4.1

CHUNK_SIZE_TOKENS=12000
CHUNK_OVERLAP_TOKENS=800
SINGLE_PASS_TOKEN_LIMIT=20000
MAX_CONCURRENT_CHUNKS=8
LLM_MAX_RETRIES=5

# Most themes allowed in one run (default 5). Over the limit = rejected up front.
MAX_THEMES=5

# Excerpt words per Word volume (default 150000, about 300 pages).
DOCX_VOLUME_WORDS=150000
```

The code's default model is `deepseek/deepseek-v4-flash`; set
`LLM_DEFAULT_MODEL` (or pass `--models`) to use another. No output token cap
is applied unless you set `LLM_MAX_OUTPUT_TOKENS`. If a model still runs out
of output space mid-answer (more likely with many themes), that run fails at
once rather than retrying, with a hint to lower `MAX_THEMES` or pick a model
with a larger output limit.

Switching providers is a config-only change (`client.py` never hardcodes a
provider) — same pipeline code runs against OpenAI or any OpenAI-compatible
endpoint by changing `base_url` / `api_key` / the model string.

> **Model ID note:** `anthropic/claude-sonnet-5` is Anthropic's current
> model (`claude-sonnet-5`, not the older `claude-sonnet-4.5`). OpenRouter mirrors upstream provider model IDs as
> `anthropic/<model-id>`, but OpenRouter's catalog is a third-party listing —
> confirm the exact slug at https://openrouter.ai/models before relying on it
> in production, since it can lag a provider's own naming.

## Running

Transcripts can be `.txt` or `.docx` — format is auto-detected per file
(`loader.py`). For `.docx`, only the paragraph text is extracted; run-level
formatting (bold/italic/fonts/colors) is intentionally dropped since it
carries no meaning for a theme-relevance judgment, but paragraph breaks are
preserved (one paragraph per line), since the chunker's speaker-label
detection depends on line boundaries.

Single model, one theme:

```bash
python -m transcript_theme_analyzer.cli --transcript path.docx --theme "the glory of God"
```

Several themes at once. Repeat `--theme`; every theme goes in the **same**
LLM call per transcript (or per chunk), so each extra theme costs far less than
a separate run:

```bash
python -m transcript_theme_analyzer.cli \
  --transcript path.docx --theme "the glory of God" --theme "forgiveness" --theme "prayer"
```

Or read the themes from a CSV with `--themes-file`:

```bash
python -m transcript_theme_analyzer.cli --transcript path.docx --themes-file themes.csv
```

```csv
theme,description
forgiveness,"Extending or receiving pardon, including reconciliation"
prayer,
the glory of God,
```

The header needs a `theme` column; `description` is optional and, when given,
is shown to the model to disambiguate the theme. Header names are
case-insensitive and blank rows are skipped. `--theme` and `--themes-file`
combine (the `--theme` entries come first). Duplicate theme names are rejected.

**Theme limit.** A run takes at most `MAX_THEMES` themes (env var, default 5;
`--max-themes N` overrides it for one run). Over the limit, the run stops
before any API call, exit code 2:

```
You entered 7 themes. The limit is 5 (MAX_THEMES). Remove 2 theme(s) or raise MAX_THEMES.
```

Multiple models per transcript (for each transcript and theme, the
best-scoring model's excerpts are what show up in the reports; `report.html`
lists the other models' scores for that theme in its detail panel):

```bash
python -m transcript_theme_analyzer.cli \
  --transcript path.docx --theme "the glory of God" --theme "forgiveness" \
  --models gpt-4.1 anthropic/claude-sonnet-5 google/gemini-2.5-pro
```

A whole folder of transcripts against the same themes (mixing `.txt` and
`.docx` files freely):

```bash
python -m transcript_theme_analyzer.cli \
  --transcript-dir path/to/transcripts/ --theme "forgiveness" --theme "prayer" \
  --models anthropic/claude-sonnet-5
```

`--glob` controls which files are picked up inside `--transcript-dir`
(default: both `*.txt` and `*.docx`; pass `--glob` to restrict to one
pattern). Every discovered file's format is validated upfront — before any
model is called — so an unsupported or corrupt file is reported immediately
instead of after other transcripts in the same batch have already been
(expensively) analyzed. A file that fails to read *during* the batch (e.g. a
docx that's corrupt in a way the upfront check didn't catch) doesn't abort
the run — that transcript is recorded as failed and shows up as `FAILED` in
the reports, same as a model-call failure.

## Outputs

Every CLI run writes these into `--out-dir` (default `results/`):

- **`report.html`** — one self-contained, no-dependency interactive page for
  the current run (all transcripts x all themes):
  - A grid with one row per transcript and one colour-shaded score per theme
    (`–` where the theme wasn't found, `FAILED` where every model failed),
    plus each transcript's tokens / cost. Pick a theme in the theme picker to
    show only that column and rank the transcripts by it; the filter box
    narrows by transcript name.
  - Click a score to open its details below the grid: the model's reasoning
    and the key passages (with timestamp/speaker when known). On a
    multi-model run it also lists the other models' scores for that theme.
  - A **Cost** section with the run's tokens and cost — OpenRouter's actual
    charge, else an estimate from prices, else `n/a` — and the estimated
    provider discount, plus a **Per theme** table (est.; see below).
  - Mobile-friendly and light/dark styled.
- **`docx/<theme_id>.docx`** — one Word document per theme (the id is a slug of
  the theme name, e.g. `docx/forgiveness.docx`), the curated excerpt deliverable:
  - **Title** — the theme.
  - **Overview table** listing every transcript and its relevance score, ranked
    highest first; transcript names are clickable links to that transcript's
    section (transcripts with no located excerpts, or where every model run
    failed, still show up with their score/`FAILED`, just without a link).
  - **One section per transcript** where the theme was found (score above 0
    and not marked absent — the same rule the HTML grid uses) and that has at
    least one located excerpt, headed `Transcript: <name>`.
  - **One sub-heading per passage**, a short section-title (2-6 words)
    generated from that passage's own content, with the full continuous
    passage underneath and its timestamp/speaker if known.

Failed model runs (e.g. a rate limit or timeout) are captured per-model and
shown as `FAILED`/error rather than crashing either report.

Reusing an `--out-dir` is safe: `docx/` only ever holds the current run's
themes. A Word file for a theme that is no longer in the run is removed.

### Cost reporting

Tokens and cost appear in the `report.html` Cost section:

- **Actual vs estimated.** When OpenRouter reports the amount it charged
  (`usage.cost`), that figure is used. Otherwise the cost is tokens x prices,
  from `LLM_PRICE_INPUT_PER_M` / `LLM_PRICE_CACHED_INPUT_PER_M` /
  `LLM_PRICE_OUTPUT_PER_M` (USD per 1M tokens; env values win) or OpenRouter's
  public price list. If neither is known the figure is shown as `n/a`, never
  `$0.00`.
- **Every paid call counts.** Retries (a bad or incomplete answer) and the
  calls of a run that later failed are included, so the totals match the bill.
- **Totals vs per-theme.** Run totals and per-transcript figures are exact.
  Themes share one call, so the per-theme tokens and cost are marked **est.**
- **Provider discount.** Cached input tokens x (input price - cached price) —
  an estimate from list prices. OpenRouter's list price can differ from the
  rate actually charged, so the reported cost is the authoritative number.

### Regenerating the reports

Each run also writes dot-prefixed internal files — **not** final outputs:
`.raw_results.json` (enough raw data to regenerate the reports without
re-running the paid analysis) and `.checkpoint.jsonl` (resume state). To rebuild:

```bash
python -m transcript_theme_analyzer.run_report --cache results/.raw_results.json [--out path.html]
python -m transcript_theme_analyzer.word_report --cache results/.raw_results.json [--out-dir dir]
```

`run_report` rewrites `report.html` (next to the cache by default); `word_report`
rewrites `docx/<theme_id>.docx` under `--out-dir` (default: the cache's folder).

Passage boundaries are fixed at analysis time and only the extracted text is
cached, so a change to passage extraction (such as the whole-word marker
matching, which stops a passage starting or ending inside a word) shows up in
transcripts analysed after it. To apply it to a finished run, delete that run's
`.checkpoint.jsonl` and re-run, which pays the model cost again.

### Resuming

Re-running the same command after an interruption picks up where it left off.
A checkpoint is reused only when the theme set (order does not matter; names
and descriptions do), the models and the prompt version are all unchanged;
otherwise the run starts fresh. `--no-resume` ignores the checkpoint.

### Comparing models

Score and locations per theme, plus total tokens and cost, for each model on
one transcript (themes via repeated `--theme` or `--themes-file`; the same
`MAX_THEMES` limit applies):

```bash
python -m transcript_theme_analyzer.eval.compare_models \
  --transcript path.txt --theme "forgiveness" --theme "prayer" \
  --models gpt-4.1 anthropic/claude-sonnet-5
```

## How chunking and aggregation work

- **Fast path (single-pass):** if the transcript is under
  `SINGLE_PASS_TOKEN_LIMIT` (token count approximated with a
  chars-per-token heuristic — this pipeline is model-agnostic, so a real
  provider-specific tokenizer isn't assumed), it's sent to the model in one
  call. This is the common case for the ~30-minute-to-a-few-hours range.
- **Chunked path (map-reduce):** longer transcripts are split into
  overlapping chunks (`chunker.py`), each carrying its character-offset
  range, chunk index, and (if detected) the nearest embedded timestamp /
  speaker label. Each chunk is analyzed independently and concurrently
  (bounded by `MAX_CONCURRENT_CHUNKS`) against the same themes (all in one call per chunk), producing a
  partial score, reasoning, and locations per chunk and theme.
- **Aggregation (`aggregate.py`):** the final `relevance_score` is an
  explicit, tunable heuristic — not a naive max or average of chunk scores:

  ```
  final = 0.30 * peak_intensity + 0.40 * length_weighted_average + 0.30 * coverage_breadth
  ```

  - `peak_intensity` — the single highest chunk score, so one strong,
    concentrated discussion isn't diluted away in a long transcript.
  - `length_weighted_average` — chunk scores weighted by chunk length,
    rewarding sustained discussion over a single spike.
  - `coverage_breadth` — the fraction of the transcript (by length) whose
    chunks were rated "explicit" (full weight) or "tangential" (half
    weight) vs "absent" (zero), so score reflects how much of the
    transcript actually engages the theme.

  The weights are keyword arguments to `compute_aggregate_score(...)` — swap
  them, or replace the whole function with an LLM aggregation pass over the
  chunk-level outputs, without touching the map step.
- **Passage extraction:** the model never reproduces a passage's text
  itself — it reports a short `start_marker`/`end_marker` boundary phrase
  pair, and `extraction.py` deterministically locates and extracts the
  exact original text between them (whitespace/case-tolerant matching).
  This is far cheaper in completion tokens than verbatim reproduction and
  avoids the model truncating or paraphrasing a long passage. If a marker
  can't be located, that location is dropped rather than fabricated.
- **Location merging:** locations from the overlap region of two adjacent
  chunks (the same passage flagged by both) are merged by comparing
  normalized excerpt text — an exact match after collapsing
  whitespace/case — keeping the longer/more complete capture.
- **Reasoning synthesis:** the final `reasoning` is itself produced by an
  LLM call (cheap relative to the chunk-analysis pass) that synthesizes the
  chunk-level reasonings into one coherent paragraph — not a concatenation.

## Structured output enforcement

Every model call first tries the OpenAI SDK's native structured-output mode
(`response_format={"type": "json_schema", ...}`, schema generated from the
pydantic models in `schema.py`). If the provider/model rejects that
parameter (many OpenRouter-routed non-OpenAI models don't support it),
`analyzer._call_with_retry` falls back to plain JSON-mode prompting with a
strict parse-and-retry loop (bounded by `LLM_MAX_RETRIES`, with backoff on
both transient API errors and JSON parse/validation failures). An answer cut
off at the output limit is not retried: the same request would be cut off
again.

## Cost / token logging

Every call's token usage and cost are recorded (`RunStats` in `analyzer.py`,
prices from `pricing.py`) — retries and the calls of a failed run included —
and stored under `_meta.usage` in each result (a failed run's error entry
too), in
the `report.html` Cost section (see *Cost reporting* above) and in the
comparison table, so multi-model runs can be judged on cost as well as quality.

## Sample outputs

`samples/` contains:
- `short_transcript.txt` / `sample_output_short.json` — demonstrates the
  **single-pass** path.
- `long_transcript.txt` / `sample_output_long_chunked.json` — demonstrates
  the **chunked map-reduce** path (`"chunked": true`).

The sample output JSONs predate multi-theme runs (one theme, old shape).
They were generated with a content-aware stub client (this
sandbox has no live API key) that mimics the shape of a real model response —
they demonstrate the pipeline's wiring and output schema, not real model
judgment. Run the CLI yourself with a real key to get real analysis.

## Tests

```bash
pytest transcript_theme_analyzer/tests
```

Covers the chunker (overlap, boundary detection, structure detection) and
the aggregation heuristic (score behavior under absent/sparse/sustained
discussion, location dedup).

## Deviations from the suggested structure

- Added `samples/` (not in the original file tree) to hold the deliverable
  example transcripts/outputs.
- `.env` loading is a ~15-line built-in loader in `config.py` rather than a
  `python-dotenv` dependency, to keep the dependency list minimal
  (`openai`, `pydantic`, `pytest`, `pytest-asyncio`).
