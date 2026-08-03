# Transcript Theme-Relevance Analyzer

Scores how thoroughly a transcript discusses a given theme, with a per-run
`relevance_score` (0-100), `reasoning`, and a list of `locations` in the
source transcript — model-agnostic (OpenAI direct or any OpenAI-compatible
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
├── cli.py            # run one or many (transcript, theme, model) combos, save results
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
LLM_DEFAULT_MODEL=anthropic/claude-sonnet-5

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
```

Switching providers is a config-only change (`client.py` never hardcodes a
provider) — same pipeline code runs against OpenAI or any OpenAI-compatible
endpoint by changing `base_url` / `api_key` / the model string.

> **Model ID note:** `anthropic/claude-sonnet-5` is this repo's default per
> Anthropic's current model catalog (`claude-sonnet-5`, not the older
> `claude-sonnet-4.5`). OpenRouter mirrors upstream provider model IDs as
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

Single model:

```bash
python -m transcript_theme_analyzer.cli --transcript path.docx --theme "the glory of God"
```

Multiple models per transcript (the best-scoring model's excerpts are what
show up in the reports; the others are listed for comparison):

```bash
python -m transcript_theme_analyzer.cli \
  --transcript path.docx --theme "the glory of God" \
  --models gpt-4.1 anthropic/claude-sonnet-5 google/gemini-2.5-pro
```

A whole folder of transcripts against the same theme (mixing `.txt` and
`.docx` files freely):

```bash
python -m transcript_theme_analyzer.cli \
  --transcript-dir path/to/transcripts/ --theme "forgiveness" \
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
both reports, same as a model-call failure.

## Outputs

Every CLI run produces exactly two files in `--out-dir` (default `results/`),
both ranked by relevance score, highest first:

- **`report.docx`** — a single consolidated Word document, not one file per
  transcript:
  - **Title** — the search theme.
  - **Overview table** right under the title, listing every transcript and
    its relevance score; transcript names are clickable links that jump
    straight to that transcript's section further down (transcripts with no
    located excerpts, or where every model run failed, still show up in the
    table with their score/`FAILED`, just without a link — there's nothing
    to jump to).
  - **One section per transcript** that has at least one located excerpt,
    headed `Transcript: <name>`, so an excerpt can always be traced back to
    its source.
  - **One sub-heading per excerpt**, generated from that excerpt's own
    context (not the search keyword), with the quoted passage underneath and
    its timestamp/speaker if known.
- **`report.html`** — a self-contained, no-dependency interactive page:
  - A sortable ranked table (click any column header) with a relevance bar
    per transcript, and a live filter box to narrow by transcript name.
  - Clicking a transcript name jumps to and expands its detail panel
    (reasoning + key excerpts) further down the page.
  - Mobile-friendly: the table collapses into stacked cards below ~640px
    wide instead of squeezing into unreadable columns.
  - Light/dark mode both styled.
  - Unlike the docx, every analyzed transcript gets a detail panel here
    (including zero-score or failed ones) — the html is the full audit view,
    the docx is the curated excerpt deliverable.

Failed model runs (e.g. a rate limit or timeout) are captured per-model and
shown as `FAILED`/error rather than crashing either report.

Alongside those two, each run also writes a dot-prefixed internal cache file,
`.raw_results.json` — **not** one of the two outputs above, just enough raw
data to regenerate `report.html`/`report.docx` later without re-running the
(paid) analysis:

```bash
python -m transcript_theme_analyzer.report --cache results/.raw_results.json
python -m transcript_theme_analyzer.word_report --cache results/.raw_results.json
# optionally: --out custom_name.{html,docx} --min-score 50 (docx only)
```

`--min-score` (docx regeneration only) restricts sections to transcripts
whose best score meets a threshold (by default, any transcript with a
located excerpt gets a section).

Comparison table (score / token cost proxy / latency / reasoning) across models:

```bash
python -m transcript_theme_analyzer.eval.compare_models \
  --transcript path.txt --theme "forgiveness" \
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
  (bounded by `MAX_CONCURRENT_CHUNKS`) against the same theme, producing a
  partial score, reasoning, and locations per chunk.
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
- **Location merging:** chunk-relative offsets are translated to
  full-transcript offsets, then locations whose ranges overlap by more than
  50% of the smaller range's length (the overlap-region double-count case)
  are merged, keeping the longer/more complete excerpt.
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
both transient API errors and JSON parse/validation failures).

## Cost / token logging

Every call's token usage is recorded (`RunStats` in `analyzer.py`) and
included under `_meta.usage` in each result JSON and in the comparison
table, so multi-model runs can be judged on cost as well as quality.

## Sample outputs

`samples/` contains:
- `short_transcript.txt` / `sample_output_short.json` — demonstrates the
  **single-pass** path.
- `long_transcript.txt` / `sample_output_long_chunked.json` — demonstrates
  the **chunked map-reduce** path (`"chunked": true`).

These sample outputs were generated with a content-aware stub client (this
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
