# Multi-Theme Analysis — Design

Decisions agreed in conversation on 2026-10-20.

## Problem

The pipeline analyses one theme per run. 30 themes means 30 full passes over
a corpus of hundreds to thousands of transcripts, re-sending every transcript
30 times.

## Decisions

1. **Several themes per run, all evaluated together.** Every theme in a run
   goes to the model in a **single call per transcript** (or per chunk, for
   transcripts long enough to be chunked). There is no internal splitting
   into smaller groups.
2. **`MAX_THEMES` setting caps themes per run.** Adjustable (env var, plus a
   CLI override). A run with more themes than the cap is **rejected before
   any API call**, with a message naming the count, the cap, and how many to
   remove. Purpose: test quality at different theme counts (5, 10, 15 …), and
   let users see a clear limit when adding themes.
3. **Every mention counts.** Explicit and tangential passages are both
   extracted. There is no screening stage and no score threshold for
   extraction.
4. **Extraction is unchanged in kind.** For each theme the model returns a
   score, explicitness, reasoning and start/end markers per passage; code
   extracts the exact text between the markers. Per theme, the result is the
   same as today's single-theme result.
5. **Word output: one `.docx` per theme**, each in today's report format
   (theme title, ranked overview table, sections per transcript, titled
   passages).
6. **HTML output: one page for the current run only** — every transcript ×
   every theme in this run (a score grid; click a cell for that transcript's
   reasoning and passages for that theme; pick one theme for a ranked view).
7. **Cost reporting in the HTML:**
   - Tokens and cost **in total** (exact) and **per transcript** (exact).
   - Tokens and cost **per theme** — an **estimate**, labelled as such, since
     themes share each call.
   - Money: **OpenRouter's actual charged amount** when available; otherwise
     tokens × prices from settings.
   - **Provider discounts** (cached-input savings) shown.
8. **No output token cap** by default (already implemented).

## Out of scope

- Screening / two-stage analysis.
- Accumulating results across runs into one page.
- Embedding pre-filters.
