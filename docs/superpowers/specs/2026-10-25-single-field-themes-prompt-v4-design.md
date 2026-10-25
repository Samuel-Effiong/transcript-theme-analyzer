# Single-Field Themes and Analysis Prompt v4 — Design

Decisions agreed in conversation on 2026-10-25.

## Problem

The 20-transcript trial exposed two weaknesses in how the analyzer is asked
its question:

- **Scores do not discriminate.** 18 of 18 readable transcripts scored 64 or
  higher for "The will and plan of God". The prompt gives a 0–100 scale with
  no meaning attached to any band, so a broad theme ranks everything high and
  the Word document's ranking is no help to an author.
- **Passages are too coarse.** The prompt says one continuous discussion is
  one passage and never says where a passage ends, so the model reported
  whole 8,000-word chunks as single passages. In the worst theme such
  passages were 60% of the text. Authors looking for one teaching point must
  read ten pages to find it.

A third problem is input shape. Themes are a name plus an optional
description in a second CSV column. The user does not want two values. A
theme should be one field that may be a short label ("will of God") or a
full description with inclusions and exclusions, and the model should use
whatever is given.

## Decisions

### 1. The theme is one field

- A theme is a single text. It may be a phrase or several sentences. Users
  enter it once: `--theme "..."`, one `theme` column in the CSV, one string
  in the notebook's `THEMES` list.
- The `description` concept is removed from `Theme`, the CSV reader, the
  prompt's theme block, the checkpoint signature, the notebook text and the
  README. A CSV that still has a `description` column is rejected before any
  API call with a message telling the user to put the full text in the
  `theme` column, since a theme may now be a whole paragraph. *Builder-check
  call:* rejection was chosen over silently dropping the column (which would
  quietly change the question asked) and over merging name and description
  into one field (which would keep the two-column format alive in practice).
  No two-column CSV is known to exist; the feature is weeks old.
- Whitespace in the theme text is still collapsed to one line (existing
  behaviour), because the prompt lists one theme per line.

### 2. Every theme gets a short label, derived automatically

- The label is the first six words of the theme text, cut at a word
  boundary, with trailing punctuation removed. A theme of six words or fewer
  is its own label. Labels are made unique within a run by a numeric suffix,
  as ids are today.
- The id (file-system slug) is derived from the label, not from the full
  text, so a paragraph-length theme gives a short file name such as
  `teachings-about-gods-will-for-an.docx`.
- Where label and full text are used:
  - **Prompt:** the full text. Theme block line: `- id: <id> | <full text>`.
  - **Word report:** heading = label; when the text is longer than the label,
    the full text is a paragraph directly under the heading, in every file of
    a multi-volume theme.
  - **HTML report:** the theme column, the page title and the subtitle show
    the label; the full text is available on hover (a `title` attribute) and
    is also printed once in full in the Per theme cost table or a themes
    list, so the page records exactly what was asked. *Builder decides where.*
  - **CLI output and logs:** the label.
  - **Checkpoint signature:** id and full text. A changed theme text is a
    different question and invalidates the checkpoint, as a changed
    description does today.

### 3. Analysis prompt v4

`PROMPT_VERSION` becomes `multi-v4`, so no existing checkpoint is reused and
the trial is re-run. The system prompt changes in these ways; exact wording
is the builder's, within the content below.

- **The theme text is the definition.** When a theme is a phrase, interpret
  it in its ordinary sense within this kind of material. When it is a fuller
  description, every inclusion and exclusion in it is binding. The name is
  not searched for as a string (existing text, kept).
- **Score bands with meanings.**
  - 85–100: the theme is the main subject of the segment.
  - 60–84: a substantial discussion, but not the centre of the segment.
  - 30–59: a real but brief treatment.
  - 1–29: a passing mention.
  - 0: absent.
  Tell the model that across a large corpus most segments fall in the lower
  bands, and that a high score must be earned by what the segment itself
  contains.
- **Passages are teaching points.** The prompt must express these rules;
  exact wording is the builder's.
  - A passage is one distinct teaching point: a single point the speaker
    makes about the theme, together with the explanation, Scripture and
    illustration that belong to it. It must preserve a complete thought.
  - A passage ends where the speaker moves to a substantially different
    point, even while staying on the theme. Cues: a new claim or heading, a
    new text being opened, an illustration that serves a new point, a
    transition such as "secondly" or "now let's turn to".
  - One passage per teaching point, never one passage per chunk or per
    sermon section. A passage spanning an entire segment is almost always
    several teaching points reported as one. Typically a teaching point runs
    one to several paragraphs; a connected argument that genuinely develops
    one point across pages stays together.
  - Never split a passage merely to make it shorter, and never trim a
    passage to a highlight sentence when the surrounding paragraphs carry
    the same point.
  - Adjacent teaching points may touch or lightly overlap: the end marker of
    one may sit at the start of the next. Together the passages of a long
    discussion should cover it, so nothing the speaker said on the theme is
    lost between them.
  - Each passage carries a title derived from what the speaker actually
    teaches in it, never from predetermined categories. Illustrative
    example to give the model: a ten-page discussion of the will of God
    might become "Understanding the Will of God", "How to Discover God's
    Will", "Aligning Your Decisions with God's Will", "The Role of Prayer in
    Knowing God's Will", "Obedience to God's Will", "Walking in God's
    Purpose".
  - Titles are labels; the passage text is the speaker's words exactly as
    spoken, located by the markers.
- **Markers on whole words.** `start_marker` and `end_marker` begin and end
  on complete words, with punctuation copied exactly as it appears.
- **Two worked examples instead of one**, at clearly different scores (one
  explicit and high, one tangential and low) plus the absent case, so the
  model does not gravitate to the single illustrated value.
- **Shorter format section.** The repeated "never omit" warnings are cut by
  about a third: every rule is stated once, with its example, and not
  restated. The mandatory keys, the one-entry-per-theme rule and the
  JSON-only rule stay. *Builder-check call:* the reviewer noted that when a
  provider rejects `response_format=json_schema` the analyzer falls back to
  plain calls where the prompt and the parse-and-retry loop are the only
  enforcement. The repository already asserts the DeepSeek models support
  structured output, and the retry loop covers the fallback, so the cut
  stands. The trial re-run is the check: the log line "rejected structured
  output" or a rise in validation retries means the format section needs
  more weight, and the prompt is versioned so that is a cheap follow-up.
- The synthesis prompt is unchanged apart from receiving the theme text
  rather than name plus description.

### 4. Old caches and checkpoints

- A `.raw_results.json` written before this change stores themes with a
  `description` field. The regenerators (`run_report --cache`,
  `word_report --cache`) reject it with a one-line message in the style of
  the existing single-theme-cache message, telling the user to re-run the
  analysis. *Builder-check call:* a migration shim was considered and
  dropped; the design already requires a v4 re-run, and the v3 trial's Word
  output is committed in `docx_reports/` for side-by-side comparison.
- The `multi-v4` prompt version means every existing checkpoint is stale and
  ignored, as the checkpoint rules already provide.

### 5. What is not changing

- Extraction, chunking, aggregation, the docx layout and volumes, and the
  "found" rule are untouched.
- The analyzer's fallback that matches a theme by name when the model
  mislabels an id now matches on the full text (same behaviour, new field).

## Out of scope (future projects, agreed)

- Carrying the tail of the previous chunk into the next call so a
  discussion that straddles a chunk boundary is scored once, properly.
- A hand-rated golden set from the trial transcripts to measure prompt
  changes rather than eyeball them.
- Any threshold, cap or trimming in the outputs.

## Left to the builder

- Field names on `Theme` (the design needs an id, a short label and the full
  text; `description` goes away).
- Exact prompt wording within the content above; keep the theme block first
  so the cacheable prefix is unchanged across a run.
- Where the full theme text is printed once in the HTML report.
- Whether the label's word count (six) is a constant or configurable
  (constant is enough).
- Tests must be synthetic and fast. Re-running the paid trial is the user's
  manual check after the change lands, not part of the test suite.
