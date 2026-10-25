# Per-Theme Word Reports in Volumes — Design

Decisions agreed in conversation on 2026-10-25.

## Problem

The per-theme Word document (`docx/<theme_id>.docx`) is a source book for
authors writing on that theme. It must hold every transcript where the theme
is found and every passage in full, word for word. A 20-transcript trial run
produced documents of 33 to 181 pages of excerpt text per theme. The corpus
is over 2,000 transcripts. At the trial's density one theme comes to
roughly 3,300 to 18,000 pages, which no word processor handles well (Word is
sluggish past about 1,000 pages; Google Docs rejects anything over about
1 million characters).

Measurements from the trial (`docx_reports/`, 20 transcripts, 5 themes):

- Passages of 7,180 to 7,975 words occur. Each is one whole 12,000-token
  chunk reported as a single passage. In "The will and plan of God" the 12
  passages over 2,000 words are 60% of the text.
- Broad themes match nearly every transcript (18 of 18 readable transcripts
  scored 64 or higher for "The will and plan of God").
- One passage begins mid-word ("r is in great danger"): the marker matcher
  matches inside words.

## Decisions

1. **Nothing is dropped or shortened.** Every transcript where the theme is
   found (the existing `report.is_found` rule, plus at least one located
   passage) appears, and every passage is printed in full. Whole-section
   passages stay verbatim. No thresholds, no caps, no summaries. The user
   chose this explicitly.
2. **One theme becomes a set of volumes.** When a theme's content exceeds a
   size budget, the Word output is split into numbered files. A transcript is
   never split across volumes. Ranking stays global (strongest transcript
   first), so volume 1 holds the highest-scoring transcripts and the last
   volume the weakest. A theme that fits the budget is written as one file.
3. **Volume size budget: 150,000 words of excerpt text per volume** (about
   300 Word pages). Configurable: env var `DOCX_VOLUME_WORDS` and CLI flag
   `--docx-volume-words`. Clamped to ≥ 1,000. A single transcript larger than
   the budget gets a volume of its own (never split).
4. **An index document per theme** lists every transcript in the run (found
   or not, failed or not) with its score, passage count and volume number,
   ranked as today. Each volume opens with its own overview table of only
   the transcripts it contains, with the clickable internal links that work
   today.
5. **File layout.**
   - Fits in one volume (the common case for narrow themes and small runs):
     `docx/<theme_id>.docx`, unchanged from today, so existing users and the
     notebook see no difference.
   - Needs several volumes: `docx/<theme_id>/<theme_id>-index.docx` and
     `docx/<theme_id>/<theme_id>-vol-01.docx`, `-vol-02.docx`, … (zero-padded
     to the width of the largest volume number). The flat
     `docx/<theme_id>.docx` is not written in that case.
6. **Housekeeping.** Re-running into the same `--out-dir` removes stale
   output of this tool and nothing else. Removed: any top-level `docx/*.docx`
   not written by this run (as today), any `docx/<dir>/` that is
   recognisably a volume directory (it contains `<dir>-index.docx`) for a
   theme not in the run or whose layout changed, and inside a theme's
   directory any `<theme_id>-*.docx` the run did not write (for example
   `-vol-07.docx` after a run that needs only six). Anything else under
   `docx/` is left alone: other subdirectories, non-docx files, Word lock
   files (`~$…`) and LibreOffice lock files (`.~lock.*#`). The existing test
   that `docx/nested/other.docx` and `docx/notes.txt` survive a rerun stays
   true. *Builder-check call:* the reviewer asked whether every unknown
   subdirectory should go; the recognisable-shape rule was chosen because
   deleting a user's own folder is not recoverable, while a stray tool
   folder is a cosmetic problem.
7. **Side fix: whole-word marker matching.** `extraction._find_marker`
   anchors each marker so it cannot match inside a word (a token may be
   edged by punctuation, so the anchors are "not preceded by / not followed
   by a word character", not `\b`). Passages extracted from now on begin and
   end on word boundaries. A marker that only ever matched mid-word is
   dropped by the existing "marker not found" policy, which is the correct
   behaviour. *Reach, builder-check call:* markers are not persisted. The
   cache (`.raw_results.json`) and checkpoint (`.checkpoint.jsonl`) store
   only the extracted excerpt, so the fix applies to transcripts analysed
   after it lands. Already-checkpointed excerpts keep their old boundaries,
   and `word_report --cache` regenerates them as they are. The checkpoint
   signature is not changed and no re-analysis is forced; whoever wants the
   fix in an existing run deletes its checkpoint and re-runs, paying the
   model cost again. The trial corpus is 20 transcripts, so for the current
   state of the project this is cheap; the README notes it.
8. **Regeneration.** `python -m transcript_theme_analyzer.word_report --cache
   …` produces the same volume layout, with the same `--docx-volume-words`
   flag.
9. **Reports and the notebook.** `report.html` is unchanged. The CLI prints
   every written docx path, as today. The notebook's "zip the docx folder"
   step needs no change since it archives the whole `docx/` tree.

## Out of scope

- Any trimming, scoring threshold, passage cap or summary in the Word output.
- Changing how passages are extracted, chunked or scored (other than the
  word-boundary fix).
- Cross-file hyperlinks between the index and the volumes (Word supports
  them but they break when files move; the index gives the volume number
  instead).
- Changes to `report.html`.

## Left to the builder

- How the word budget is counted (whitespace-split words of excerpt text is
  sufficient; headings and metadata lines need not count).
- Exact wording of the index document's title, the volume subtitle
  ("Volume 3 of 12"), and the extra "Volume" and "Passages" columns.
- Whether the volume files repeat the theme title and generation time (they
  should, so each file stands alone).
- Test fixtures: the trial data in `docx_reports/` is useful for a manual
  check but tests must be synthetic and fast.
- Volumes are filled greedily in ranked order: a transcript goes into the
  current volume if the volume's total stays within the budget, otherwise a
  new volume starts. Bookmark names inside a volume reuse the transcript's
  global rank so the index numbering and the volume numbering agree.
- `write_theme_docxs` returns every written path, index first, then the
  volumes in order. `word_report.main` loads the configuration so that
  `DOCX_VOLUME_WORDS` from the environment or `.env` applies to `--cache`
  regeneration too.
