"""Append-only checkpoint of completed transcripts, so an interrupted batch
can resume instead of re-paying for work already done.

Format is JSON Lines -- one self-contained JSON object per completed
transcript -- chosen deliberately over rewriting a single JSON document:

* **Crash safety.** Rewriting one growing document means every save has a
  window where the file on disk is half-written. A crash there destroys the
  whole run's results, which is the exact failure checkpointing exists to
  prevent. Appending only ever risks the final line, and a torn final line is
  simply dropped on read.
* **Cost.** Rewriting an N-entry document after each of N transcripts is
  O(N^2) work. At 1800+ transcripts that is a meaningful share of runtime,
  spent re-serializing results that have not changed.

Each record is keyed by the transcript's path relative to the input
directory. Basenames are *not* unique in real corpora (nested folders,
"foo (1).docx" style duplicates), and a key collision on resume would skip a
file that was never actually analyzed.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile

CHECKPOINT_FILENAME = ".checkpoint.jsonl"

logger = logging.getLogger("transcript_theme_analyzer")


def checkpoint_path(out_dir: str) -> str:
    return os.path.join(out_dir, CHECKPOINT_FILENAME)


def _signature(theme: str, models: list[str]) -> str:
    """Results are only reusable for the same theme and model set; anything
    else is a different question and must be re-analyzed."""
    return json.dumps({"theme": theme, "models": sorted(models)}, sort_keys=True)


class CheckpointWriter:
    """Appends one record per completed transcript, flushed to disk
    immediately so an abrupt kill (Colab timeout, closed tab) loses at most
    the transcript currently in flight."""

    def __init__(self, path: str, theme: str, models: list[str]) -> None:
        self.path = path
        self.signature = _signature(theme, models)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def record(self, key: str, display_name: str, payloads: list[tuple[str, dict]]) -> None:
        line = json.dumps(
            {
                "signature": self.signature,
                "key": key,
                "display_name": display_name,
                "results": [{"model": model, "payload": payload} for model, payload in payloads],
            },
            ensure_ascii=False,
        )
        self._fh.write(line + "\n")
        # flush() alone only reaches the OS buffer -- fsync is what survives a
        # hard kill of the whole machine/VM, which is the realistic failure
        # mode on a hosted notebook.
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass

    def __enter__(self) -> "CheckpointWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_checkpoint(path: str, theme: str, models: list[str]) -> dict[str, dict]:
    """Load previously completed transcripts for this exact theme/model set.

    Returns ``{key: {"display_name": str, "results": [(model, payload), ...]}}``.
    Records from a different theme/model set are ignored, as are malformed
    lines -- the last line of a killed run is routinely half-written, and one
    torn line must not invalidate thousands of good records. A later record
    for the same key supersedes an earlier one, which is what lets a retried
    failure overwrite its own error entry.
    """
    if not os.path.isfile(path):
        return {}

    wanted = _signature(theme, models)
    done: dict[str, dict] = {}
    skipped_other = 0
    skipped_bad = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = record["key"]
                results = [(entry["model"], entry["payload"]) for entry in record["results"]]
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped_bad += 1
                continue
            if record.get("signature") != wanted:
                skipped_other += 1
                continue
            done[key] = {
                "display_name": record.get("display_name", key),
                "results": results,
            }

    if skipped_other:
        logger.info(
            "Checkpoint: ignored %d record(s) from a different theme/model set", skipped_other
        )
    if skipped_bad:
        logger.warning(
            "Checkpoint: skipped %d unreadable record(s) (expected if a previous run was killed "
            "mid-write); those transcripts will simply be re-analyzed",
            skipped_bad,
        )
    return done


def has_error(results: list[tuple[str, dict]]) -> bool:
    """True if any model's payload for this transcript is an error entry."""
    return any("error" in payload for _, payload in results)


def compact(path: str, theme: str, models: list[str]) -> int:
    """Rewrite the checkpoint with one line per key, dropping superseded and
    unusable records. Purely housekeeping -- returns the surviving count."""
    done = read_checkpoint(path, theme, models)
    if not done:
        return 0
    signature = _signature(theme, models)
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for key, entry in done.items():
                fh.write(
                    json.dumps(
                        {
                            "signature": signature,
                            "key": key,
                            "display_name": entry["display_name"],
                            "results": [
                                {"model": model, "payload": payload}
                                for model, payload in entry["results"]
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)  # atomic; readers never see a partial file
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return len(done)
