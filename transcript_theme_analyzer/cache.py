"""Internal cache of raw per-run analysis payloads — not a user-facing
output. Lets ``run_report.py``/``word_report.py`` regenerate ``report.html``
and the per-theme ``docx/`` reports later without re-running the (paid) LLM
analysis. Written as a dot-prefixed file so it reads as internal, distinct
from the real outputs.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from .themes import Theme

CACHE_FILENAME = ".raw_results.json"
OLD_FORMAT_MESSAGE = "This cache was written by the single-theme version; re-run the analysis."


def write_cache(
    themes: list[Theme], transcripts: dict[str, list[tuple[str, dict]]], out_dir: str
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CACHE_FILENAME)
    serializable = {
        "themes": [asdict(t) for t in themes],
        "results": {
            name: [{"model": model, "payload": payload} for model, payload in payloads]
            for name, payloads in transcripts.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return path


def read_cache(path: str) -> tuple[list[Theme], dict[str, list[tuple[str, dict]]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "themes" not in data:
        # Single-theme payloads have no per-theme entries, so there is nothing
        # the multi-theme reports could honestly render from them.
        raise ValueError(OLD_FORMAT_MESSAGE)
    themes = [Theme(**t) for t in data["themes"]]
    transcripts = {
        name: [(entry["model"], entry["payload"]) for entry in entries]
        for name, entries in data.get("results", {}).items()
    }
    return themes, transcripts
