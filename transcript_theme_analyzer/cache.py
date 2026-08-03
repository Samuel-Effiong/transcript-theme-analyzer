"""Internal cache of raw per-run analysis payloads — not a user-facing
output. Lets ``report.py``/``word_report.py`` regenerate the html/docx
reports later without re-running the (paid) LLM analysis. Written as a
dot-prefixed file so it reads as internal, distinct from the two real
outputs (``report.html``, ``report.docx``).
"""
from __future__ import annotations

import json
import os

CACHE_FILENAME = ".raw_results.json"


def write_cache(theme: str, transcripts: dict[str, list[tuple[str, dict]]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CACHE_FILENAME)
    serializable = {
        "theme": theme,
        "results": {
            name: [{"model": model, "payload": payload} for model, payload in payloads]
            for name, payloads in transcripts.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return path


def read_cache(path: str) -> tuple[str, dict[str, list[tuple[str, dict]]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    transcripts = {
        name: [(entry["model"], entry["payload"]) for entry in entries]
        for name, entries in data.get("results", {}).items()
    }
    return data.get("theme", "(unknown theme)"), transcripts
