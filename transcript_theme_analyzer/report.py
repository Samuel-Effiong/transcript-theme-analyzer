"""Shared results data model + the HTML building blocks.

``project_theme`` narrows the multi-theme run payloads (as produced by
``cli.py``) to one theme, and ``build_report`` turns that into a
``BatchReport`` — the structure ``word_report.py`` renders to docx. The CSS
and bar-colour helper here are used by ``run_report.py``, which renders the
run-wide ``report.html``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .themes import Theme


@dataclass
class ModelResult:
    model: str
    score: int | None = None
    reasoning: str = ""
    locations: list[dict] = field(default_factory=list)
    chunked: bool = False
    total_tokens: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


@dataclass
class TranscriptResult:
    name: str
    models: list[ModelResult]

    @property
    def best_model(self) -> ModelResult | None:
        ok = [m for m in self.models if m.error is None and m.score is not None]
        if not ok:
            return None
        return max(ok, key=lambda m: m.score)

    @property
    def best_score(self) -> int | None:
        best = self.best_model
        return best.score if best else None


@dataclass
class BatchReport:
    theme: str
    transcripts: list[TranscriptResult]
    generated_at: str


def _model_result_from_data(model: str, data: dict) -> ModelResult:
    if "relevance_score" not in data:
        return ModelResult(model=model, error=data.get("error", "unknown error (malformed result)"))

    meta = data.get("_meta", {}) or {}
    usage = meta.get("usage", {}) or {}
    return ModelResult(
        model=model,
        score=data.get("relevance_score"),
        reasoning=data.get("reasoning", ""),
        locations=data.get("locations", []) or [],
        chunked=bool(data.get("chunked", False)),
        total_tokens=usage.get("total_tokens"),
        elapsed_seconds=meta.get("elapsed_seconds"),
    )


def build_report(theme: str, transcripts: dict[str, list[tuple[str, dict]]]) -> BatchReport:
    """Build a report directly from in-memory run payloads — one entry per
    transcript name, each a list of ``(model_name, payload_dict)`` pairs (as
    produced by ``cli.run_one_model``; a failed run's payload is
    ``{"model": ..., "error": ...}``).
    """
    transcript_results = [
        TranscriptResult(
            name=name,
            models=[_model_result_from_data(model, data) for model, data in model_payloads],
        )
        for name, model_payloads in transcripts.items()
    ]

    def sort_key(t: TranscriptResult):
        return (t.best_score is None, -(t.best_score or 0), t.name)

    transcript_results.sort(key=sort_key)
    return BatchReport(
        theme=theme,
        transcripts=transcript_results,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def _project_payload(model: str, data: dict, theme: Theme) -> dict:
    if "error" in data:
        return data  # failed runs pass through
    entry = (data.get("themes") or {}).get(theme.id)
    if entry is None:
        return {"model": model, "error": f"no result for theme {theme.id}"}

    meta = data.get("_meta", {}) or {}
    tokens = ((meta.get("theme_usage") or {}).get(theme.id) or {}).get("tokens")
    return {
        "relevance_score": entry.get("relevance_score"),
        "reasoning": entry.get("reasoning", ""),
        "locations": entry.get("locations", []) or [],
        "model_used": data.get("model_used"),
        "chunked": bool(data.get("chunked", False)),
        "_meta": {
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "usage": {"total_tokens": round(tokens) if tokens is not None else None},
        },
    }


def project_theme(
    transcripts: dict[str, list[tuple[str, dict]]], theme: Theme
) -> dict[str, list[tuple[str, dict]]]:
    """Project multi-theme payloads down to the single-theme shape
    ``build_report`` reads, for one ``theme``."""
    return {
        name: [(model, _project_payload(model, data, theme)) for model, data in payloads]
        for name, payloads in transcripts.items()
    }


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

BASE_CSS = r"""
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --seq-300: #6da7ec;
    --seq-500: #256abf;
    --seq-700: #0d366b;
    --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --seq-300: #5598e7;
      --seq-500: #2a78d6;
      --seq-700: #3987e5;
      --critical: #e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --seq-300: #5598e7;
    --seq-500: #2a78d6;
    --seq-700: #3987e5;
    --critical: #e66767;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 2rem 1rem 4rem;
    background: var(--page);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--text-secondary); margin-top: 0; }
  .stats { display: flex; gap: 1.5rem; margin: 1.25rem 0; flex-wrap: wrap; }
  .stat { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1rem; }
  .stat .num { font-size: 1.4rem; font-weight: 600; display: block; }
  .stat .label { font-size: 0.8rem; color: var(--text-secondary); }
  .filter-bar { margin: 1.5rem 0 0.75rem; }
  .filter-bar input {
    width: 100%;
    font-size: 0.95rem;
    padding: 0.6rem 0.85rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface-1);
    color: var(--text-primary);
  }
  .filter-bar input:focus { outline: 2px solid var(--seq-500); outline-offset: 1px; }
  table { width: 100%; border-collapse: collapse; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th, td { padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--grid); font-size: 0.9rem; }
  th { color: var(--text-secondary); font-weight: 600; cursor: pointer; user-select: none; }
  th:hover { color: var(--text-primary); }
  th.sort-active::after { content: " \25BC"; font-size: 0.7em; }
  tr:last-child td { border-bottom: none; }
  .col-rank { width: 3rem; color: var(--muted); }
  .col-score { width: 4rem; font-variant-numeric: tabular-nums; text-align: right; }
  .col-bar { width: 35%; }
  .col-name a { color: inherit; text-decoration: none; border-bottom: 1px solid var(--border); }
  .col-name a:hover { border-bottom-color: var(--seq-500); color: var(--seq-500); }
  .bar-track { height: 10px; background: var(--grid); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .badge { font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.4rem; }
  .badge-error { background: var(--critical); color: #fff; }
  .score-pill { font-variant-numeric: tabular-nums; background: var(--seq-500); color: #fff; border-radius: 999px; padding: 0.1rem 0.6rem; font-size: 0.85rem; margin-left: 0.5rem; }
  .no-matches { text-align: center; color: var(--muted); padding: 1.5rem !important; }
  h2 { font-size: 1.15rem; margin-top: 2.5rem; }
  .transcript-detail { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 0.6rem; padding: 0.6rem 1rem; scroll-margin-top: 1rem; }
  .transcript-detail summary { cursor: pointer; font-weight: 600; padding: 0.4rem 0.1rem; }
  .detail-body { margin-top: 0.75rem; }
  .run-meta { color: var(--text-secondary); font-size: 0.85rem; }
  .reasoning { line-height: 1.5; }
  .locations { list-style: none; padding: 0; margin: 0; }
  .locations li { padding: 0.5rem 0; border-top: 1px solid var(--grid); }
  .locations li:first-child { border-top: none; }
  .loc-title { font-weight: 600; display: block; margin-bottom: 0.15rem; }
  .loc-meta { font-size: 0.75rem; color: var(--muted); display: block; margin-bottom: 0.3rem; }
  blockquote { margin: 0.2rem 0; padding-left: 0.75rem; border-left: 2px solid var(--seq-500); color: var(--text-primary); }
  blockquote p { margin: 0 0 0.6rem; }
  blockquote p:last-child { margin-bottom: 0; }
  .chip { display: inline-block; background: var(--grid); border-radius: 999px; padding: 0.1rem 0.6rem; font-size: 0.8rem; margin: 0 0.3rem 0.3rem 0; }
  .other-models { margin: 0.5rem 0; }
  .muted { color: var(--muted); font-size: 0.85rem; }
  code { background: var(--grid); border-radius: 4px; padding: 0.05rem 0.3rem; }

  @media (max-width: 640px) {
    body { padding: 1.25rem 0.75rem 3rem; }
    .stats { gap: 0.6rem; }
    .stat { flex: 1 1 40%; padding: 0.6rem 0.75rem; }
    table, thead, tbody, th, td, tr { display: block; }
    thead { position: absolute; left: -9999px; top: -9999px; }
    tbody tr {
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 0.6rem;
      padding: 0.35rem 0.75rem;
    }
    tbody tr:last-child { margin-bottom: 0; }
    td {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      border-bottom: 1px solid var(--grid);
      padding: 0.5rem 0;
    }
    td:last-child { border-bottom: none; }
    td::before {
      content: attr(data-label);
      font-weight: 600;
      color: var(--text-secondary);
      font-size: 0.8rem;
      flex-shrink: 0;
    }
    .col-bar { flex-direction: column; align-items: stretch; }
    .col-bar .bar-track { width: 100%; margin-top: 0.25rem; }
    .col-name { text-align: right; }
  }
"""


def _bar_color_var(score: int) -> str:
    """Sequential single-hue (blue) bucket: low scores recede toward the
    surface, high scores get the strongest step — per the dataviz sequential
    convention (lightest = near zero)."""
    if score >= 70:
        return "var(--seq-700)"
    if score >= 40:
        return "var(--seq-500)"
    return "var(--seq-300)"
