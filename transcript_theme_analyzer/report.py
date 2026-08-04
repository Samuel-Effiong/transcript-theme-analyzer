"""Shared results data model + the interactive HTML report.

``build_report`` turns the raw in-memory run payloads (as produced by
``cli.py``) into a ``BatchReport`` — the same structure ``word_report.py``
renders to docx. This module renders it to a single self-contained,
mobile-friendly, interactive ``report.html``.

Usage (regenerate later from the internal cache, without re-running analysis):
    python -m transcript_theme_analyzer.report --cache results/.raw_results.json
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape as h

from .cache import read_cache


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


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _bar_color_var(score: int) -> str:
    """Sequential single-hue (blue) bucket: low scores recede toward the
    surface, high scores get the strongest step — per the dataviz sequential
    convention (lightest = near zero)."""
    if score >= 70:
        return "var(--seq-700)"
    if score >= 40:
        return "var(--seq-500)"
    return "var(--seq-300)"


def _render_excerpt_html(excerpt: str) -> str:
    """Renders a (possibly multi-paragraph) excerpt as one <p> per source
    paragraph inside the blockquote, instead of flattening paragraph breaks
    into a single run-on line."""
    paragraphs = [p.strip() for p in excerpt.split("\n\n") if p.strip()] or [excerpt.strip()]
    inner = "".join(f"<p>{h(p)}</p>" for p in paragraphs)
    return f"<blockquote>{inner}</blockquote>"


def _render_locations_html(locations: list[dict], limit: int) -> str:
    if not locations:
        return "<p class='muted'>No specific locations recorded.</p>"
    items = []
    for loc in locations[:limit]:
        ts = loc.get("timestamp")
        speaker = loc.get("speaker")
        meta_bits = [b for b in (ts, speaker) if b]
        meta_str = h(" · ".join(meta_bits)) if meta_bits else ""
        title = h(loc.get("title") or "Relevant Passage")
        excerpt_html = _render_excerpt_html(loc.get("excerpt") or "")
        items.append(
            f"<li><span class='loc-title'>{title}</span>"
            f"<span class='loc-meta'>{meta_str}</span>"
            f"{excerpt_html}</li>"
        )
    more = len(locations) - limit
    more_note = f"<p class='muted'>+{more} more location(s).</p>" if more > 0 else ""
    return f"<ul class='locations'>{''.join(items)}</ul>{more_note}"


def _render_transcript_row(rank: int, t: TranscriptResult) -> str:
    score = t.best_score
    score_display = str(score) if score is not None else "—"
    bar_width = score if score is not None else 0
    bar_color = _bar_color_var(score) if score is not None else "var(--muted)"
    status = "" if score is not None else "<span class='badge badge-error'>FAILED</span>"
    return f"""
    <tr class="rank-row" data-score="{score if score is not None else -1}" data-name="{h(t.name)}">
      <td class="col-rank" data-label="#">{rank}</td>
      <td class="col-name" data-label="Transcript"><a href="#detail-{rank}">{h(t.name)}</a> {status}</td>
      <td class="col-score" data-label="Score">{score_display}</td>
      <td class="col-bar" data-label="Relevance">
        <div class="bar-track">
          <div class="bar-fill" style="width:{bar_width}%; background:{bar_color};"></div>
        </div>
      </td>
    </tr>
    """


def _render_transcript_detail(rank: int, t: TranscriptResult, top_locations: int) -> str:
    best = t.best_model
    if best is None:
        errors = "".join(f"<li><strong>{h(m.model)}</strong>: {h(m.error or '')}</li>" for m in t.models)
        return f"""
        <details class="transcript-detail" id="detail-{rank}">
          <summary>{rank}. {h(t.name)} <span class="badge badge-error">FAILED</span></summary>
          <div class="detail-body">
            <p>All model runs failed for this transcript:</p>
            <ul>{errors}</ul>
          </div>
        </details>
        """

    other_models_html = ""
    if len(t.models) > 1:
        chips = "".join(
            f"<span class='chip'>{h(m.model)}: {m.score if m.error is None else 'error'}</span>"
            for m in t.models if m is not best
        )
        other_models_html = f"<div class='other-models'>{chips}</div>"

    meta_bits = [f"model: <code>{h(best.model)}</code>"]
    if best.chunked:
        meta_bits.append("chunked (map-reduce)")
    if best.total_tokens is not None:
        meta_bits.append(f"{best.total_tokens:,} tokens")
    if best.elapsed_seconds is not None:
        meta_bits.append(f"{best.elapsed_seconds:.1f}s")

    return f"""
    <details class="transcript-detail" id="detail-{rank}">
      <summary>{rank}. {h(t.name)} <span class="score-pill">{best.score}</span></summary>
      <div class="detail-body">
        <p class="run-meta">{' &middot; '.join(meta_bits)}</p>
        {other_models_html}
        <h4>Reasoning</h4>
        <p class="reasoning">{h(best.reasoning)}</p>
        <h4>Key locations</h4>
        {_render_locations_html(best.locations, top_locations)}
      </div>
    </details>
    """


def render_html(report: BatchReport, top_locations: int = 5) -> str:
    rows = "".join(_render_transcript_row(i, t) for i, t in enumerate(report.transcripts, start=1))
    details = "".join(_render_transcript_detail(i, t, top_locations) for i, t in enumerate(report.transcripts, start=1))
    n_ok = sum(1 for t in report.transcripts if t.best_score is not None)
    n_failed = len(report.transcripts) - n_ok

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theme Analysis: {h(report.theme)}</title>
<style>
  :root {{
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
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
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
    }}
  }}
  :root[data-theme="dark"] {{
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
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 2rem 1rem 4rem;
    background: var(--page);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--text-secondary); margin-top: 0; }}
  .stats {{ display: flex; gap: 1.5rem; margin: 1.25rem 0; flex-wrap: wrap; }}
  .stat {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1rem; }}
  .stat .num {{ font-size: 1.4rem; font-weight: 600; display: block; }}
  .stat .label {{ font-size: 0.8rem; color: var(--text-secondary); }}
  .filter-bar {{ margin: 1.5rem 0 0.75rem; }}
  .filter-bar input {{
    width: 100%;
    font-size: 0.95rem;
    padding: 0.6rem 0.85rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface-1);
    color: var(--text-primary);
  }}
  .filter-bar input:focus {{ outline: 2px solid var(--seq-500); outline-offset: 1px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--grid); font-size: 0.9rem; }}
  th {{ color: var(--text-secondary); font-weight: 600; cursor: pointer; user-select: none; }}
  th:hover {{ color: var(--text-primary); }}
  th.sort-active::after {{ content: " \\25BC"; font-size: 0.7em; }}
  tr:last-child td {{ border-bottom: none; }}
  .col-rank {{ width: 3rem; color: var(--muted); }}
  .col-score {{ width: 4rem; font-variant-numeric: tabular-nums; text-align: right; }}
  .col-bar {{ width: 35%; }}
  .col-name a {{ color: inherit; text-decoration: none; border-bottom: 1px solid var(--border); }}
  .col-name a:hover {{ border-bottom-color: var(--seq-500); color: var(--seq-500); }}
  .bar-track {{ height: 10px; background: var(--grid); border-radius: 4px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .badge {{ font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.4rem; }}
  .badge-error {{ background: var(--critical); color: #fff; }}
  .score-pill {{ font-variant-numeric: tabular-nums; background: var(--seq-500); color: #fff; border-radius: 999px; padding: 0.1rem 0.6rem; font-size: 0.85rem; margin-left: 0.5rem; }}
  .no-matches {{ text-align: center; color: var(--muted); padding: 1.5rem !important; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.5rem; }}
  .transcript-detail {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 0.6rem; padding: 0.6rem 1rem; scroll-margin-top: 1rem; }}
  .transcript-detail summary {{ cursor: pointer; font-weight: 600; padding: 0.4rem 0.1rem; }}
  .detail-body {{ margin-top: 0.75rem; }}
  .run-meta {{ color: var(--text-secondary); font-size: 0.85rem; }}
  .reasoning {{ line-height: 1.5; }}
  .locations {{ list-style: none; padding: 0; margin: 0; }}
  .locations li {{ padding: 0.5rem 0; border-top: 1px solid var(--grid); }}
  .locations li:first-child {{ border-top: none; }}
  .loc-title {{ font-weight: 600; display: block; margin-bottom: 0.15rem; }}
  .loc-meta {{ font-size: 0.75rem; color: var(--muted); display: block; margin-bottom: 0.3rem; }}
  blockquote {{ margin: 0.2rem 0; padding-left: 0.75rem; border-left: 2px solid var(--seq-500); color: var(--text-primary); }}
  blockquote p {{ margin: 0 0 0.6rem; }}
  blockquote p:last-child {{ margin-bottom: 0; }}
  .chip {{ display: inline-block; background: var(--grid); border-radius: 999px; padding: 0.1rem 0.6rem; font-size: 0.8rem; margin: 0 0.3rem 0.3rem 0; }}
  .other-models {{ margin: 0.5rem 0; }}
  .muted {{ color: var(--muted); font-size: 0.85rem; }}
  code {{ background: var(--grid); border-radius: 4px; padding: 0.05rem 0.3rem; }}

  @media (max-width: 640px) {{
    body {{ padding: 1.25rem 0.75rem 3rem; }}
    .stats {{ gap: 0.6rem; }}
    .stat {{ flex: 1 1 40%; padding: 0.6rem 0.75rem; }}
    table, thead, tbody, th, td, tr {{ display: block; }}
    thead {{ position: absolute; left: -9999px; top: -9999px; }}
    tbody tr {{
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 0.6rem;
      padding: 0.35rem 0.75rem;
    }}
    tbody tr:last-child {{ margin-bottom: 0; }}
    td {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      border-bottom: 1px solid var(--grid);
      padding: 0.5rem 0;
    }}
    td:last-child {{ border-bottom: none; }}
    td::before {{
      content: attr(data-label);
      font-weight: 600;
      color: var(--text-secondary);
      font-size: 0.8rem;
      flex-shrink: 0;
    }}
    .col-bar {{ flex-direction: column; align-items: stretch; }}
    .col-bar .bar-track {{ width: 100%; margin-top: 0.25rem; }}
    .col-name {{ text-align: right; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Transcript Theme Analysis</h1>
    <p class="subtitle">Theme: &ldquo;{h(report.theme)}&rdquo; &middot; generated {h(report.generated_at)}</p>

    <div class="stats">
      <div class="stat"><span class="num">{len(report.transcripts)}</span><span class="label">transcripts</span></div>
      <div class="stat"><span class="num">{n_ok}</span><span class="label">scored</span></div>
      <div class="stat"><span class="num">{n_failed}</span><span class="label">failed</span></div>
    </div>

    <div class="filter-bar">
      <input type="search" id="transcript-filter" placeholder="Filter transcripts by name…" aria-label="Filter transcripts by name">
    </div>

    <table id="rank-table">
      <thead>
        <tr>
          <th data-sort="rank">#</th>
          <th data-sort="name">Transcript</th>
          <th data-sort="score" class="sort-active">Score</th>
          <th>Relevance</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>

    <h2>Details</h2>
    {details}
  </div>

  <script>
    (function() {{
      var table = document.getElementById('rank-table');
      var tbody = table.querySelector('tbody');
      var headers = table.querySelectorAll('th[data-sort]');
      var state = {{ key: 'score', dir: -1 }};

      function rowsArray() {{ return Array.prototype.slice.call(tbody.querySelectorAll('tr.rank-row')); }}

      function sortBy(key, dir) {{
        var rows = rowsArray();
        rows.sort(function(a, b) {{
          var av, bv;
          if (key === 'score') {{
            av = parseFloat(a.dataset.score); bv = parseFloat(b.dataset.score);
          }} else if (key === 'name') {{
            av = a.dataset.name.toLowerCase(); bv = b.dataset.name.toLowerCase();
            return dir * (av < bv ? -1 : av > bv ? 1 : 0);
          }} else {{
            av = rows.indexOf(a); bv = rows.indexOf(b);
          }}
          return dir * (av - bv);
        }});
        rows.forEach(function(r) {{ tbody.appendChild(r); }});
      }}

      headers.forEach(function(th) {{
        th.addEventListener('click', function() {{
          var key = th.dataset.sort;
          state.dir = (state.key === key) ? -state.dir : -1;
          state.key = key;
          headers.forEach(function(hh) {{ hh.classList.remove('sort-active'); }});
          th.classList.add('sort-active');
          sortBy(key, state.dir);
        }});
      }});

      // Filter box: hide non-matching rows by transcript name; show a
      // "no matches" placeholder row when the filter empties the table.
      var filterInput = document.getElementById('transcript-filter');
      var noMatchRow = document.createElement('tr');
      noMatchRow.innerHTML = '<td class="no-matches" colspan="4">No transcripts match your filter.</td>';
      filterInput.addEventListener('input', function() {{
        var query = filterInput.value.trim().toLowerCase();
        var rows = rowsArray();
        var anyVisible = false;
        rows.forEach(function(r) {{
          var match = r.dataset.name.toLowerCase().indexOf(query) !== -1;
          r.style.display = match ? '' : 'none';
          if (match) anyVisible = true;
        }});
        if (!anyVisible && !tbody.contains(noMatchRow)) {{
          tbody.appendChild(noMatchRow);
        }} else if (anyVisible && tbody.contains(noMatchRow)) {{
          tbody.removeChild(noMatchRow);
        }}
      }});

      // Clicking a transcript name jumps to its detail section and expands it
      // (plain anchor scrolling doesn't set the `open` attribute on <details>).
      document.querySelectorAll('.col-name a').forEach(function(a) {{
        a.addEventListener('click', function() {{
          var id = a.getAttribute('href').slice(1);
          var el = document.getElementById(id);
          if (el) el.open = true;
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def write_html(report: BatchReport, out_path: str, top_locations: int = 5) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_html(report, top_locations=top_locations))
    return out_path


def write_html_from_data(
    theme: str,
    transcripts: dict[str, list[tuple[str, dict]]],
    out_path: str,
    top_locations: int = 5,
) -> str:
    """Build and write ``report.html`` directly from in-memory run payloads
    (see ``build_report``), without touching disk for input."""
    report = build_report(theme, transcripts)
    return write_html(report, out_path, top_locations=top_locations)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate report.html from the internal analysis cache"
    )
    parser.add_argument("--cache", required=True, help="Path to the .raw_results.json cache file")
    parser.add_argument("--out", default=None, help="Output .html path (defaults to report.html next to the cache file)")
    parser.add_argument("--top-locations", type=int, default=5, help="Max locations to show per transcript")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_path = args.out or os.path.join(os.path.dirname(args.cache) or ".", "report.html")
    try:
        theme, transcripts = read_cache(args.cache)
    except (FileNotFoundError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    report = build_report(theme, transcripts)
    write_html(report, out_path, top_locations=args.top_locations)
    print(f"[html] {out_path}")


if __name__ == "__main__":
    main()
