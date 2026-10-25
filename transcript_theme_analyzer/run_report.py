"""Run-wide HTML page: every transcript x every theme, plus cost reporting.

``build_run_report`` turns the multi-theme run payloads (see ``report`` for
the ``transcripts`` shape) into a ``RunReport``; ``render_run_html`` renders
one self-contained page (inline CSS/JS only). The grid and cost sections are
plain HTML; a cell's reasoning and passages sit in one JSON blob and are
built into the page only when that cell is clicked, which keeps a
thousands-of-transcripts page small and quick to open.

Usage (regenerate later from the internal cache, without re-running analysis):
    python -m transcript_theme_analyzer.run_report --cache results/.raw_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape as h

from .cache import read_cache
from .report import BASE_CSS, _bar_color_var, is_found
from .themes import Theme


@dataclass
class Cell:
    model: str
    score: int
    explicitness: str
    reasoning: str
    locations: list[dict]
    # The other models' scores for this theme, payload order; None = failed.
    others: list[tuple[str, int | None]] = field(default_factory=list)


@dataclass
class RunRow:
    name: str
    cells: dict[str, Cell | None]  # None = no model succeeded for that theme
    tokens: int = 0
    cost: float | None = None
    has_usage: bool = False


@dataclass
class ThemeCost:
    theme: Theme
    found: int
    tokens: int
    cost: float | None


@dataclass
class CostSummary:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None
    discount: float | None = None
    cost_source: str | None = None
    per_theme: list[ThemeCost] = field(default_factory=list)


@dataclass
class RunReport:
    themes: list[Theme]
    rows: list[RunRow]
    costs: CostSummary
    generated_at: str


def _sum_or_none(values: list[float | None]) -> float | None:
    """None if empty or any value is unknown (same rule as RunStats.summary)."""
    if not values or any(v is None for v in values):
        return None
    return sum(values)


def _ok_payloads(payloads: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    return [(m, d) for m, d in payloads if "error" not in d and isinstance(d.get("themes"), dict)]


def _theme_score(data: dict, theme: Theme) -> int | None:
    themes = data.get("themes")
    if "error" in data or not isinstance(themes, dict):
        return None
    return (themes.get(theme.id) or {}).get("relevance_score")


def _best_cell(payloads: list[tuple[str, dict]], theme: Theme) -> Cell | None:
    best: Cell | None = None
    best_data: dict | None = None
    for model, data in _ok_payloads(payloads):
        e = data["themes"].get(theme.id)
        if not e or e.get("relevance_score") is None:
            continue
        if best is None or e["relevance_score"] > best.score:
            best_data = data
            best = Cell(
                model=model,
                score=e["relevance_score"],
                explicitness=e.get("explicitness", ""),
                reasoning=e.get("reasoning", ""),
                locations=e.get("locations", []) or [],
            )
    if best is not None:
        best.others = [(m, _theme_score(d, theme)) for m, d in payloads if d is not best_data]
    return best


def _counted(payloads: list[tuple[str, dict]]) -> list[dict]:
    """Payloads whose usage counts: every successful run, plus a failed run
    that made calls. An error with no usage (unreadable transcript, failure
    before any response) paid for nothing and must not make a figure n/a."""
    return [
        d for _, d in payloads
        if "error" not in d or (d.get("_meta") or {}).get("usage")
    ]


def _usages(payloads: list[tuple[str, dict]]) -> list[dict]:
    return [((d.get("_meta") or {}).get("usage") or {}) for d in _counted(payloads)]


def _tok(usages: list[dict], key: str) -> int:
    return sum(u.get(key) or 0 for u in usages)


def _source(usages: list[dict]) -> str | None:
    sources = {u.get("cost_source") for u in usages if u.get("cost_source")}
    if not sources:
        return None
    return sources.pop() if len(sources) == 1 else "mixed"


def build_run_report(themes: list[Theme], transcripts: dict[str, list[tuple[str, dict]]]) -> RunReport:
    rows: list[RunRow] = []
    all_usages: list[dict] = []
    theme_tokens = {t.id: 0.0 for t in themes}
    theme_costs: dict[str, list[float | None]] = {t.id: [] for t in themes}

    for name, payloads in transcripts.items():
        # Usage counts from every payload that has it: a failed run still
        # paid for the calls it made.
        usages = _usages(payloads)
        all_usages += usages
        for d in _counted(payloads):
            tu = (d.get("_meta") or {}).get("theme_usage") or {}
            for t in themes:
                u = tu.get(t.id) or {}
                theme_tokens[t.id] += u.get("tokens") or 0
                theme_costs[t.id].append(u.get("cost"))
        rows.append(
            RunRow(
                name=name,
                cells={t.id: _best_cell(payloads, t) for t in themes},
                tokens=_tok(usages, "total_tokens"),
                cost=_sum_or_none([u.get("cost") for u in usages]),
                has_usage=bool(usages),
            )
        )

    known_discounts = [u["discount"] for u in all_usages if u.get("discount") is not None]
    costs = CostSummary(
        prompt_tokens=_tok(all_usages, "prompt_tokens"),
        completion_tokens=_tok(all_usages, "completion_tokens"),
        cached_tokens=_tok(all_usages, "cached_tokens"),
        total_tokens=_tok(all_usages, "total_tokens"),
        cost=_sum_or_none([u.get("cost") for u in all_usages]),
        discount=sum(known_discounts) if known_discounts else None,
        cost_source=_source(all_usages),
        per_theme=[
            ThemeCost(
                theme=t,
                found=sum(1 for r in rows if r.cells[t.id] and not _is_absent(r.cells[t.id])),
                tokens=round(theme_tokens[t.id]),
                cost=_sum_or_none(theme_costs[t.id]),
            )
            for t in themes
        ],
    )
    return RunReport(
        themes=themes,
        rows=rows,
        costs=costs,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def _money(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.4f}"


def _is_absent(c: Cell) -> bool:
    return not is_found(c.score, c.explicitness)


def _render_cost_summary(c: CostSummary) -> str:
    source_lines = {
        "openrouter": ["Actual amounts charged by OpenRouter."],
        "prices": ["Estimated from prices in settings (LLM_PRICE_*) or the OpenRouter price list."],
        "mixed": [
            "Actual amounts charged by OpenRouter.",
            "Estimated from prices in settings (LLM_PRICE_*) or the OpenRouter price list.",
        ],
    }
    # Unknown cost: say nothing about where it came from.
    lines = source_lines.get(c.cost_source, []) if c.cost is not None else []
    source_html = "".join(f"<p class='muted'>{h(s)}</p>" for s in lines)
    return f"""
    <h2>Cost</h2>
    <div class="stats">
      <div class="stat"><span class="num">{c.total_tokens:,}</span><span class="label">total tokens</span></div>
      <div class="stat"><span class="num">{c.prompt_tokens:,}</span><span class="label">input tokens</span></div>
      <div class="stat"><span class="num">{c.cached_tokens:,}</span><span class="label">cached tokens</span></div>
      <div class="stat"><span class="num">{c.completion_tokens:,}</span><span class="label">output tokens</span></div>
      <div class="stat"><span class="num">{_money(c.cost)}</span><span class="label">total cost</span></div>
      <div class="stat"><span class="num">{_money(c.discount)}</span><span class="label">provider discount (est.)</span></div>
    </div>
    <p class='muted'>Provider discount (est., from list prices): {_money(c.discount)}</p>
    {source_html}
    """


def _theme_text_div(theme: Theme) -> str:
    """The full theme text, printed once under the label when it is longer."""
    if theme.text == theme.name:
        return ""
    return f'<div class="theme-text">{h(theme.text)}</div>'


def _theme_title_attr(theme: Theme) -> str:
    return "" if theme.text == theme.name else f' title="{h(theme.text)}"'


def _render_theme_table(c: CostSummary, n_transcripts: int) -> str:
    rows = "".join(
        f"""<tr>
      <td data-label="Theme">{h(tc.theme.name)}{_theme_text_div(tc.theme)}</td>
      <td data-label="Found in">{tc.found} / {n_transcripts}</td>
      <td data-label="Tokens (est.)">{tc.tokens:,}</td>
      <td data-label="Cost (est.)">{_money(tc.cost)}</td>
    </tr>"""
        for tc in c.per_theme
    )
    return f"""
    <h2>Per theme</h2>
    <table class="theme-table">
      <thead><tr><th>Theme</th><th>Transcripts where found</th><th>Tokens (est.)</th><th>Cost (est.)</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _render_cell(i: int, t: Theme, cell: Cell | None) -> str:
    attrs = f'data-theme="{h(t.id)}"'
    if cell is None:
        return f'<td class="cell cell-failed" {attrs} data-score="-1"><span class="badge badge-error">FAILED</span></td>'
    if _is_absent(cell):
        return f'<td class="cell cell-absent" {attrs} data-score="0">–</td>'
    cls = "cell cell-tangential" if cell.explicitness == "tangential" else "cell"
    fg = "#fff" if cell.score >= 40 else "#0b0b0b"
    # data-row: this cell has a detail entry (rows[i - 1] in the JSON blob).
    return (
        f'<td class="{cls}" {attrs} data-score="{cell.score}" data-row="{i}" tabindex="0" '
        f'style="background:{_bar_color_var(cell.score)}; color:{fg};">{cell.score}</td>'
    )


def _render_grid(r: RunReport) -> str:
    head = "".join(
        f'<th data-theme="{h(t.id)}"{_theme_title_attr(t)}>{h(t.name)}</th>' for t in r.themes
    )
    body = []
    for i, row in enumerate(r.rows, start=1):
        cells = "".join(_render_cell(i, t, row.cells[t.id]) for t in r.themes)
        cost_txt = f"{row.tokens:,} / {_money(row.cost)}" if row.has_usage else "n/a"
        body.append(
            f'<tr class="grid-row" data-idx="{i}" data-name="{h(row.name)}">'
            f'<th class="row-name" scope="row">{h(row.name)}</th>{cells}'
            f'<td class="col-usage">{cost_txt}</td></tr>'
        )
    return f"""
    <h2>Results</h2>
    <div class="filter-bar controls">
      <select id="theme-picker" aria-label="Show theme">
        <option value="">All themes</option>
        {''.join(f'<option value="{h(t.id)}">{h(t.name)}</option>' for t in r.themes)}
      </select>
      <input type="search" id="transcript-filter" placeholder="Filter transcripts by name…" aria-label="Filter transcripts by name">
    </div>
    <div class="grid-scroll">
      <table class="grid" id="grid">
        <thead><tr><th>Transcript</th>{head}<th class="col-usage">Tokens / Cost</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    """


def _detail_data(r: RunReport, top_locations: int) -> dict:
    """What the click-to-open panel needs, for found cells only. A passage
    reported under several themes is stored once per transcript."""
    rows = []
    for row in r.rows:
        locations: list[dict] = []
        seen: dict[str, int] = {}
        cells = {}
        for t in r.themes:
            c = row.cells[t.id]
            if c is None or _is_absent(c):
                continue
            refs = []
            for loc in c.locations[:top_locations]:
                key = json.dumps(loc, sort_keys=True)
                if key not in seen:
                    seen[key] = len(locations)
                    locations.append(loc)
                refs.append(seen[key])
            cells[t.id] = {
                "model": c.model,
                "score": c.score,
                "explicitness": c.explicitness,
                "reasoning": c.reasoning,
                "locations": refs,
                "more": max(len(c.locations) - top_locations, 0),
                "others": [[m, score] for m, score in c.others],
            }
        rows.append({"name": row.name, "locations": locations, "cells": cells})
    return {"themes": {t.id: t.name for t in r.themes}, "rows": rows}


def _json_for_script(data: dict) -> str:
    """JSON safe inside <script>: no "<" survives, so no "</script>" either.
    ``\\u003c`` is still "<" once JSON.parse has read it."""
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


EXTRA_CSS = r"""
    .theme-text { font-size: 0.85em; color: var(--text-secondary); margin-top: 0.2rem; font-weight: normal; }
  .wrap { max-width: 1100px; }
  .controls { display: flex; gap: 0.75rem; flex-wrap: wrap; }
  .controls select { font-size: 0.95rem; padding: 0.6rem 0.85rem; border-radius: 8px; border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary); }
  .controls input { flex: 1 1 14rem; width: auto; }
  .grid-scroll { overflow-x: auto; max-width: 100%; border: 1px solid var(--border); border-radius: 8px; }
  /* keep the grid a real table even under BASE_CSS's mobile card layout */
  .grid-scroll table { display: table; width: auto; min-width: 100%; border: none; }
  .grid-scroll thead { display: table-header-group; position: static; }
  .grid-scroll tbody { display: table-row-group; }
  .grid-scroll tr { display: table-row; }
  .grid-scroll th, .grid-scroll td { display: table-cell; white-space: nowrap; cursor: default; }
  .grid-scroll tbody tr { border: 0; margin: 0; padding: 0; border-radius: 0; }
  .grid-scroll td::before { content: none; }
  .grid-scroll .row-name { position: sticky; left: 0; background: var(--surface-1); text-align: left; font-weight: 500; }
  .cell { text-align: center; font-variant-numeric: tabular-nums; font-weight: 600; min-width: 4rem; }
  .cell[data-row] { cursor: pointer; }
  .cell-tangential { outline: 2px dashed var(--muted); outline-offset: -4px; font-style: italic; }
  .cell-absent { color: var(--muted); }
  .col-usage { font-variant-numeric: tabular-nums; color: var(--text-secondary); }
  .muted { margin: 0.25rem 0; }
  .stat .label { display: block; }
"""

JS = r"""
    (function() {
      var table = document.getElementById('grid');
      var tbody = table.querySelector('tbody');
      var picker = document.getElementById('theme-picker');
      var filter = document.getElementById('transcript-filter');
      var panel = document.getElementById('detail');
      var data = JSON.parse(document.getElementById('detail-data').textContent);

      function rows() { return Array.prototype.slice.call(tbody.querySelectorAll('tr.grid-row')); }

      function scoreFor(row, theme) {
        var cells = row.querySelectorAll('td.cell');
        for (var i = 0; i < cells.length; i++) {
          if (cells[i].dataset.theme === theme) return parseFloat(cells[i].dataset.score);
        }
        return -1;
      }

      function apply() {
        var theme = picker.value;
        var query = filter.value.trim().toLowerCase();
        table.querySelectorAll('[data-theme]').forEach(function(el) {
          el.style.display = (!theme || el.dataset.theme === theme) ? '' : 'none';
        });
        var rs = rows();
        rs.sort(function(a, b) {
          if (theme) {
            var d = scoreFor(b, theme) - scoreFor(a, theme);
            if (d) return d;
          }
          return parseInt(a.dataset.idx, 10) - parseInt(b.dataset.idx, 10);
        });
        rs.forEach(function(r) {
          r.style.display = r.dataset.name.toLowerCase().indexOf(query) !== -1 ? '' : 'none';
          tbody.appendChild(r);
        });
      }

      var timer = null;
      picker.addEventListener('change', apply);
      filter.addEventListener('input', function() {
        clearTimeout(timer);
        timer = setTimeout(apply, 150);
      });

      // Every piece of data goes in via textContent, never as markup.
      function el(tag, cls, text) {
        var node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text != null) node.textContent = text;
        return node;
      }

      function excerpt(text) {
        var quote = el('blockquote');
        var parts = (text || '').split('\n\n').map(function(p) { return p.trim(); })
          .filter(function(p) { return p; });
        if (!parts.length) parts = [(text || '').trim()];
        parts.forEach(function(p) { quote.appendChild(el('p', null, p)); });
        return quote;
      }

      function show(rowIdx, themeId) {
        var row = data.rows[rowIdx - 1];
        var c = row && row.cells[themeId];
        if (!c) return;
        panel.textContent = '';
        var head = el('h3', null, row.name + ' \u00b7 ' + data.themes[themeId] + ' ');
        head.appendChild(el('span', 'score-pill', String(c.score)));
        panel.appendChild(head);
        var meta = el('p', 'run-meta', 'model: ');
        meta.appendChild(el('code', null, c.model));
        meta.appendChild(document.createTextNode(' \u00b7 ' + c.explicitness));
        panel.appendChild(meta);
        if (c.others.length) {
          panel.appendChild(el('p', 'run-meta other-models', 'other models: ' + c.others.map(function(o) {
            return o[0] + '=' + (o[1] === null ? 'error' : o[1]);
          }).join(' / ')));
        }
        panel.appendChild(el('h4', null, 'Reasoning'));
        panel.appendChild(el('p', 'reasoning', c.reasoning));
        panel.appendChild(el('h4', null, 'Key locations'));
        if (!c.locations.length) {
          panel.appendChild(el('p', 'muted', 'No specific locations recorded.'));
        } else {
          var list = el('ul', 'locations');
          c.locations.forEach(function(i) {
            var loc = row.locations[i];
            var li = el('li');
            li.appendChild(el('span', 'loc-title', loc.title || 'Relevant Passage'));
            li.appendChild(el('span', 'loc-meta', [loc.timestamp, loc.speaker].filter(function(b) { return b; }).join(' \u00b7 ')));
            li.appendChild(excerpt(loc.excerpt));
            list.appendChild(li);
          });
          panel.appendChild(list);
        }
        if (c.more > 0) panel.appendChild(el('p', 'muted', '+' + c.more + ' more location(s).'));
        panel.hidden = false;
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

      function open(e) {
        var td = e.target.closest ? e.target.closest('td.cell') : null;
        if (td && td.dataset.row) show(parseInt(td.dataset.row, 10), td.dataset.theme);
      }
      table.addEventListener('click', open);
      table.addEventListener('keydown', function(e) { if (e.key === 'Enter') open(e); });
    })();
"""


def render_run_html(report: RunReport, top_locations: int = 5) -> str:
    theme_names = ", ".join(t.name for t in report.themes)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theme Analysis: {h(theme_names)}</title>
<style>
{BASE_CSS}
{EXTRA_CSS}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Transcript Theme Analysis</h1>
    <p class="subtitle">Themes: {h(theme_names)} &middot; generated {h(report.generated_at)}</p>

    {_render_cost_summary(report.costs)}
    {_render_theme_table(report.costs, len(report.rows))}
    {_render_grid(report)}

    <h2>Details</h2>
    <p class="muted">Click a score in the grid to see its reasoning and passages here.</p>
    <section class="transcript-detail" id="detail" hidden></section>
  </div>

  <script type="application/json" id="detail-data">{_json_for_script(_detail_data(report, top_locations))}</script>
  <script>{JS}  </script>
</body>
</html>
"""


def write_run_html(
    themes: list[Theme],
    transcripts: dict[str, list[tuple[str, dict]]],
    out_path: str,
    top_locations: int = 5,
) -> str:
    report = build_run_report(themes, transcripts)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_run_html(report, top_locations=top_locations))
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate report.html from the internal analysis cache"
    )
    parser.add_argument("--cache", required=True, help="Path to the .raw_results.json cache file")
    parser.add_argument("--out", default=None, help="Output .html path (defaults to report.html next to the cache file)")
    parser.add_argument("--top-locations", type=int, default=5, help="Max locations to show per transcript and theme")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_path = args.out or os.path.join(os.path.dirname(args.cache) or ".", "report.html")
    try:
        themes, transcripts = read_cache(args.cache)
    except (ValueError, OSError) as exc:  # OSError covers FileNotFoundError
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    write_run_html(themes, transcripts, out_path, top_locations=args.top_locations)
    print(f"[report:html] {out_path}")


if __name__ == "__main__":
    main()
