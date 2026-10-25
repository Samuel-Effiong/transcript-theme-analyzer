"""Wrap the artifact fragment into a standalone, print-optimised HTML document
and render it to PDF with headless Chrome.

The artifact source is a body fragment (the Artifact host supplies the
doctype/head/body skeleton), so it needs wrapping before Chrome will treat it
as a full page. Print needs three further things the screen version doesn't:
the light palette pinned (a PDF has no viewer theme to follow), the collapsed
<details> in section 8 forced open (a printed page cannot be clicked), and
page-break rules so tables, figures and cards aren't sliced across sheets.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
SRC = HERE / "openrouter-proposal.html"
OUT_HTML = HERE / "proposal-print.html"

PRINT_CSS = """
/* ---------- print / PDF ---------- */
@page { size: A4; margin: 15mm 14mm 16mm; }

/* Background must sit on <html>, not only <body>: body's background box is
   painted at the element's own size, leaving the page margins white. The root
   element's background is what propagates across the whole canvas. */
html {
  background: var(--paper);
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}

body { background: var(--paper); font-size: 10.4pt; line-height: 1.55; }
.doc { max-width: 100%; padding: 0 0 8mm; }
.prose { max-width: 74ch; }

/* Screen shadows muddy on paper; keep the borders, drop the glow. */
.verdict, .panel, .tw, figure, .vs > div, .cat, details { box-shadow: none !important; }

/* Never slice a *small* block across a sheet. Tall blocks - the long price
   tables, the chart panel - are deliberately excluded: forcing them whole
   pushes them to the next sheet and leaves the current one blank, which is
   what produced the empty pages in the first render. They break internally
   instead, with the header row repeating. */
.note, .vs > div, .cat, details, .steps li, tr, figure {
  break-inside: avoid; page-break-inside: avoid;
}
thead { display: table-header-group; }
tfoot { display: table-footer-group; }

/* A heading stranded at the foot of a page is the classic print bug. */
h2, h3, .shead { break-after: avoid; page-break-after: avoid; }
.shead { margin-top: 34px; }
section { break-before: auto; }

/* Section 1 opens the argument; give it a clean sheet after the summary. */
section:first-of-type { break-before: page; page-break-before: always; }

/* Collapsible risk register is printed fully expanded. */
details { border: 1px solid var(--rule); }
details > summary { list-style: none; font-weight: 600; }
details > summary::before { content: "" !important; }
.details-body { padding-top: 14px; }

/* Tables scroll on screen; on paper they must fit the measure. */
.tw { overflow: visible; }
table { min-width: 0; font-size: 9.3pt; }
th, td { padding: 7px 12px; }
caption { padding: 13px 12px 10px; }

.masthead { padding-top: 0; }
.masthead h1 { font-size: 30pt; }
.chart { font-size: 9pt; }
.row-c { grid-template-columns: 150px 1fr; }
.axis { margin-left: 164px; }
/* Leave room for the value label sitting to the right of each bar. */
.track { padding-right: 52px; }
figure svg { max-height: 112mm; }
.cat-grid { grid-template-columns: 1fr 1fr; gap: 12px; }
.vs { grid-template-columns: 1fr 1fr; }

/* Links are dead on paper - keep them ink-coloured rather than accent-blue. */
a { color: var(--ink); text-decoration: none; }

footer { break-inside: avoid; }
"""


def build() -> str:
    src = SRC.read_text(encoding="utf-8")

    # The fragment leads with <title> and the font <link>; lift them into a head.
    title = re.search(r"<title>(.*?)</title>", src, re.S)
    title_text = title.group(1).strip() if title else "Proposal"

    # Force every disclosure open - a printed page has nothing to click.
    src = src.replace("<details>", "<details open>")

    return (
        "<!doctype html>\n"
        # Pin the light palette: a PDF has no viewer theme to inherit.
        '<html lang="en" data-theme="light">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=1100">\n'
        f"<title>{title_text}</title>\n"
        f"{src}\n"
        f"<style>{PRINT_CSS}</style>\n"
        "</head>\n<body>\n</body>\n</html>\n"
    )


def main() -> int:
    OUT_HTML.write_text(build(), encoding="utf-8")
    out_pdf = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "proposal.pdf")

    cmd = [
        "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        # Fonts and layout need a beat to settle before the snapshot.
        "--virtual-time-budget=12000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={out_pdf}",
        OUT_HTML.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    p = pathlib.Path(out_pdf)
    if not p.exists() or p.stat().st_size == 0:
        print(proc.stderr[-3000:] or proc.stdout[-3000:], file=sys.stderr)
        return 1
    print(f"wrote {out_pdf}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
