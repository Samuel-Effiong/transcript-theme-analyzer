#!/usr/bin/env python3
"""Render an artifact HTML page to a print-ready PDF via headless Chrome.

The artifact files in ``docs/`` are page *fragments*: the publishing pipeline
wraps them in ``<!doctype html><head>...</head><body>`` at deploy time, so they
have no document shell of their own and cannot be opened directly by a browser.
This script supplies that shell, layers a print stylesheet on top, and drives
Chrome's ``--print-to-pdf``.

The print stylesheet is kept *here* rather than in the source document so the
published web page stays free of print concerns, and so re-exporting after an
edit is one command:

    python tools/build_pdf.py docs/openrouter-decision-proposal.html
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

CHROME_CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)

# Headless Chrome does not reliably fetch webfonts before the print snapshot,
# and a PDF should not depend on the network at render time anyway. We resolve
# the Google Fonts CSS ourselves, inline every face as a data URI, and cache the
# result so later builds are fully offline.
FONT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fontcache")
FONT_CSS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=IBM+Plex+Mono:wght@400;500"
    "&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400"
    "&family=Public+Sans:ital,wght@0,400;0,500;0,600;1,400"
    "&display=swap"
)
# Asking as a browser is what makes Google serve woff2 rather than legacy TTF.
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# The document is English; the other subsets would triple the embedded weight.
WANTED_SUBSETS = ("latin", "latin-ext")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def inline_fonts() -> str:
    """Return a <style> block with every needed face embedded as a data URI."""
    cached = os.path.join(FONT_CACHE, "fonts.css")
    if os.path.exists(cached):
        with open(cached, "r", encoding="utf-8") as f:
            return "<style>\n" + f.read() + "\n</style>"

    try:
        css = _get(FONT_CSS_URL).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - degrade to system fonts, don't fail
        sys.stderr.write(
            f"warning: could not fetch webfont CSS ({exc}); "
            "the PDF will use fallback fonts\n"
        )
        return ""

    # Google emits a "/* subset */" comment before each @font-face block.
    blocks = re.findall(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", css)
    kept, seen = [], {}
    for subset, block in blocks:
        if subset not in WANTED_SUBSETS:
            continue
        m = re.search(r"url\((https://[^)]+\.woff2)\)", block)
        if not m:
            continue
        url = m.group(1)
        if url not in seen:
            try:
                seen[url] = base64.b64encode(_get(url)).decode("ascii")
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"warning: font fetch failed for {url}: {exc}\n")
                continue
        kept.append(
            block.replace(
                f"url({url})",
                f"url(data:font/woff2;base64,{seen[url]})",
            )
        )

    if not kept:
        return ""

    out = "\n".join(kept)
    os.makedirs(FONT_CACHE, exist_ok=True)
    with open(cached, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Embedded {len(kept)} font faces ({len(out) / 1024:.0f} KB) -> {cached}")
    return "<style>\n" + out + "\n</style>"

# Layered over the document's own <style>. Only concerns that differ on paper:
# a fixed light theme, a single column, and break control.
PRINT_CSS = """
<style>
  /* Paper is always light. The document guards its dark rules with
     :root:not([data-theme="light"]), so stamping the attribute on <html>
     wins in both the media-query and the explicit-toggle direction. */
  @page {
    size: A4;
    margin: 17mm 15mm 19mm;
  }

  html, body {
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
    background: #FFFFFF !important;
  }

  body { font-size: 9.7pt; line-height: 1.48; }

  .page { max-width: none; padding: 0; }
  main { gap: 26px !important; }
  p { margin-bottom: 11px; }

  /* The sticky sidebar is a screen affordance; on paper the contents
     become an ordinary opening block. */
  .body-grid { display: block !important; }
  .toc {
    position: static !important;
    break-after: page;
    margin-bottom: 0;
  }
  .toc ol { display: block; columns: 2; column-gap: 28px; }
  .toc li { margin-bottom: 7px; break-inside: avoid; }

  main { display: block !important; }

  .masthead { padding-top: 0; padding-bottom: 16px; margin-bottom: 26px; break-after: avoid; }
  .masthead h1 { font-size: 27pt; margin-bottom: 12px; }
  .standfirst { font-size: 12pt; margin-bottom: 18px; }

  /* Sections flow rather than each claiming a fresh page - forcing breaks
     added a dozen pages of whitespace. Only the two sections a reader is
     most often sent to directly get a clean start. */
  section#summary, section#ask { break-before: page; }
  section#summary { break-before: avoid; }

  .sect-head { break-after: avoid; padding-top: 10px; margin-bottom: 14px; }
  h3, h4 { break-after: avoid; }

  /* Blocks that read as a single unit must not split. Tables are the
     exception: the long ones are taller than a page, and forbidding the
     break just pushes them whole onto the next sheet and leaves half the
     previous one blank. They break by row instead, repeating their header. */
  .callout, .appcard, .ask, .layer, .figure,
  figure.diagram, blockquote.pull, .figures { break-inside: avoid; }
  .tablewrap, table { break-inside: auto; }
  tr, caption { break-inside: avoid; }
  caption { break-after: avoid; }
  thead { display: table-header-group; }

  /* Cards reflow to two columns; three is too narrow at A4 width. On screen
     the grid draws its own separators by showing a background through 1px
     gaps - on paper that trick paints any *empty* cell (a group with an odd
     number of cards, or one whose cards reflowed to the next page) as a solid
     block. Borders on the cards themselves have no such failure mode. */
  /* Flex with an explicit basis rather than grid: grid tracks size to the
     widest card's min-content and overflowed the right margin. */
  .appgrid {
    display: flex !important;
    flex-wrap: wrap;
    align-items: stretch;
    background: transparent !important;
    border: none !important;
    gap: 7px !important;
  }
  .appcard {
    flex: 0 0 calc(50% - 4px);
    max-width: calc(50% - 4px);
    border: 1px solid var(--rule);
  }
  .appcard.built { border-color: var(--accent); }
  /* Let a long group flow across a page rather than reserving a whole sheet
     for it; the heading stays welded to its first row of cards, and the cards
     themselves are individually unbreakable. */
  .appgroup { break-inside: auto; }
  .appgroup > h4 { break-after: avoid; }

  .figures { grid-template-columns: repeat(2, 1fr); }

  figure.diagram { margin: 18px 0 22px; }

  a { text-decoration: none; }

  footer.colophon { break-inside: avoid; }
</style>
"""

SHELL = """<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{fonts}
{body}
{print_css}
</body>
</html>
"""


def find_chrome() -> str:
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    sys.exit(
        "No Chrome or Chromium binary found. Looked for: "
        + ", ".join(CHROME_CANDIDATES)
    )


def build(src: str, out: str, keep_html: bool = False) -> None:
    with open(src, "r", encoding="utf-8") as f:
        fragment = f.read()

    # Drop the remote Google Fonts <link>: the faces are embedded as data URIs
    # instead, so the render never waits on (or silently loses) a network fetch.
    fragment = re.sub(
        r'<link[^>]+fonts\.(googleapis|gstatic)\.com[^>]*>\s*', "", fragment
    )

    doc = SHELL.format(
        fonts=inline_fonts(),
        body="<body>\n" + fragment,
        print_css=PRINT_CSS,
    )

    tmp_dir = tempfile.mkdtemp(prefix="pdfbuild-")
    tmp_html = os.path.join(tmp_dir, "print.html")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(doc)

    chrome = find_chrome()
    out = os.path.abspath(out)
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        # Google Fonts and the SVG must finish loading before the snapshot.
        "--virtual-time-budget=20000",
        f"--print-to-pdf={out}",
        f"file://{tmp_html}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out):
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        sys.exit(f"Chrome failed to produce {out}")

    size = os.path.getsize(out)
    print(f"Wrote {out} ({size / 1024:.0f} KB)")
    if keep_html:
        print(f"Print HTML kept at {tmp_html}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="artifact HTML fragment to render")
    ap.add_argument("-o", "--output", help="output PDF path (default: alongside source)")
    ap.add_argument("--keep-html", action="store_true",
                    help="keep the generated print HTML for debugging")
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.source)[0] + ".pdf"
    build(args.source, out, keep_html=args.keep_html)


if __name__ == "__main__":
    main()
