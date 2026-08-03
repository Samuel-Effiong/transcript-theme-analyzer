"""CLI: run one theme against one or many .txt/.docx transcripts and one or
many models, producing exactly two output files: report.html and report.docx.

Usage (single transcript):
    python -m transcript_theme_analyzer.cli --transcript path.docx --theme "the glory of God" \\
        --models gpt-4.1 anthropic/claude-sonnet-5

Usage (whole folder of transcripts against the same theme):
    python -m transcript_theme_analyzer.cli --transcript-dir transcripts/ --theme "forgiveness" \\
        --models anthropic/claude-sonnet-5
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import sys
import time

from .analyzer import analyze
from .cache import write_cache
from .client import make_client
from .config import load_config
from .loader import detect_format, load_transcript_text
from .report import write_html_from_data
from .word_report import write_docx_from_data

DEFAULT_GLOB_PATTERNS = ["*.txt", "*.docx"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcript theme-relevance analyzer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transcript", help="Path to a single transcript file (.txt or .docx)")
    source.add_argument(
        "--transcript-dir",
        help="Directory of transcript files (.txt and/or .docx) to run the same theme against",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Filename pattern to match inside --transcript-dir (default: *.txt and *.docx)",
    )
    parser.add_argument("--theme", required=True, help="Theme to analyze for (word, phrase, or paragraph)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more model identifiers to run (defaults to config default_model)",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory to write the report (report.html, report.docx) into",
    )
    return parser.parse_args(argv)


def _discover_transcript_paths(transcript_dir: str, glob_pattern: str | None) -> list[str]:
    patterns = [glob_pattern] if glob_pattern else DEFAULT_GLOB_PATTERNS
    paths = {p for pattern in patterns for p in glob.glob(os.path.join(transcript_dir, pattern))}
    return sorted(paths)


def _validate_formats_upfront(paths: list[str]) -> None:
    """Check every path's format before running any (paid) analysis, so a
    bad/unsupported file is caught immediately rather than after other
    transcripts in the same batch have already been analyzed."""
    errors = []
    for path in paths:
        try:
            detect_format(path)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        print("Cannot proceed — some input files are unsupported/invalid:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


async def run_one_model(transcript: str, theme: str, model: str, config) -> dict:
    client = make_client(config)
    start = time.monotonic()
    result, stats = await analyze(client, model, theme, transcript, config)
    elapsed = time.monotonic() - start
    payload = result.model_dump()
    payload["_meta"] = {
        "elapsed_seconds": round(elapsed, 2),
        "usage": stats.summary(),
    }
    return payload


async def run_all(transcript: str, theme: str, models: list[str]) -> list[tuple[str, dict]]:
    """Run every model against one transcript. Returns a list of
    ``(model, raw_result_payload)`` pairs, in the same order as ``models`` —
    a failed run's payload is ``{"model": ..., "error": ...}``.
    """
    config = load_config()

    results = await asyncio.gather(
        *[run_one_model(transcript, theme, model, config) for model in models],
        return_exceptions=True,
    )

    raw_results: list[tuple[str, dict]] = []
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            print(f"[{model}] FAILED: {result}", file=sys.stderr)
            raw_results.append((model, {"model": model, "error": str(result)}))
            continue
        print(f"[{model}] score={result['relevance_score']}")
        raw_results.append((model, result))
    return raw_results


def _write_outputs(theme: str, raw_by_transcript: dict[str, list[tuple[str, dict]]], out_dir: str) -> None:
    cache_path = write_cache(theme, raw_by_transcript, out_dir)
    print(f"[cache] {cache_path} (internal — not a final output)")

    html_path = write_html_from_data(theme, raw_by_transcript, os.path.join(out_dir, "report.html"))
    print(f"[report:html] {html_path}")

    docx_path = write_docx_from_data(theme, raw_by_transcript, os.path.join(out_dir, "report.docx"))
    print(f"[report:docx] {docx_path}")


async def run_batch(
    transcript_paths: list[str], theme: str, models: list[str], out_dir: str
) -> None:
    raw_by_transcript: dict[str, list[tuple[str, dict]]] = {}
    for path in transcript_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== {stem} ===")
        try:
            transcript = load_transcript_text(path)
        except (ValueError, OSError) as exc:
            print(f"[{stem}] FAILED to read transcript: {exc}", file=sys.stderr)
            raw_by_transcript[stem] = [
                (model, {"model": model, "error": f"could not read transcript: {exc}"})
                for model in models
            ]
            continue
        raw_by_transcript[stem] = await run_all(transcript, theme, models)

    print()
    _write_outputs(theme, raw_by_transcript, out_dir)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config()
    models = args.models or [config.default_model]

    if args.transcript_dir:
        paths = _discover_transcript_paths(args.transcript_dir, args.glob)
        if not paths:
            pattern_desc = args.glob or " or ".join(DEFAULT_GLOB_PATTERNS)
            print(
                f"No files matching {pattern_desc!r} found in {args.transcript_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        _validate_formats_upfront(paths)
        asyncio.run(run_batch(paths, args.theme, models, args.out_dir))
        return

    transcript = load_transcript_text(args.transcript)
    transcript_stem = os.path.splitext(os.path.basename(args.transcript))[0]
    raw_results = asyncio.run(run_all(transcript, args.theme, models))
    _write_outputs(args.theme, {transcript_stem: raw_results}, args.out_dir)


if __name__ == "__main__":
    main()
