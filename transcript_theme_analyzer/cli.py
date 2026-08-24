"""CLI: run one theme against one or many .txt/.docx transcripts and one or
many models, producing exactly two output files: report.html and report.docx.

Usage (single transcript):
    python -m transcript_theme_analyzer.cli --transcript path.docx --theme "the glory of God" \\
        --models gpt-4.1 anthropic/claude-sonnet-5

Usage (whole folder of transcripts against the same theme):
    python -m transcript_theme_analyzer.cli --transcript-dir transcripts/ --theme "forgiveness" \\
        --models anthropic/claude-sonnet-5

Large batches resume automatically: completed transcripts are checkpointed as
they finish, so re-running the same command after an interruption picks up
where it left off instead of re-paying for finished work.
"""
from __future__ import annotations

import argparse
import asyncio
import fnmatch
import logging
import os
import sys
import time
from dataclasses import dataclass

from .analyzer import Progress, analyze, get_request_gate
from .cache import write_cache
from .checkpoint import CheckpointWriter, checkpoint_path, has_error, read_checkpoint
from .client import make_client
from .config import load_config
from .loader import SUPPORTED_EXTENSIONS, detect_format, load_transcript_text
from .report import write_html_from_data
from .word_report import write_docx_from_data

DEFAULT_GLOB_PATTERNS = ["*.txt", "*.docx"]

logger = logging.getLogger("transcript_theme_analyzer")


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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyze at most this many transcripts. Useful for a small trial run to "
        "check the theme and output before committing to a full corpus",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and re-analyze every transcript from scratch",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="On resume, leave previously-failed transcripts as failures instead of retrying them",
    )
    parser.add_argument(
        "--max-concurrent-transcripts",
        type=int,
        default=None,
        help="How many transcripts to analyze at once (default: MAX_CONCURRENT_TRANSCRIPTS or 4)",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Global ceiling on simultaneous API calls (default: MAX_CONCURRENT_REQUESTS or 12). "
        "Lower this if you see rate-limit warnings",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Progress verbosity. INFO (default) reports each chunk as it completes; "
        "WARNING reports only retries and failures",
    )
    return parser.parse_args(argv)


def _setup_logging(level: str) -> None:
    """Send progress to stdout, line-buffered.

    Both matter in a notebook: `!python -m ...` gives the process a pipe
    rather than a TTY, so stdout would otherwise be block-buffered and the
    cell would stay blank until the whole run finished. Logging to stdout
    (not stderr) also keeps progress lines in order with the plain prints
    below, which notebooks otherwise render as two separate streams.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # not a reconfigurable stream
        pass
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # httpx logs every request at INFO ("HTTP Request: POST ... 200 OK"), which
    # at one line per API call drowns out the progress lines this level exists
    # to show. Only raise them when the user explicitly asks for DEBUG.
    http_level = logging.NOTSET if getattr(logging, level) == logging.DEBUG else logging.WARNING
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(http_level)


def _discover_transcript_paths(transcript_dir: str, glob_pattern: str | None) -> list[str]:
    """Find transcripts anywhere under transcript_dir, including nested
    subfolders (e.g. a Google Drive folder with sub-folders of transcripts).

    Skips hidden files (starting with '.') and Word temporary lock files
    (starting with '~$'). Supported extensions are matched case-insensitively.
    """
    paths = []
    for root, _, files in os.walk(transcript_dir):
        for filename in files:
            if filename.startswith((".", "~$")):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if glob_pattern:
                if fnmatch.fnmatch(filename.lower(), glob_pattern.lower()):
                    paths.append(os.path.join(root, filename))
            else:
                if ext in SUPPORTED_EXTENSIONS:
                    paths.append(os.path.join(root, filename))
    return sorted(paths)


def _validate_formats_upfront(paths: list[str]) -> int:
    """Report unsupported/invalid input files before any (paid) analysis runs.

    This warns rather than aborts. Each bad file is recorded as a per-
    transcript error during the run and shows up in the report, so the
    failure is never silent -- while one unreadable file in a corpus of
    thousands cannot block the entire batch, or worse, block a *resume* of a
    batch that is already most of the way done.
    """
    errors = []
    for path in paths:
        try:
            detect_format(path)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        print(
            f"WARNING: {len(errors)} input file(s) are unsupported/invalid. They will be "
            "reported as errors and the rest of the batch will continue:",
            file=sys.stderr,
        )
        for err in errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
    return len(errors)


@dataclass
class WorkItem:
    """One transcript to analyze.

    ``key`` is the stable identity used for checkpointing and must be unique
    per file; ``display_name`` is what appears in the report and is made
    unique separately so two same-named files in different folders stay
    distinguishable to a reader.
    """

    path: str
    key: str
    display_name: str
    index: int


def _build_work_items(paths: list[str], base_dir: str | None) -> list[WorkItem]:
    """Assign each transcript a unique checkpoint key and display name.

    Basenames collide constantly in real corpora -- nested folders, and
    "Sermon (1).docx" style duplicates. Keying on the basename would make a
    resume skip a file that was never analyzed (its twin was), and would make
    two transcripts overwrite each other in the report. Keys therefore use the
    path relative to the input directory, which is unique by construction.
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    duplicated = {stem for stem in stems if stems.count(stem) > 1} if len(stems) < 5000 else set()
    if len(stems) >= 5000:
        # count() in a loop is quadratic; only worth the dict for big corpora
        seen_counts: dict[str, int] = {}
        for stem in stems:
            seen_counts[stem] = seen_counts.get(stem, 0) + 1
        duplicated = {stem for stem, count in seen_counts.items() if count > 1}

    items: list[WorkItem] = []
    used_names: set[str] = set()
    for index, (path, stem) in enumerate(zip(paths, stems)):
        if base_dir:
            try:
                key = os.path.relpath(path, base_dir)
            except ValueError:  # different drive on Windows
                key = path
        else:
            key = path
        # Only disambiguate names that actually collide, so the common case
        # keeps the clean filename a reader expects.
        base_name = os.path.splitext(key)[0] if stem in duplicated else stem
        name = base_name
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base_name} #{suffix}"
        used_names.add(name)
        items.append(WorkItem(path=path, key=key, display_name=name, index=index))
    return items


async def run_one_model(
    transcript: str, theme: str, model: str, config, client, label: str = "", gate=None
) -> dict:
    start = time.monotonic()
    progress = Progress(label=f"{label}{model}" if label else model)
    result, stats = await analyze(client, model, theme, transcript, config, progress, gate)
    elapsed = time.monotonic() - start
    progress.note(f"finished in {elapsed:.0f}s ({stats.total_tokens:,} tokens)")
    payload = result.model_dump()
    payload["_meta"] = {
        "elapsed_seconds": round(elapsed, 2),
        "usage": stats.summary(),
    }
    return payload


async def run_all(
    transcript: str,
    theme: str,
    models: list[str],
    label: str = "",
    config=None,
    client=None,
    gate=None,
) -> list[tuple[str, dict]]:
    """Run every model against one transcript. Returns a list of
    ``(model, raw_result_payload)`` pairs, in the same order as ``models`` —
    a failed run's payload is ``{"model": ..., "error": ...}``.
    """
    config = config or load_config()
    owns_client = client is None
    client = client or make_client(config)
    try:
        results = await asyncio.gather(
            *[
                run_one_model(transcript, theme, model, config, client, label, gate)
                for model in models
            ],
            return_exceptions=True,
        )
    finally:
        if owns_client:
            await client.close()

    raw_results: list[tuple[str, dict]] = []
    for model, result in zip(models, results):
        if isinstance(result, BaseException):
            # CancelledError/KeyboardInterrupt/SystemExit are the caller
            # shutting us down, not a model failure. They inherit from
            # BaseException rather than Exception precisely so they are not
            # swallowed by broad handlers -- recording one as a transcript
            # error would bake a Ctrl+C into the checkpoint as a permanent
            # failure, and resume would never retry it.
            if not isinstance(result, Exception):
                raise result
            logger.error("[%s%s] FAILED: %s", label, model, result)
            raw_results.append((model, {"model": model, "error": str(result)}))
            continue
        logger.info("[%s%s] score=%s", label, model, result["relevance_score"])
        raw_results.append((model, result))
    return raw_results


def _write_outputs(theme: str, raw_by_transcript: dict[str, list[tuple[str, dict]]], out_dir: str) -> None:
    if not raw_by_transcript:
        print("No results to report — nothing was analyzed.", file=sys.stderr)
        return

    cache_path = write_cache(theme, raw_by_transcript, out_dir)
    print(f"[cache] {cache_path} (internal — not a final output)")

    html_path = write_html_from_data(theme, raw_by_transcript, os.path.join(out_dir, "report.html"))
    print(f"[report:html] {html_path}")

    docx_path = write_docx_from_data(theme, raw_by_transcript, os.path.join(out_dir, "report.docx"))
    print(f"[report:docx] {docx_path}")


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


async def _analyze_item(
    item: WorkItem,
    theme: str,
    models: list[str],
    config,
    client,
    gate,
    semaphore: asyncio.Semaphore,
    total: int,
) -> tuple[WorkItem, list[tuple[str, dict]]]:
    """Analyze one transcript. Never raises for per-transcript problems —
    a read failure is recorded as an error payload so the rest of the batch
    continues and the failure is visible in the report."""
    async with semaphore:
        label = f"{item.index + 1}/{total} {item.display_name} | "
        try:
            # python-docx parsing is synchronous and can take a noticeable
            # moment on a large file. Off the event loop it would stall every
            # other transcript's in-flight API calls, not just this one.
            transcript = await asyncio.to_thread(load_transcript_text, item.path)
        except asyncio.CancelledError:
            raise
        except (ValueError, OSError) as exc:
            logger.error("[%s] FAILED to read transcript: %s", item.display_name, exc)
            return item, [
                (model, {"model": model, "error": f"could not read transcript: {exc}"})
                for model in models
            ]

        if not transcript.strip():
            logger.error("[%s] transcript is empty — skipping analysis", item.display_name)
            return item, [
                (model, {"model": model, "error": "transcript is empty"}) for model in models
            ]

        return item, await run_all(
            transcript, theme, models, label=label, config=config, client=client, gate=gate
        )


async def run_batch(
    transcript_paths: list[str],
    theme: str,
    models: list[str],
    out_dir: str,
    config=None,
    base_dir: str | None = None,
    resume: bool = True,
    retry_failed: bool = True,
) -> bool:
    """Analyze every transcript, checkpointing as each completes. Returns
    True if the run was interrupted before finishing."""
    config = config or load_config()
    os.makedirs(out_dir, exist_ok=True)

    items = _build_work_items(transcript_paths, base_dir)
    total = len(items)
    cp_path = checkpoint_path(out_dir)

    previous = read_checkpoint(cp_path, theme, models) if resume else {}
    results_by_key: dict[str, tuple[str, list[tuple[str, dict]]]] = {}
    pending: list[WorkItem] = []

    for item in items:
        prior = previous.get(item.key)
        if prior is None:
            pending.append(item)
            continue
        if retry_failed and has_error(prior["results"]):
            pending.append(item)
            continue
        results_by_key[item.key] = (item.display_name, prior["results"])

    reused = len(results_by_key)
    print(f"Analyzing {total} transcript(s) x {len(models)} model(s) for theme: {theme!r}")
    if reused:
        print(f"Resuming: {reused} already complete, {len(pending)} to go.")
    elif resume and os.path.isfile(cp_path):
        print("Checkpoint found but nothing reusable for this theme/model set — starting fresh.")
    if pending:
        print(
            f"Concurrency: {config.max_concurrent_transcripts} transcript(s) at a time, "
            f"{config.max_concurrent_requests} API call(s) in flight max.\n"
        )

    interrupted = False
    if pending:
        client = make_client(config)
        gate = get_request_gate(config.max_concurrent_requests)
        semaphore = asyncio.Semaphore(config.max_concurrent_transcripts)
        started = time.monotonic()
        completed = 0

        writer = CheckpointWriter(cp_path, theme, models)
        tasks = [
            asyncio.create_task(
                _analyze_item(item, theme, models, config, client, gate, semaphore, total)
            )
            for item in pending
        ]
        try:
            for finished in asyncio.as_completed(tasks):
                item, payloads = await finished
                # Record before anything else can fail: the whole point is
                # that this transcript never needs paying for twice.
                writer.record(item.key, item.display_name, payloads)
                results_by_key[item.key] = (item.display_name, payloads)

                completed += 1
                elapsed = time.monotonic() - started
                remaining = len(pending) - completed
                eta = (elapsed / completed) * remaining if completed else 0
                print(
                    f"--- {completed}/{len(pending)} done "
                    f"({reused + completed}/{total} overall) | "
                    f"elapsed {_format_duration(elapsed)} | "
                    f"eta {_format_duration(eta)} ---"
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            interrupted = True
            print(
                "\nInterrupted — cancelling in-flight work. "
                "Finished transcripts are saved; re-run the same command to resume.",
                file=sys.stderr,
            )
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            writer.close()
            await client.close()

    write_report_from_results(theme, items, results_by_key, out_dir, total)
    return interrupted


def write_report_from_results(
    theme: str,
    items: list[WorkItem],
    results_by_key: dict[str, tuple[str, list[tuple[str, dict]]]],
    out_dir: str,
    total: int,
) -> None:
    """Emit the report in discovery order rather than completion order, so
    output is deterministic no matter how the concurrency interleaved."""
    raw_by_transcript = {
        results_by_key[item.key][0]: results_by_key[item.key][1]
        for item in items
        if item.key in results_by_key
    }
    analyzed = len(raw_by_transcript)
    failed = sum(1 for payloads in raw_by_transcript.values() if has_error(payloads))
    print(f"\n{analyzed}/{total} transcript(s) in report" + (f", {failed} with errors" if failed else ""))
    _write_outputs(theme, raw_by_transcript, out_dir)


def salvage_report(
    transcript_paths: list[str],
    theme: str,
    models: list[str],
    out_dir: str,
    base_dir: str | None,
) -> None:
    """Rebuild the report purely from the checkpoint on disk.

    Used when an interrupt escaped the event loop, so no in-memory results
    survived. The checkpoint is the authoritative record of completed work
    precisely so this is always possible.
    """
    items = _build_work_items(transcript_paths, base_dir)
    stored = read_checkpoint(checkpoint_path(out_dir), theme, models)
    results_by_key = {
        key: (entry["display_name"], entry["results"]) for key, entry in stored.items()
    }
    if not results_by_key:
        return
    print("\nRebuilding the report from the checkpoint...")
    write_report_from_results(theme, items, results_by_key, out_dir, len(items))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    config = load_config()
    if args.max_concurrent_transcripts is not None:
        config.max_concurrent_transcripts = args.max_concurrent_transcripts
    if args.max_concurrent_requests is not None:
        config.max_concurrent_requests = args.max_concurrent_requests
    config = config.validated()
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
        # Apply --limit before validating: a trial run must not be held up
        # (or warned about) by files it is never going to touch.
        if args.limit is not None and args.limit > 0:
            if args.limit < len(paths):
                print(f"--limit {args.limit}: analyzing the first {args.limit} of {len(paths)} transcript(s).")
            paths = paths[: args.limit]
        _validate_formats_upfront(paths)
        try:
            interrupted = asyncio.run(
                run_batch(
                    paths,
                    args.theme,
                    models,
                    args.out_dir,
                    config=config,
                    base_dir=args.transcript_dir,
                    resume=not args.no_resume,
                    retry_failed=not args.skip_failed,
                )
            )
        except KeyboardInterrupt:
            # The interrupt landed somewhere the batch loop couldn't catch it,
            # so nothing in memory survived. Everything finished is still on
            # disk, so rebuild the report from there rather than leaving the
            # user with a completed run and no output.
            print("\nInterrupted.", file=sys.stderr)
            salvage_report(paths, args.theme, models, args.out_dir, args.transcript_dir)
            sys.exit(130)
        if interrupted:
            sys.exit(130)
        return

    transcript = load_transcript_text(args.transcript)
    transcript_stem = os.path.splitext(os.path.basename(args.transcript))[0]

    async def _single() -> list[tuple[str, dict]]:
        client = make_client(config)
        gate = get_request_gate(config.max_concurrent_requests)
        try:
            return await run_all(
                transcript, args.theme, models, config=config, client=client, gate=gate
            )
        finally:
            await client.close()

    raw_results = asyncio.run(_single())
    _write_outputs(args.theme, {transcript_stem: raw_results}, args.out_dir)


if __name__ == "__main__":
    main()
