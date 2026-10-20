"""Run the same transcript and theme list across multiple models and print a
side-by-side comparison: one row per (model, theme) with the score and the
number of passages, then the model's total tokens and cost for the run.

Usage:
    python -m transcript_theme_analyzer.eval.compare_models --transcript path.txt \\
        --theme "forgiveness" --theme "prayer" \\
        --models gpt-4.1 anthropic/claude-sonnet-5 google/gemini-2.5-pro

Themes can also come from a CSV: --themes-file themes.csv. The same
MAX_THEMES limit as the main CLI applies (--max-themes overrides it).
"""
from __future__ import annotations

import argparse
import asyncio

from ..analyzer import get_request_gate
from ..cli import _load_themes, run_one_model
from ..client import make_client
from ..config import load_config
from ..loader import load_transcript_text
from ..pricing import resolve_prices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple models on one transcript and theme list")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--theme", action="append", default=[], help="Repeat for several themes")
    parser.add_argument("--themes-file", default=None, help="CSV with a 'theme' column and optional 'description'")
    parser.add_argument("--max-themes", type=int, default=None, help="Override MAX_THEMES")
    parser.add_argument("--models", nargs="+", required=True)
    return parser.parse_args(argv)


async def compare(transcript: str, themes, models: list[str], config) -> list[dict]:
    prices = resolve_prices(models, config)
    client = make_client(config)
    gate = get_request_gate(config.max_concurrent_requests)
    try:
        results = await asyncio.gather(
            *[
                run_one_model(transcript, themes, model, config, client, gate=gate, prices=prices.get(model))
                for model in models
            ],
            return_exceptions=True,
        )
    finally:
        await client.close()
    rows = []
    for model, result in zip(models, results):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result
            rows.append({"model": model, "error": str(result)})
            continue
        usage = result["_meta"]["usage"]
        rows.append(
            {
                "model": model,
                "themes": [
                    {
                        "name": entry["theme"],
                        "score": entry["relevance_score"],
                        "locations": len(entry["locations"]),
                    }
                    for entry in result["themes"].values()
                ],
                "tokens": usage["total_tokens"],
                "cost": usage["cost"],
                "seconds": result["_meta"]["elapsed_seconds"],
            }
        )
    return rows


def _cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.4f}"


def print_table(rows: list[dict]) -> None:
    header = f"{'model':<35} {'theme':<30} {'score':>5} {'locs':>5}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['model']:<35} ERROR: {row['error']}")
            continue
        for entry in row["themes"]:
            print(f"{row['model']:<35} {entry['name'][:30]:<30} {entry['score']:>5} {entry['locations']:>5}")
    print()
    for row in rows:
        if "error" not in row:
            print(
                f"{row['model']}: {row['tokens']:,} tokens, cost {_cost(row['cost'])}, "
                f"{row['seconds']:.1f}s"
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config()
    if args.max_themes is not None:
        config.max_themes = args.max_themes
    config = config.validated()
    themes = _load_themes(args, config)  # exits 2 on bad input, before any API call
    transcript = load_transcript_text(args.transcript)
    print_table(asyncio.run(compare(transcript, themes, args.models, config)))


if __name__ == "__main__":
    main()
