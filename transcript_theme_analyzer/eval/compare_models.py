"""Run the same (transcript, theme) pair across multiple models and print a
side-by-side comparison table (score, cost proxy via token usage, latency).

Usage:
    python -m transcript_theme_analyzer.eval.compare_models --transcript path.txt \\
        --theme "forgiveness" --models gpt-4.1 anthropic/claude-sonnet-5 google/gemini-2.5-pro
"""
from __future__ import annotations

import argparse
import asyncio

from ..cli import run_one_model
from ..config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple models on one (transcript, theme) pair")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--theme", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    return parser.parse_args(argv)


async def compare(transcript: str, theme: str, models: list[str]) -> list[dict]:
    config = load_config()
    results = await asyncio.gather(
        *[run_one_model(transcript, theme, model, config) for model in models],
        return_exceptions=True,
    )
    rows = []
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            rows.append({"model": model, "error": str(result)})
        else:
            rows.append(
                {
                    "model": model,
                    "score": result["relevance_score"],
                    "locations": len(result["locations"]),
                    "chunked": result["chunked"],
                    "tokens": result["_meta"]["usage"]["total_tokens"],
                    "seconds": result["_meta"]["elapsed_seconds"],
                    "reasoning": result["reasoning"],
                }
            )
    return rows


def print_table(rows: list[dict]) -> None:
    header = f"{'model':<35} {'score':>5} {'locs':>5} {'chunked':>7} {'tokens':>8} {'sec':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['model']:<35} ERROR: {row['error']}")
            continue
        print(
            f"{row['model']:<35} {row['score']:>5} {row['locations']:>5} "
            f"{str(row['chunked']):>7} {row['tokens']:>8} {row['seconds']:>6.1f}"
        )
    print()
    for row in rows:
        if "error" not in row:
            print(f"--- {row['model']} reasoning ---")
            print(row["reasoning"])
            print()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with open(args.transcript, "r", encoding="utf-8") as f:
        transcript = f.read()
    rows = asyncio.run(compare(transcript, args.theme, args.models))
    print_table(rows)


if __name__ == "__main__":
    main()
