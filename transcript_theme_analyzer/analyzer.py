"""Core analysis pipeline: single-pass fast path + chunked map-reduce path."""
from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import get_origin

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from . import prompts
from .aggregate import compute_aggregate_score, merge_and_dedupe_locations
from .chunker import chunk_transcript, estimate_tokens
from .config import Config
from .schema import AnalysisResult, ChunkAnalysis, Location

logger = logging.getLogger("transcript_theme_analyzer")


@dataclass
class CallLog:
    label: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RunStats:
    calls: list[CallLog] = field(default_factory=list)

    def add(self, label: str, model: str, usage) -> None:
        if usage is None:
            self.calls.append(CallLog(label=label, model=model))
            return
        self.calls.append(
            CallLog(
                label=label,
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        )

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    def summary(self) -> dict:
        return {
            "calls": len(self.calls),
            "total_tokens": self.total_tokens,
            "by_call": [c.__dict__ for c in self.calls],
        }


RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.APITimeoutError,
)


def _json_schema_for(model_cls, name: str) -> dict:
    schema = model_cls.model_json_schema()
    schema["additionalProperties"] = False
    return {"name": name, "schema": schema, "strict": True}


def _repair_parsed_json(parsed, schema_cls):
    """Best-effort repair of a parsed JSON object against schema_cls's
    required fields before validation.

    Models are unreliable about including every required field consistently
    even when instructed to (missing `reasoning`, a `relevance_score` sent as
    a fractional float, etc.) -- retrying the identical call over and over
    just burns the retry budget on the same failure. This fills only safe,
    non-fabricating defaults (empty string/zero/empty list) for missing
    required fields, and coerces an out-of-range float score to the nearest
    valid int -- it never invents substantive content like reasoning or
    excerpts.
    """
    if not isinstance(parsed, dict):
        return parsed
    parsed = dict(parsed)

    for name, field in schema_cls.model_fields.items():
        if not field.is_required() or name in parsed:
            continue
        if field.annotation is str:
            parsed[name] = ""
        elif field.annotation is int:
            parsed[name] = 0
        elif get_origin(field.annotation) is list:
            parsed[name] = []

    score = parsed.get("relevance_score")
    if isinstance(score, float):
        parsed["relevance_score"] = max(0, min(100, round(score)))

    locations = parsed.get("locations")
    if isinstance(locations, list):
        # `excerpt` is the one thing on Location that can't be safely
        # defaulted (there's no non-fabricating placeholder for a quote) --
        # drop any entry missing it rather than fail the whole response.
        parsed["locations"] = [
            loc for loc in locations if isinstance(loc, dict) and loc.get("excerpt")
        ]

    return parsed


async def _call_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    system: str,
    user: str,
    schema_cls,
    schema_name: str,
    max_retries: int,
    max_output_tokens: int,
):
    """Call the chat completions API, enforcing the schema.

    Tries native structured output (response_format=json_schema) first; if
    the provider/model rejects that parameter, falls back to a plain call
    with a strict parse-and-retry loop.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max_output_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": _json_schema_for(schema_cls, schema_name),
                    },
                )
            except openai.BadRequestError:
                # Provider doesn't support structured output mode -- fall back
                # to plain JSON-mode prompting and a strict parse retry below.
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max_output_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": user
                            + "\n\nRespond with ONLY a single valid JSON object matching "
                            "the required schema. No markdown fences, no commentary.",
                        },
                    ],
                )

            content = resp.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            parsed = _repair_parsed_json(parsed, schema_cls)
            result = schema_cls.model_validate(parsed)
            return result, resp.usage
        except RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            delay = min(2 ** attempt + random.uniform(0, 1), 30)
            logger.warning("Retryable error on attempt %d/%d: %s. Sleeping %.1fs", attempt + 1, max_retries, exc, delay)
            await asyncio.sleep(delay)
        except openai.APIStatusError:
            # Non-retryable API error (bad request, auth, billing/quota, not
            # found, etc.) -- retrying identical input against these just
            # burns the whole budget for a guaranteed-identical failure.
            raise
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - parse/validation retry
            last_exc = exc
            logger.warning("Parse/validation error on attempt %d/%d: %s", attempt + 1, max_retries, exc)
            await asyncio.sleep(min(1.5 ** attempt, 10))

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}") from last_exc


async def analyze_single_pass(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    transcript: str,
    config: Config,
    stats: RunStats,
) -> AnalysisResult:
    user = prompts.SINGLE_PASS_USER_PROMPT_TEMPLATE.format(theme=theme, transcript=transcript)
    result, usage = await _call_with_retry(
        client,
        model=model,
        system=prompts.CHUNK_ANALYSIS_SYSTEM_PROMPT_V1,
        user=user,
        schema_cls=ChunkAnalysis,
        schema_name="chunk_analysis",
        max_retries=config.max_retries,
        max_output_tokens=config.max_output_tokens,
    )
    stats.add("single_pass", model, usage)

    locations = [
        Location(
            excerpt=loc.excerpt,
            context_summary=loc.context_summary,
            timestamp=loc.timestamp,
            speaker=loc.speaker,
        )
        for loc in result.locations
    ]

    return AnalysisResult(
        theme=theme,
        relevance_score=result.relevance_score,
        reasoning=result.reasoning,
        locations=locations,
        model_used=model,
        chunked=False,
    )


async def _analyze_one_chunk(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    chunk,
    total_chunks: int,
    config: Config,
    stats: RunStats,
    semaphore: asyncio.Semaphore,
) -> tuple[ChunkAnalysis, "object"]:
    async with semaphore:
        user = prompts.CHUNK_USER_PROMPT_TEMPLATE.format(
            theme=theme,
            chunk_index=chunk.index + 1,
            total_chunks=total_chunks,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            chunk_text=chunk.text,
        )
        result, usage = await _call_with_retry(
            client,
            model=model,
            system=prompts.CHUNK_ANALYSIS_SYSTEM_PROMPT_V1,
            user=user,
            schema_cls=ChunkAnalysis,
            schema_name="chunk_analysis",
            max_retries=config.max_retries,
            max_output_tokens=config.max_output_tokens,
        )
        stats.add(f"chunk_{chunk.index}", model, usage)
        return result, chunk


async def _synthesize_reasoning(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    final_score: int,
    chunk_results: list[tuple[ChunkAnalysis, object]],
    config: Config,
    stats: RunStats,
) -> str:
    lines = []
    for result, chunk in sorted(chunk_results, key=lambda pair: pair[1].index):
        lines.append(
            f"- Segment {chunk.index + 1} (chars {chunk.char_start}-{chunk.char_end}, "
            f"score {result.relevance_score}, {result.explicitness}): {result.reasoning}"
        )
    user = (
        f"THEME:\n{theme}\n\n"
        f"OVERALL COMPUTED SCORE: {final_score}\n\n"
        f"PER-SEGMENT PARTIAL ANALYSES (in transcript order):\n" + "\n".join(lines)
    )

    class SynthesisOutput(BaseModel):
        reasoning: str

    result, usage = await _call_with_retry(
        client,
        model=model,
        system=prompts.SYNTHESIS_SYSTEM_PROMPT_V1,
        user=user,
        schema_cls=SynthesisOutput,
        schema_name="synthesis_output",
        max_retries=config.max_retries,
        max_output_tokens=config.max_output_tokens,
    )
    stats.add("synthesis", model, usage)
    return result.reasoning


def _fill_location_defaults(loc: Location, chunk) -> Location:
    """Fills in a location's timestamp/speaker from the chunk's nearest match
    when the model didn't report one itself."""
    return Location(
        excerpt=loc.excerpt,
        context_summary=loc.context_summary,
        timestamp=loc.timestamp or chunk.nearest_timestamp,
        speaker=loc.speaker or chunk.nearest_speaker,
    )


async def analyze_map_reduce(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    transcript: str,
    config: Config,
    stats: RunStats,
) -> AnalysisResult:
    chunks = chunk_transcript(transcript, config.chunk_size_tokens, config.chunk_overlap_tokens)
    semaphore = asyncio.Semaphore(config.max_concurrent_chunks)

    tasks = [
        _analyze_one_chunk(client, model, theme, chunk, len(chunks), config, stats, semaphore)
        for chunk in chunks
    ]
    chunk_results = await asyncio.gather(*tasks)

    analyses = [r for r, _ in chunk_results]
    lengths = [c.char_end - c.char_start for _, c in chunk_results]
    final_score = compute_aggregate_score(analyses, lengths)

    all_locations: list[Location] = [
        _fill_location_defaults(loc, chunk)
        for analysis, chunk in chunk_results
        for loc in analysis.locations
    ]
    merged_locations = merge_and_dedupe_locations(all_locations)

    reasoning = await _synthesize_reasoning(
        client, model, theme, final_score, chunk_results, config, stats
    )

    return AnalysisResult(
        theme=theme,
        relevance_score=final_score,
        reasoning=reasoning,
        locations=merged_locations,
        model_used=model,
        chunked=True,
    )


async def analyze(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    transcript: str,
    config: Config,
) -> tuple[AnalysisResult, RunStats]:
    """Entry point: picks the single-pass fast path or the chunked map-reduce path."""
    stats = RunStats()
    estimated_tokens = estimate_tokens(transcript)
    if estimated_tokens <= config.single_pass_token_limit:
        result = await analyze_single_pass(client, model, theme, transcript, config, stats)
    else:
        result = await analyze_map_reduce(client, model, theme, transcript, config, stats)
    return result, stats
