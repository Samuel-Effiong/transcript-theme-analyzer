"""Core analysis pipeline: single-pass fast path + chunked map-reduce path."""
from __future__ import annotations

import asyncio
import contextlib
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
from .extraction import extract_passage
from .schema import AnalysisResult, ChunkAnalysis, Location, LocationMarker

logger = logging.getLogger("transcript_theme_analyzer")


@dataclass
class Progress:
    """Emits a heartbeat as one transcript/model run works through its chunks.

    A long run is otherwise completely silent between the start of a
    transcript and its final score -- this reports each chunk as it lands so
    a stalled run is distinguishable from a slow one.
    """

    label: str
    total: int = 0
    done: int = 0

    def start(self, mode: str, total: int = 0) -> None:
        self.total = total
        self.done = 0
        if total:
            logger.info("[%s] %s: %d chunks", self.label, mode, total)
        else:
            logger.info("[%s] %s", self.label, mode)

    def step(self) -> None:
        self.done += 1
        logger.info("[%s] chunk %d/%d done", self.label, self.done, self.total)

    def note(self, message: str) -> None:
        logger.info("[%s] %s", self.label, message)


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


class RequestGate:
    """Caps total in-flight API calls and coordinates backoff across them.

    Two jobs, both about not tripping provider rate limits once transcripts
    and chunks run concurrently:

    * **A single global ceiling.** Transcript- and chunk-level concurrency
      multiply, so bounding either alone doesn't bound the request rate.
    * **Shared cooldown.** When one call is rate-limited, every other worker
      is about to be too. Backing off only the unlucky caller leaves the rest
      hammering a limit that is already tripped, which is what turns a brief
      429 into a sustained one. A 429 here pauses *all* callers.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self._pause_until = 0.0

    def pause(self, seconds: float) -> None:
        """Hold every caller back for `seconds`. Only ever extends an
        existing cooldown -- a shorter one must not cut a longer one short."""
        target = asyncio.get_running_loop().time() + seconds
        self._pause_until = max(self._pause_until, target)

    async def __aenter__(self) -> "RequestGate":
        await self._semaphore.acquire()
        try:
            # Re-check in a loop: another worker can extend the cooldown
            # while this one is sleeping through it.
            while True:
                remaining = self._pause_until - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return self
                await asyncio.sleep(remaining)
        except BaseException:
            # Cancelled mid-cooldown -- must not leak the slot.
            self._semaphore.release()
            raise

    async def __aexit__(self, *exc_info) -> None:
        self._semaphore.release()


_gate: RequestGate | None = None
_gate_loop: asyncio.AbstractEventLoop | None = None


def get_request_gate(limit: int) -> RequestGate:
    """One gate per event loop. Rebuilt if the loop or limit changes, since an
    asyncio primitive from a previous loop is unusable in the current one."""
    global _gate, _gate_loop
    loop = asyncio.get_running_loop()
    if _gate is None or _gate_loop is not loop or _gate.limit != limit:
        _gate = RequestGate(limit)
        _gate_loop = loop
    return _gate


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read the provider's own Retry-After hint. Honouring it beats guessing
    with exponential backoff -- the server is stating exactly how long the
    limit lasts."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-after")
    if raw is None:
        return None
    try:
        # Retry-After may also be an HTTP date; a date fails this parse and
        # falls through to normal exponential backoff, which is fine.
        return max(0.0, min(float(raw), 120.0))
    except (TypeError, ValueError):
        return None


# Models whose endpoint rejected `response_format={"type": "json_schema"}`.
# Anthropic's OpenAI-compatible endpoint is one of them, and without this the
# pipeline pays a guaranteed-400 round trip before the fallback on *every*
# call -- doubling request count and latency for the whole run.
_NO_STRUCTURED_OUTPUT: set[str] = set()


# Validation keywords that OpenAI-style strict structured output does not
# support. Sending them gets the whole request rejected, so they are stripped
# from the wire schema -- Pydantic still enforces them when validating the
# response, which is where they actually matter.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "format",
    "minItems", "maxItems", "uniqueItems", "default",
})


def _strictify(node):
    """Rewrite a Pydantic-generated JSON Schema into the strict subset.

    Strict mode has two requirements Pydantic's output violates by default:
    every object must set ``additionalProperties: false``, and every property
    must appear in ``required`` -- optionality is expressed by a nullable type
    union, not by omission from ``required``. Pydantic already emits
    ``Optional[str]`` as an ``anyOf`` with ``null``, so promoting every
    property to required is safe: the model can still answer ``null``.

    Without this the provider rejects the request, the caller silently falls
    back to prompt-based JSON, and the reliability benefit of native
    structured output is lost on every model that actually supports it.
    """
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {k: v for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYWORDS}
    for key, value in list(result.items()):
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            result[key] = {k: _strictify(v) for k, v in value.items()}
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            result[key] = _strictify(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [_strictify(v) for v in value]

    if result.get("type") == "object" or "properties" in result:
        properties = result.get("properties") or {}
        result["additionalProperties"] = False
        result["required"] = list(properties.keys())
    return result


def _json_schema_for(model_cls, name: str) -> dict:
    return {
        "name": name,
        "schema": _strictify(model_cls.model_json_schema()),
        "strict": True,
    }


VALID_EXPLICITNESS = {"explicit", "tangential", "absent"}


def _repair_parsed_json(parsed, schema_cls):
    """Best-effort repair of a parsed JSON object against schema_cls's
    required fields before validation.

    Models are unreliable about including every required field consistently
    even when instructed to (missing `reasoning`, a `relevance_score` sent as
    a fractional float, an `explicitness` outside the allowed enum, etc.) --
    retrying the identical call over and over just burns the retry budget on
    the same failure. This fills only safe, non-fabricating defaults (empty
    string/zero/empty list/"absent") for missing or invalid required fields,
    and coerces an out-of-range float score to the nearest valid int -- it
    never invents substantive content like reasoning or excerpts.
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

    # `explicitness` is a Literal enum, not a plain str/int/list, so the
    # generic loop above doesn't cover it -- handle missing AND out-of-enum
    # values here (covers both cases with one check).
    if "explicitness" in schema_cls.model_fields and parsed.get("explicitness") not in VALID_EXPLICITNESS:
        parsed["explicitness"] = "absent"

    # The 0-100 bound is enforced by Pydantic but deliberately not sent on the
    # wire (strict mode rejects minimum/maximum), so an out-of-range answer
    # reaches us intact. Clamping beats burning a retry on an off-by-a-little
    # score; a fractional score is rounded the same way.
    score = parsed.get("relevance_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        parsed["relevance_score"] = max(0, min(100, round(score)))

    locations = parsed.get("locations")
    if isinstance(locations, list):
        # start_marker/end_marker are the two things on LocationMarker that
        # can't be safely defaulted -- without them there's nothing to
        # anchor extraction on. Drop any entry missing either rather than
        # fail the whole response.
        parsed["locations"] = [
            loc for loc in locations
            if isinstance(loc, dict) and loc.get("start_marker") and loc.get("end_marker")
        ]

    return parsed


class TruncatedResponseError(Exception):
    """The model hit its output cap mid-JSON. Retrying the identical request
    usually truncates again, so this carries a pointed message rather than
    presenting as a generic parse failure."""


def _extract_content(resp, model: str) -> str:
    """Pull the JSON text out of a completion, failing loudly on the response
    shapes that are easy to mistake for a parse error.

    Aggregators like OpenRouter report some upstream failures as HTTP 200 with
    an error payload and no choices, so the SDK raises nothing. Reasoning
    models add a second trap: the answer can be empty while the token budget
    went to a separate reasoning field. Both look like "invalid JSON" if read
    naively, which sends the retry loop chasing the wrong problem.
    """
    error = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"Provider returned an error for {model}: {message}")

    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError(f"Provider returned no choices for {model} (empty response)")

    choice = choices[0]
    # finish_reason is optional in practice: some OpenAI-compatible providers
    # omit it entirely, so it must never be dereferenced directly.
    finish_reason = getattr(choice, "finish_reason", None)
    content = (getattr(getattr(choice, "message", None), "content", None) or "").strip()

    if not content:
        # A reasoning model that spent its whole budget thinking leaves the
        # answer empty; say so, because the fix is a bigger output cap or a
        # non-reasoning model, not another identical attempt.
        extra = getattr(getattr(choice, "message", None), "model_extra", None) or {}
        if extra.get("reasoning") or extra.get("reasoning_content"):
            raise TruncatedResponseError(
                f"{model} returned reasoning but no answer content -- raise "
                f"LLM_MAX_OUTPUT_TOKENS or use a non-reasoning model"
            )
        raise RuntimeError(f"{model} returned empty content (finish_reason={finish_reason!r})")

    if finish_reason == "length":
        raise TruncatedResponseError(
            f"{model} hit the output cap mid-response -- raise LLM_MAX_OUTPUT_TOKENS "
            f"(currently the JSON is cut off and cannot be parsed)"
        )

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    return content.strip()


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
    label: str = "",
    gate: "RequestGate | None" = None,
):
    """Call the chat completions API, enforcing the schema.

    Tries native structured output (response_format=json_schema) first; if
    the provider/model rejects that parameter, falls back to a plain call
    with a strict parse-and-retry loop. Once a model is known to reject it,
    later calls skip straight to the fallback rather than re-paying the
    failed round trip every time.
    """
    async def _plain_call():
        return await client.chat.completions.create(
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

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            # The gate is held across the structured-output attempt and its
            # fallback so the pair counts as one unit of concurrency.
            async with (gate if gate is not None else contextlib.nullcontext()):
                if model in _NO_STRUCTURED_OUTPUT:
                    resp = await _plain_call()
                else:
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
                        resp = await _plain_call()
                        # Only remember the model as unsupported once the fallback
                        # has actually succeeded: a BadRequestError from something
                        # else (oversized input, bad params) would otherwise
                        # permanently disable structured output for no reason.
                        if model not in _NO_STRUCTURED_OUTPUT:
                            _NO_STRUCTURED_OUTPUT.add(model)
                            logger.info(
                                "%s rejected structured output (response_format=json_schema); "
                                "using JSON-mode prompting for the rest of this run",
                                model,
                            )

            content = _extract_content(resp, model)
            parsed = json.loads(content)
            parsed = _repair_parsed_json(parsed, schema_cls)
            result = schema_cls.model_validate(parsed)
            return result, resp.usage
        except RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            # Exponential backoff, jittered so concurrent workers that were
            # rate-limited together don't retry in lockstep and re-trip the
            # limit. A server-supplied Retry-After overrides the guess.
            delay = min(2 ** attempt + random.uniform(0, 1), 30)
            hinted = _retry_after_seconds(exc)
            if hinted is not None:
                delay = max(delay, hinted)
            if isinstance(exc, openai.RateLimitError) and gate is not None:
                # Hold every other in-flight worker back too, not just this
                # one -- see RequestGate for why.
                gate.pause(delay)
            logger.warning(
                "[%s] Retryable error on attempt %d/%d: %s. Sleeping %.1fs",
                label, attempt + 1, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
        except openai.APIStatusError:
            # Non-retryable API error (bad request, auth, billing/quota, not
            # found, etc.) -- retrying identical input against these just
            # burns the whole budget for a guaranteed-identical failure.
            raise
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - parse/validation retry
            last_exc = exc
            logger.warning(
                "[%s] Parse/validation error on attempt %d/%d: %s",
                label, attempt + 1, max_retries, exc,
            )
            await asyncio.sleep(min(1.5 ** attempt, 10))

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}") from last_exc


async def analyze_single_pass(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    transcript: str,
    config: Config,
    stats: RunStats,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> AnalysisResult:
    if progress:
        progress.start("single-pass")
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
        label=progress.label if progress else "",
        gate=gate,
    )
    stats.add("single_pass", model, usage)
    if progress:
        progress.note("single-pass call complete")

    locations = []
    for marker in result.locations:
        excerpt = extract_passage(transcript, marker.start_marker, marker.end_marker)
        if excerpt is None:
            continue
        locations.append(
            Location(
                excerpt=excerpt,
                title=marker.title,
                timestamp=marker.timestamp,
                speaker=marker.speaker,
            )
        )

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
    progress: Progress | None = None,
    gate: RequestGate | None = None,
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
            label=progress.label if progress else "",
            gate=gate,
        )
        stats.add(f"chunk_{chunk.index}", model, usage)
        if progress:
            progress.step()
        return result, chunk


async def _synthesize_reasoning(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    final_score: int,
    chunk_results: list[tuple[ChunkAnalysis, object]],
    config: Config,
    stats: RunStats,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
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
        label=progress.label if progress else "",
        gate=gate,
    )
    stats.add("synthesis", model, usage)
    return result.reasoning


def _extract_and_fill_location(marker: LocationMarker, chunk) -> Location | None:
    """Extracts the full passage text for one chunk-relative marker, and
    fills in timestamp/speaker from the chunk's nearest match when the
    model didn't report one itself. Returns None if the marker's boundaries
    couldn't be located in the chunk's text (dropped, same policy as a
    missing excerpt)."""
    excerpt = extract_passage(chunk.text, marker.start_marker, marker.end_marker)
    if excerpt is None:
        return None
    return Location(
        excerpt=excerpt,
        title=marker.title,
        timestamp=marker.timestamp or chunk.nearest_timestamp,
        speaker=marker.speaker or chunk.nearest_speaker,
    )


async def analyze_map_reduce(
    client: AsyncOpenAI,
    model: str,
    theme: str,
    transcript: str,
    config: Config,
    stats: RunStats,
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> AnalysisResult:
    chunks = chunk_transcript(transcript, config.chunk_size_tokens, config.chunk_overlap_tokens)
    semaphore = asyncio.Semaphore(config.max_concurrent_chunks)
    if progress:
        progress.start(f"chunked (max {config.max_concurrent_chunks} concurrent)", len(chunks))

    tasks = [
        _analyze_one_chunk(
            client, model, theme, chunk, len(chunks), config, stats, semaphore, progress, gate
        )
        for chunk in chunks
    ]
    chunk_results = await asyncio.gather(*tasks)

    analyses = [r for r, _ in chunk_results]
    lengths = [c.char_end - c.char_start for _, c in chunk_results]
    final_score = compute_aggregate_score(analyses, lengths)

    all_locations: list[Location] = [
        loc
        for analysis, chunk in chunk_results
        for marker in analysis.locations
        if (loc := _extract_and_fill_location(marker, chunk)) is not None
    ]
    merged_locations = merge_and_dedupe_locations(all_locations)
    if progress:
        progress.note(
            f"all chunks done, score={final_score}, "
            f"{len(merged_locations)} passages — synthesizing reasoning"
        )

    reasoning = await _synthesize_reasoning(
        client, model, theme, final_score, chunk_results, config, stats, progress, gate
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
    progress: Progress | None = None,
    gate: RequestGate | None = None,
) -> tuple[AnalysisResult, RunStats]:
    """Entry point: picks the single-pass fast path or the chunked map-reduce path."""
    stats = RunStats()
    estimated_tokens = estimate_tokens(transcript)
    if estimated_tokens <= config.single_pass_token_limit:
        result = await analyze_single_pass(
            client, model, theme, transcript, config, stats, progress, gate
        )
    else:
        result = await analyze_map_reduce(
            client, model, theme, transcript, config, stats, progress, gate
        )
    return result, stats
