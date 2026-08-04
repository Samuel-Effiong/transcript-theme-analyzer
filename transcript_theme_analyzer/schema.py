"""Pydantic models for the analysis output schema."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LocationMarker(BaseModel):
    """What the model reports for one relevant passage -- boundary markers
    only, not the passage text itself. The pipeline extracts the full text
    deterministically (see extraction.extract_passage) from these, which is
    far cheaper in completion tokens and more reliable than asking the
    model to reproduce a potentially long passage verbatim."""

    title: Optional[str] = Field(
        default=None, description="Short section-title label (2-6 words, title case) for this passage"
    )
    start_marker: str = Field(description="A short (5-12 word) verbatim phrase marking where this passage begins")
    end_marker: str = Field(description="A short (5-12 word) verbatim phrase marking where this passage ends")
    timestamp: Optional[str] = None
    speaker: Optional[str] = None


class Location(BaseModel):
    """A fully-extracted passage, ready for display. `excerpt` is filled in
    by the pipeline from a LocationMarker's boundaries -- the model never
    produces it directly."""

    excerpt: str = Field(description="The full extracted passage text")
    title: Optional[str] = Field(default=None, description="Short section-title label for this passage")
    timestamp: Optional[str] = None
    speaker: Optional[str] = None


class AnalysisResult(BaseModel):
    """Final, top-level result returned by the pipeline for one (transcript, theme, model) run."""

    theme: str
    relevance_score: int = Field(ge=0, le=100)
    reasoning: str
    locations: list[Location] = Field(default_factory=list)
    model_used: str
    chunked: bool


class ChunkAnalysis(BaseModel):
    """Partial result produced by the map step, for a single chunk."""

    relevance_score: int = Field(ge=0, le=100)
    explicitness: Literal["explicit", "tangential", "absent"] = Field(
        description="How direct the theme's presence in this chunk is"
    )
    reasoning: str
    locations: list[LocationMarker] = Field(default_factory=list)
