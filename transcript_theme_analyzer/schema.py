"""Pydantic models for the analysis output schema."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    excerpt: str = Field(description="Short verbatim quote from the transcript, enough to locate it")
    context_summary: Optional[str] = Field(
        default=None, description="1-sentence description of what's happening here re: the theme"
    )
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
    explicitness: str = Field(
        description="How direct the theme's presence in this chunk is: 'explicit', 'tangential', or 'absent'"
    )
    reasoning: str
    locations: list[Location] = Field(default_factory=list)
