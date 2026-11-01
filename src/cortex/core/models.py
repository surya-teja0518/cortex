"""Pydantic models. Every LLM output must validate against one of these."""
from typing import Literal

from pydantic import BaseModel, Field

Track = Literal["data", "database", "ai"]
Band = Literal["P0", "P1", "P2", "P3"]
# Normalized lifecycle statuses (tracker import maps raw sheet values onto these)
Status = Literal["to_apply", "applied", "interview", "offer", "rejected",
                 "not_accepting", "exp_mismatch", "closed",
                 "new", "gated_out", "scored", "tailored", "pending_approval", "approved"]


class Job(BaseModel):
    source: str
    external_id: str
    title: str
    company: str
    location: str = ""
    url: str
    description: str = ""
    min_yoe: int | None = None
    max_yoe: int | None = None
    applicants: int | None = None
    posted_at: str | None = None


class ScoreResult(BaseModel):
    track: Track
    score: int = Field(ge=0, le=100)
    band: Band
    reasons: list[str] = Field(max_length=6)
    red_flags: list[str] = Field(default_factory=list)

