"""Pydantic models. Every LLM output must validate against one of these."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

Track = Literal["data", "database", "ai"]
Band = Literal["P0", "P1", "P2", "P3"]
Status = Literal["new", "gated_out", "scored", "tailored", "pending_approval",
                 "approved", "rejected", "applied", "interview", "offer", "closed"]


class Job(BaseModel):
    source: str
    external_id: str
    title: str
    company: str
    location: str = ""
    url: str
    description: str = ""
    min_yoe: Optional[int] = None
    max_yoe: Optional[int] = None
    applicants: Optional[int] = None
    posted_at: Optional[str] = None


class ScoreResult(BaseModel):
    track: Track
    score: int = Field(ge=0, le=100)
    band: Band
    reasons: list[str] = Field(max_length=6)
    red_flags: list[str] = Field(default_factory=list)


class TailoredResume(BaseModel):
    summary: str
    bullets: list[str]
    skills_emphasized: list[str]
