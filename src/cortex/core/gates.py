"""Deterministic gates. Run before any LLM call: cheap, auditable, zero cost.

Title matching uses WORD BOUNDARIES: 'lead' blocks 'Lead Data Engineer' but not
'Leadership Analyst'; 'vp' blocks 'VP Data' but never matches inside other words.
"""
import re

from .models import Job


def _pattern(word: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(word.lower())}(?![a-z0-9])")


def title_blocked(job: Job, blocklist: list[str]) -> str | None:
    t = job.title.lower()
    for w in blocklist:
        if _pattern(w).search(t):
            return f"title contains '{w}'"
    return None


def yoe_blocked(job: Job, max_min_yoe: int) -> str | None:
    if job.min_yoe is not None and job.min_yoe > max_min_yoe:
        return f"min YOE {job.min_yoe} > {max_min_yoe}"
    return None


def run_gates(job: Job, cfg: dict) -> str | None:
    g = cfg["gates"]
    return title_blocked(job, g["title_blocklist"]) or yoe_blocked(job, g["max_min_yoe"])
