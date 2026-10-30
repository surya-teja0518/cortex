"""Deterministic gates. Run before any LLM call: cheap, auditable, zero cost."""
from .models import Job


def title_blocked(job: Job, blocklist: list[str]) -> str | None:
    t = job.title.lower()
    for w in blocklist:
        if w in t:
            return f"title contains '{w}'"
    return None


def yoe_blocked(job: Job, max_min_yoe: int) -> str | None:
    if job.min_yoe is not None and job.min_yoe > max_min_yoe:
        return f"min YOE {job.min_yoe} > {max_min_yoe}"
    return None


def run_gates(job: Job, cfg: dict) -> str | None:
    g = cfg["gates"]
    return title_blocked(job, g["title_blocklist"]) or yoe_blocked(job, g["max_min_yoe"])
