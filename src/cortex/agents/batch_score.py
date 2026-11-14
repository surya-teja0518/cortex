"""Score new jobs with the LLM scorer and keep the result, so the digest and dashboard can rank by it.

Results go into llm_score_cache (the same store eval uses, keyed by job + provider + prompt hash), so a job
scored here is never paid for again in eval and vice versa. jobs.score/band stay the DETERMINISTIC score:
eval compares the two, so overwriting the rules' score would destroy the baseline.

Which jobs: to_apply, not gated, not a duplicate, not injection-flagged (flagged jobs never reach an LLM),
and no cached LLM result yet. Best local skill-fit first (compute_fits, no LLM), then newest; capped per run
(Groq's free tier allows ~500 calls/day).
"""
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select

from cortex.agents.scorer import LLMScorer, ScorerOutcome
from cortex.db.schema import JobRow, LLMScoreCacheRow, SkillFitRow
from cortex.eval import llm_score_cached

MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class BatchResult:
    candidates: int = 0
    scored: int = 0          # new LLM scores
    unscored: int = 0        # fell back to the deterministic score (provider down, bad output)
    stopped_early: str | None = None


def _scored_ids(session, user_id: str) -> set[int]:
    return set(session.scalars(select(LLMScoreCacheRow.job_id).where(LLMScoreCacheRow.user_id == user_id)))


def _live(user_id: str):
    return select(JobRow).where(JobRow.user_id == user_id, JobRow.status == "to_apply", JobRow.gate_reason.is_(None),
                                JobRow.dup_of_id.is_(None), JobRow.sanitizer_flags.is_(None))


def compute_fits(session, user_id: str, matcher) -> int:
    """Local skill-fit for every live job whose text is new or changed. Returns how many were (re)computed."""
    have = {r.job_id: r for r in session.scalars(select(SkillFitRow).where(SkillFitRow.user_id == user_id))}
    n = 0
    for j in session.scalars(_live(user_id)):
        text = f"{j.title}\n{j.description or ''}"
        sha = hashlib.sha256(text.encode()).hexdigest()
        row = have.get(j.id)
        if row is not None and row.text_sha == sha:
            continue
        f = matcher.fit(text)
        if row is None:
            row = SkillFitRow(user_id=user_id, job_id=j.id)
            session.add(row)
        row.score, row.matched, row.missing, row.text_sha = f.score, json.dumps(f.matched), json.dumps(f.missing), sha
        n += 1
    session.commit()
    return n


def fits(session, user_id: str, job_ids: list[int] | None = None) -> dict[int, int | None]:
    q = select(SkillFitRow.job_id, SkillFitRow.score).where(SkillFitRow.user_id == user_id)
    if job_ids is not None:
        q = q.where(SkillFitRow.job_id.in_(job_ids))
    return dict(session.execute(q).all())


def candidates(session, user_id: str, limit: int) -> list[JobRow]:
    """Not yet LLM-scored, best local skill-fit first (no fit -> after), newest first within equal fit. So the
    free LLM quota goes to the jobs most likely to suit him."""
    done = _scored_ids(session, user_id)
    rows = [r for r in session.scalars(_live(user_id)) if r.id not in done]
    fit = fits(session, user_id, [r.id for r in rows])
    rows.sort(key=lambda r: (fit.get(r.id) is None, -(fit.get(r.id) or 0), -r.id))
    return rows[:limit]


def score_new(session, user_id: str, scorer: LLMScorer, limit: int = 100,
              progress: Callable[[int, int], None] | None = None) -> BatchResult:
    rows = candidates(session, user_id, limit)
    res = BatchResult(candidates=len(rows))
    fails = 0
    for i, r in enumerate(rows, 1):
        out, _ = llm_score_cached(session, scorer, r, user_id)
        if out.source == "llm":
            res.scored += 1
            fails = 0
        else:
            res.unscored += 1
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILURES:
                res.stopped_early = (f"{fails} jobs in a row fell back to rules: provider down or out of quota; "
                                     "the rest stay on their deterministic score until the next run")
                break
        if progress:
            progress(i, len(rows))
    return res


def latest_llm_scores(session, user_id: str, job_ids: list[int] | None = None) -> dict[int, ScorerOutcome]:
    """Newest cached LLM result per job (any provider / prompt version)."""
    q = select(LLMScoreCacheRow).where(LLMScoreCacheRow.user_id == user_id).order_by(LLMScoreCacheRow.id)
    if job_ids is not None:
        q = q.where(LLMScoreCacheRow.job_id.in_(job_ids))
    return {row.job_id: ScorerOutcome.model_validate_json(row.result_json) for row in session.scalars(q)}
