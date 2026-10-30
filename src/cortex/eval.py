"""Offline eval: deterministic vs LLM scorer against Surya's real decisions.

Labels today are positives only (jobs he applied to / interviewed / got an offer for).
Without negatives we can measure how well each scorer surfaces the jobs he chose (recall@P0-P1,
mean percentile rank), but NOT precision. Skip decisions from the review UI are the negatives;
precision gets reported once they exist in useful numbers.
"""
import hashlib
import json
from dataclasses import dataclass, field

from sqlalchemy import select

from cortex.agents.scorer import LLMScorer, ScorerOutcome, ScoringInput, render_job
from cortex.core.models import ScoreResult
from cortex.db.schema import JobRow, LLMScoreCacheRow, SkipRow

POSITIVE_STATUSES = ("applied", "interview", "offer")
TOP_BANDS = ("P0", "P1")


@dataclass
class ScorerMetrics:
    name: str
    recall_top: float | None      # share of positives banded P0/P1
    mean_pct_rank: float | None   # mean percentile of positives' scores within the population (higher = better)
    unscored: int = 0             # LLM column only: jobs that fell back to the deterministic score


@dataclass
class EvalReport:
    population: int
    positives: int
    positives_gated: int
    skips: int
    metrics: list[ScorerMetrics] = field(default_factory=list)
    llm_calls_cached: int = 0
    llm_jobs_scored_live: int = 0


def pct_rank(score: int, all_scores: list[int]) -> float:
    """Mid-rank percentile: ties count half, so equal scores don't flatter or punish."""
    below = sum(1 for s in all_scores if s < score)
    equal = sum(1 for s in all_scores if s == score)
    return 100.0 * (below + 0.5 * equal) / len(all_scores)


def metrics_for(name: str, scored: dict[int, tuple[int, str]], positive_ids: set[int],
                unscored: int = 0) -> ScorerMetrics:
    pos = [scored[i] for i in positive_ids if i in scored]
    if not pos:
        return ScorerMetrics(name, None, None, unscored)
    all_scores = [s for s, _ in scored.values()]
    recall = sum(1 for _, b in pos if b in TOP_BANDS) / len(pos)
    mpr = sum(pct_rank(s, all_scores) for s, _ in pos) / len(pos)
    return ScorerMetrics(name, recall, mpr, unscored)


def deterministic_result(r: JobRow) -> ScoreResult:
    return ScoreResult(track=r.track or "data", score=r.score, band=r.band,
                       reasons=json.loads(r.score_reasons or "[]")[:6])


def llm_score_cached(session, scorer: LLMScorer, r: JobRow, user_id: str) -> tuple[ScorerOutcome, bool]:
    """Return (outcome, cache_hit). Only successful LLM scores are cached; fallbacks retry next run."""
    inp = ScoringInput.from_row(r)
    prepared = render_job(inp)
    sha = hashlib.sha256(prepared[0].encode()).hexdigest()
    key = (LLMScoreCacheRow.user_id == user_id, LLMScoreCacheRow.job_id == r.id,
           LLMScoreCacheRow.provider == scorer.provider.name, LLMScoreCacheRow.prompt_sha256 == sha)
    hit = session.scalar(select(LLMScoreCacheRow).where(*key))
    if hit is not None:
        return ScorerOutcome.model_validate_json(hit.result_json), True
    out = scorer.score(inp, deterministic_result(r), user_id=user_id, prepared=prepared)
    if out.source == "llm":
        session.add(LLMScoreCacheRow(user_id=user_id, job_id=r.id, provider=scorer.provider.name,
                                     prompt_sha256=sha, result_json=out.model_dump_json()))
        session.commit()
    return out, False


def run_eval(session, user_id: str, scorer: LLMScorer | None = None) -> EvalReport:
    rows = session.scalars(select(JobRow).where(
        JobRow.user_id == user_id, JobRow.dup_of_id.is_(None), JobRow.score.is_not(None))).all()
    gated = [r for r in rows if r.gate_reason]
    pop = [r for r in rows if not r.gate_reason]
    positive_ids = {r.id for r in pop if r.status in POSITIVE_STATUSES}
    skips = len(session.scalars(select(SkipRow.id).where(SkipRow.user_id == user_id)).all())
    rep = EvalReport(population=len(pop), positives=len(positive_ids),
                     positives_gated=sum(1 for r in gated if r.status in POSITIVE_STATUSES), skips=skips)

    rep.metrics.append(metrics_for("deterministic", {r.id: (r.score, r.band) for r in pop}, positive_ids))
    if scorer is not None:
        llm, unscored = {}, 0
        for r in pop:
            out, hit = llm_score_cached(session, scorer, r, user_id)
            rep.llm_calls_cached += hit
            rep.llm_jobs_scored_live += (not hit and out.source == "llm")
            unscored += out.source != "llm"
            llm[r.id] = (out.score, out.band)
        rep.metrics.append(metrics_for(f"llm:{scorer.provider.name}", llm, positive_ids, unscored))
    return rep


def format_report(rep: EvalReport) -> str:
    def f(v, fmt):
        return "n/a" if v is None else format(v, fmt)
    lines = [(f"population={rep.population} (ungated, deduped)  positives={rep.positives} "
              f"(status in {'/'.join(POSITIVE_STATUSES)})"),
             f"{'scorer':<20}{'recall@P0-P1':>14}{'mean pct rank':>15}{'llm_unscored':>14}"]
    for m in rep.metrics:
        lines.append(f"{m.name:<20}{f(m.recall_top, '.0%'):>14}{f(m.mean_pct_rank, '.1f'):>15}"
                     f"{(m.unscored if m.name != 'deterministic' else '-'):>14}")
    if any(m.name.startswith("llm") for m in rep.metrics):
        lines.append(f"llm cache hits={rep.llm_calls_cached}  live-scored={rep.llm_jobs_scored_live}  "
                     "(llm_unscored jobs use the deterministic score)")
    if rep.positives_gated:
        lines.append(f"note: {rep.positives_gated} positive(s) were blocked by deterministic gates "
                     "and are excluded above; for the full pipeline they are misses.")
    lines.append(f"PRECISION NOT REPORTED: positives-only labels. Precision needs negatives "
                 f"(skip decisions recorded so far: {rep.skips}). Use Skip in the review UI to build them.")
    return "\n".join(lines)
