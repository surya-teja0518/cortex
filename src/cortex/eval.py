"""Offline eval: deterministic vs LLM scorer against Surya's real decisions.

Labels:
  positives = jobs he applied to / interviewed / got an offer for
  negatives = jobs he looked at and turned down: tracker status exp_mismatch, plus Skip decisions
              from the review UI. (not_accepting / closed are the employer's call, not his: unlabelled.)
Positive-only metrics: recall@P0-P1, mean percentile rank. With negatives: precision@P0-P1 (of the
labelled jobs a scorer bands P0/P1, the share he applied to) and pos-vs-neg AUC (chance a random
positive outscores a random negative; 0.5 = coin flip). Below MIN_NEGATIVES they are not reported.
"""
import hashlib
import json
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select

from cortex.agents.scorer import LLMScorer, ScorerOutcome, ScoringInput, render_job
from cortex.core.models import ScoreResult
from cortex.db.schema import JobRow, LLMScoreCacheRow, SkipRow

# A positive is a job Surya CHOSE to apply to, whatever the employer did next: the scorer predicts his
# choices, so rejected / no_response applications are still positives.
POSITIVE_STATUSES = ("applied", "interview", "offer", "rejected", "no_response")
NEGATIVE_STATUSES = ("exp_mismatch", "skills_mismatch")   # jobs he looked at and turned down
TOP_BANDS = ("P0", "P1")
MIN_NEGATIVES = 10
BOOTSTRAP_RESAMPLES = 2000
MAX_CONSECUTIVE_FAILURES = 10   # live LLM jobs in a row that fell back -> the provider is broken: stop


class EvalAborted(RuntimeError):
    """Provider keeps failing. Everything scored so far is cached; rerunning continues from there."""


@dataclass
class ScorerMetrics:
    name: str
    recall_top: float | None      # share of positives banded P0/P1
    mean_pct_rank: float | None   # mean percentile of positives' scores within the population (higher = better)
    unscored: int = 0             # LLM column only: jobs that fell back to the deterministic score
    precision_top: float | None = None   # of labelled jobs banded P0/P1, share that are positives
    precision_n: int = 0                 # labelled jobs banded P0/P1 (precision denominator)
    auc: float | None = None             # P(random positive scores above random negative), ties = 0.5
    top_share: float | None = None       # share of ALL jobs this scorer bands P0/P1
    lift: float | None = None            # recall_top / top_share (1.0 = no better than random banding)
    ranks: dict[int, float] = field(default_factory=dict, repr=False)   # job id -> pct rank (positives)


@dataclass
class PairedTest:
    """LLM minus rules, on the same applied jobs: mean pct-rank difference with a bootstrap CI."""
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    p_le_zero: float      # share of resamples where the LLM did not beat the rules (one-sided)


@dataclass
class EvalReport:
    population: int
    positives: int
    positives_gated: int
    skips: int
    negatives: int = 0
    negatives_from_status: int = 0
    negatives_gated: int = 0
    metrics: list[ScorerMetrics] = field(default_factory=list)
    llm_calls_cached: int = 0
    llm_jobs_scored_live: int = 0
    paired: PairedTest | None = None
    exclude_fields: tuple[str, ...] = ()


def pct_rank(score: int, all_scores: list[int]) -> float:
    """Mid-rank percentile: ties count half, so equal scores don't flatter or punish."""
    below = sum(1 for s in all_scores if s < score)
    equal = sum(1 for s in all_scores if s == score)
    return 100.0 * (below + 0.5 * equal) / len(all_scores)


def pos_neg_auc(pos_scores: list[int], neg_scores: list[int]) -> float:
    """Mann-Whitney AUC: fraction of (positive, negative) pairs ordered correctly; ties count half."""
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos_scores for n in neg_scores)
    return wins / (len(pos_scores) * len(neg_scores))


def metrics_for(name: str, scored: dict[int, tuple[int, str]], positive_ids: set[int],
                unscored: int = 0, negative_ids: set[int] = frozenset()) -> ScorerMetrics:
    pos = [scored[i] for i in positive_ids if i in scored]
    if not pos:
        return ScorerMetrics(name, None, None, unscored)
    all_scores = [s for s, _ in scored.values()]
    recall = sum(1 for _, b in pos if b in TOP_BANDS) / len(pos)
    ranks = {i: pct_rank(scored[i][0], all_scores) for i in positive_ids if i in scored}
    mpr = sum(ranks.values()) / len(ranks)
    m = ScorerMetrics(name, recall, mpr, unscored, ranks=ranks)
    m.top_share = sum(1 for _, b in scored.values() if b in TOP_BANDS) / len(scored)
    m.lift = recall / m.top_share if m.top_share else None
    neg = [scored[i] for i in negative_ids if i in scored]
    if len(neg) >= MIN_NEGATIVES:
        top_pos = sum(1 for _, b in pos if b in TOP_BANDS)
        top_neg = sum(1 for _, b in neg if b in TOP_BANDS)
        m.precision_n = top_pos + top_neg
        m.precision_top = top_pos / m.precision_n if m.precision_n else None
        m.auc = pos_neg_auc([s for s, _ in pos], [s for s, _ in neg])
    return m


def paired_bootstrap(diffs: list[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> PairedTest:
    """Resample the applied jobs (not the two scorers separately), so each job is compared with itself."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(resamples))
    return PairedTest(n=n, mean_diff=statistics.fmean(diffs), ci_low=means[int(0.025 * resamples)],
                      ci_high=means[int(0.975 * resamples) - 1], p_le_zero=sum(m <= 0 for m in means) / resamples)


def deterministic_result(r: JobRow) -> ScoreResult:
    return ScoreResult(track=r.track or "data", score=r.score, band=r.band,
                       reasons=json.loads(r.score_reasons or "[]")[:6])


def llm_score_cached(session, scorer: LLMScorer, r: JobRow, user_id: str) -> tuple[ScorerOutcome, bool]:
    """Return (outcome, cache_hit). Only successful LLM scores are cached; fallbacks retry next run."""
    inp = ScoringInput.from_row(r)
    prepared = render_job(inp, scorer.exclude_fields)   # excluded fields change the prompt, hence the cache key
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


Progress = Callable[[int, int, int, int, int, float], None]   # done, total, live, cached, unscored, seconds


def run_eval(session, user_id: str, scorer: LLMScorer | None = None, progress: Progress | None = None,
             max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES, clock=time.monotonic) -> EvalReport:
    rows = session.scalars(select(JobRow).where(
        JobRow.user_id == user_id, JobRow.dup_of_id.is_(None), JobRow.score.is_not(None))).all()
    gated = [r for r in rows if r.gate_reason]
    pop = [r for r in rows if not r.gate_reason]
    positive_ids = {r.id for r in pop if r.status in POSITIVE_STATUSES}
    skipped = set(session.scalars(select(SkipRow.job_id).where(SkipRow.user_id == user_id)))
    status_neg = {r.id for r in pop if r.status in NEGATIVE_STATUSES}
    negative_ids = (status_neg | (skipped & {r.id for r in pop})) - positive_ids   # applying wins over skip
    rep = EvalReport(population=len(pop), positives=len(positive_ids),
                     positives_gated=sum(1 for r in gated if r.status in POSITIVE_STATUSES), skips=len(skipped),
                     negatives=len(negative_ids), negatives_from_status=len(status_neg - positive_ids),
                     negatives_gated=sum(1 for r in gated if r.status in NEGATIVE_STATUSES or r.id in skipped))

    rep.metrics.append(metrics_for("deterministic", {r.id: (r.score, r.band) for r in pop}, positive_ids,
                                   negative_ids=negative_ids))
    if scorer is not None:
        llm, unscored, streak, t0 = {}, 0, 0, clock()
        for done, r in enumerate(pop, start=1):
            out, hit = llm_score_cached(session, scorer, r, user_id)
            rep.llm_calls_cached += hit
            rep.llm_jobs_scored_live += (not hit and out.source == "llm")
            unscored += out.source != "llm"
            llm[r.id] = (out.score, out.band)
            # A live job that fell back for a provider reason (not an injection flag) counts toward the streak.
            failed_live = not hit and out.source != "llm" and "injection_flagged" not in out.flags
            streak = streak + 1 if failed_live else 0 if not hit else streak
            if streak >= max_consecutive_failures:
                last = getattr(scorer.provider, "last_error", None)
                hint = (" This looks like a daily quota: rerun after it resets."
                        if last and "429" in last and "retry-after" in last else "")
                raise EvalAborted(f"{streak} jobs in a row could not be scored by {scorer.provider.name}; stopping. "
                                  + (f"Last provider error: {last}.{hint} " if last else "")
                                  + f"{rep.llm_calls_cached + rep.llm_jobs_scored_live} results are cached, so a "
                                  "rerun continues from here.")
            if progress:
                progress(done, len(pop), rep.llm_jobs_scored_live, rep.llm_calls_cached, unscored, clock() - t0)
        excl = "".join(f" -{f}" for f in sorted(scorer.exclude_fields))
        rep.exclude_fields = tuple(sorted(scorer.exclude_fields))
        rep.metrics.append(metrics_for(f"llm:{scorer.provider.name}{excl}", llm, positive_ids, unscored,
                                       negative_ids=negative_ids))
        det, lm = rep.metrics
        common = sorted(det.ranks.keys() & lm.ranks.keys())
        if len(common) >= 2:
            rep.paired = paired_bootstrap([lm.ranks[i] - det.ranks[i] for i in common])
    return rep


def format_report(rep: EvalReport) -> str:
    def f(v, fmt):
        return "n/a" if v is None else format(v, fmt)
    has_neg = rep.negatives >= MIN_NEGATIVES
    lines = [(f"population={rep.population} (ungated, deduped)  positives={rep.positives} "
              f"(status in {'/'.join(POSITIVE_STATUSES)})  negatives={rep.negatives} "
              f"(status {'/'.join(NEGATIVE_STATUSES)}={rep.negatives_from_status}, skips={rep.skips})"),
             f"{'scorer':<26}{'recall@P0-P1':>14}{'top share':>11}{'lift':>7}{'mean pct rank':>15}"
             + (f"{'prec@P0-P1':>16}{'pos>neg AUC':>13}" if has_neg else "") + f"{'llm_unscored':>14}"]
    for m in rep.metrics:
        prec = f"{f(m.precision_top, '.0%')} (n={m.precision_n})"
        lines.append(f"{m.name:<26}{f(m.recall_top, '.0%'):>14}{f(m.top_share, '.0%'):>11}"
                     f"{f(m.lift, '.2f') + ('x' if m.lift is not None else ''):>7}{f(m.mean_pct_rank, '.1f'):>15}"
                     + (f"{prec:>16}{f(m.auc, '.2f'):>13}" if has_neg else "")
                     + f"{(m.unscored if m.name != 'deterministic' else '-'):>14}")
    if rep.paired:
        p = rep.paired
        lines.append(f"paired (llm - deterministic) mean pct rank on the same {p.n} applied jobs: {p.mean_diff:+.1f} "
                     f"[95% CI {p.ci_low:+.1f}, {p.ci_high:+.1f}], P(diff <= 0) = {p.p_le_zero:.3f} "
                     f"(bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed 0)")
    if rep.exclude_fields:
        lines.append(f"ablation: fields excluded from the LLM prompt: {', '.join(rep.exclude_fields)}")
    if any(m.name.startswith("llm") for m in rep.metrics):
        lines.append(f"llm cache hits={rep.llm_calls_cached}  live-scored={rep.llm_jobs_scored_live}  "
                     "(llm_unscored jobs use the deterministic score)")
    if rep.positives_gated:
        lines.append(f"note: {rep.positives_gated} positive(s) were blocked by deterministic gates "
                     "and are excluded above; for the full pipeline they are misses.")
    if rep.negatives_gated:
        lines.append(f"note: {rep.negatives_gated} negative(s) were already blocked by deterministic gates "
                     "(correctly) and are excluded above.")
    if has_neg:
        lines.append(f"ROUGH: precision/AUC rest on {rep.negatives} negatives, mostly exp_mismatch, i.e. turned down "
                     "for experience; the deterministic scorer uses min YOE directly, so it has a built-in edge "
                     "on these. Skips in the review UI give broader negatives.")
    else:
        lines.append(f"PRECISION NOT REPORTED: needs >= {MIN_NEGATIVES} negatives, have {rep.negatives} "
                     f"(exp_mismatch + skips={rep.skips}). Use Skip in the review UI to build them.")
    return "\n".join(lines)
