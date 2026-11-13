"""Review-queue queries (UI-agnostic so they are testable and reusable by the dashboard)."""
import json
from datetime import date, timedelta

from sqlalchemy import func, select

from cortex.agents.tracker import FUNNEL
from cortex.core.audit import AuditLog
from cortex.db.schema import JobRow, SkipRow, TailoredResumeRow


def review_queue(session, user_id: str, *, tracks=None, city: str = "", min_synapse: int = 0,
                 include_gated: bool = False, include_dups: bool = False, statuses=("to_apply",),
                 include_skipped: bool = False, limit: int = 500, max_age_days: int | None = None,
                 today: date | None = None) -> list[dict]:
    """max_age_days hides postings older than that (posted date, else date added). Most postings close within
    weeks and LinkedIn can't be checked automatically (no LinkedIn scraping), so age is the honest proxy.
    Hiding is view-only: nothing is marked closed. Jobs with neither date are kept."""
    q = select(JobRow).where(JobRow.user_id == user_id, JobRow.status.in_(statuses))
    if max_age_days is not None:
        cutoff = (today or date.today()) - timedelta(days=max_age_days)  # noqa: DTZ011 (local date)
        q = q.where(func.coalesce(JobRow.posted, JobRow.date_added).is_(None)
                    | (func.coalesce(JobRow.posted, JobRow.date_added) >= cutoff))
    if not include_skipped:
        q = q.where(JobRow.id.not_in(select(SkipRow.job_id).where(SkipRow.user_id == user_id)))
    if tracks:
        q = q.where(JobRow.track.in_(tracks))
    if city:
        q = q.where(JobRow.location.ilike(f"%{city}%"))
    if min_synapse:
        q = q.where(JobRow.synapse_score >= min_synapse)
    if not include_gated:
        q = q.where(JobRow.gate_reason.is_(None))
    if not include_dups:
        q = q.where(JobRow.dup_of_id.is_(None))
    q = q.order_by(JobRow.synapse_score.desc().nulls_last(), JobRow.score.desc()).limit(limit)
    from cortex.agents.batch_score import fits, latest_llm_scores
    rows = list(session.scalars(q))
    llm = latest_llm_scores(session, user_id, [r.id for r in rows])
    fit = fits(session, user_id, [r.id for r in rows])
    out = []
    for r in rows:
        o = llm.get(r.id)
        out.append(dict(id=r.id, company=r.company, role=r.title, track=r.track, posted=r.posted or r.date_added,
                        llm=o.score if o else None, llm_band=o.band if o else None, fit=fit.get(r.id),
                        source=r.source,
                        synapse=r.synapse_score, synapse_p=r.synapse_priority,
                        cortex=r.score, cortex_band=r.band, location=r.location,
                        exp=r.exp_raw, stack=r.stack_fit, url=r.url, notes=r.notes,
                        flagged=bool(r.sanitizer_flags), gate=r.gate_reason,
                        why=", ".join(json.loads(r.score_reasons or "[]"))))
    # LLM score first (Surya's choice); local skill-fit breaks ties and orders the not-yet-LLM-scored rest
    out.sort(key=lambda d: (d["llm"] is None, -(d["llm"] or 0), d["fit"] is None, -(d["fit"] or 0)))
    return out


def record_skip(session, user_id: str, job_id: int, audit: AuditLog | None = None) -> bool:
    """Record a skip decision. Returns False if the job isn't this user's or is already skipped."""
    owned = session.scalar(select(JobRow.id).where(JobRow.id == job_id, JobRow.user_id == user_id))
    if owned is None:
        return False
    if session.scalar(select(SkipRow.id).where(SkipRow.user_id == user_id, SkipRow.job_id == job_id)):
        return False
    session.add(SkipRow(user_id=user_id, job_id=job_id))
    session.commit()
    if audit:
        audit.log("review", "skip", user_id=user_id, job_id=job_id)
    return True


def skipped_job_ids(session, user_id: str) -> set[int]:
    return set(session.scalars(select(SkipRow.job_id).where(SkipRow.user_id == user_id)))


def pending_approvals(session, user_id: str) -> list[dict]:
    """Tailored resumes that passed Truth Guard and wait for a human decision."""
    q = (select(TailoredResumeRow, JobRow)
         .join(JobRow, (JobRow.id == TailoredResumeRow.job_id) & (JobRow.user_id == TailoredResumeRow.user_id))
         .where(TailoredResumeRow.user_id == user_id, TailoredResumeRow.status == "pending_approval")
         .order_by(TailoredResumeRow.created_at))
    return [dict(job_id=j.id, company=j.company, role=j.title, url=j.url, attempts=t.attempts,
                 created_at=t.created_at, resume=json.loads(t.resume_json or "null"))
            for t, j in session.execute(q)]


def ready_to_apply(session, user_id: str) -> list[dict]:
    """Approved in the gate (packet built), not yet marked applied."""
    from cortex.db.schema import PacketRow
    rows = session.scalars(select(JobRow).where(JobRow.user_id == user_id, JobRow.status == "approved")
                           .order_by(JobRow.updated_at.desc()))
    out = []
    for j in rows:
        pk = session.scalars(select(PacketRow.path).where(PacketRow.user_id == user_id, PacketRow.job_id == j.id)
                             .order_by(PacketRow.id.desc())).first()
        out.append(dict(job_id=j.id, company=j.company, role=j.title, url=j.url, packet=pk))
    return out


def applications(session, user_id: str) -> list[dict]:
    """Everything in the funnel (applied and beyond), newest application first."""
    q = (select(JobRow).where(JobRow.user_id == user_id, JobRow.dup_of_id.is_(None),
                              JobRow.status.in_(FUNNEL))
         .order_by(JobRow.applied_date.desc().nulls_last()))
    return [dict(job_id=j.id, company=j.company, role=j.title, status=j.status, applied=j.applied_date, url=j.url)
            for j in session.scalars(q)]
