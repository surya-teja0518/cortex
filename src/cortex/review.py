"""Review-queue queries (UI-agnostic so they are testable and reusable by the dashboard)."""
import json

from sqlalchemy import select

from cortex.core.audit import AuditLog
from cortex.db.schema import JobRow, SkipRow


def review_queue(session, user_id: str, *, tracks=None, city: str = "", min_synapse: int = 0,
                 include_gated: bool = False, include_dups: bool = False, statuses=("to_apply",),
                 include_skipped: bool = False, limit: int = 500) -> list[dict]:
    q = select(JobRow).where(JobRow.user_id == user_id, JobRow.status.in_(statuses))
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
    out = []
    for r in session.scalars(q):
        out.append(dict(id=r.id, company=r.company, role=r.title, track=r.track,
                        synapse=r.synapse_score, synapse_p=r.synapse_priority,
                        cortex=r.score, cortex_band=r.band, location=r.location,
                        exp=r.exp_raw, stack=r.stack_fit, url=r.url, notes=r.notes,
                        flagged=bool(r.sanitizer_flags), gate=r.gate_reason,
                        why=", ".join(json.loads(r.score_reasons or "[]"))))
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
