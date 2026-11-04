"""Tracker agent: the last pipeline step. Records what Surya did with each job and what happened next.

It only records decisions he reports (applied / interview / offer / rejected); it never infers a status,
never contacts anyone, never submits anything.

- Forward-only transitions (TRANSITIONS). A tracker re-import can't downgrade either (STATUS_RANK).
- Every change is an append-only ApplicationEventRow (who/when/what/note) plus an audit-log entry, so each
  job has a full history. jobs.status/applied_date stay the current view the rest of CORTEX reads, which
  means every update is also a new eval label.
- Follow-ups: applied/interview jobs with no event for N days (config tracker.followup_days).
"""
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select

from cortex.core.audit import AuditLog
from cortex.db.schema import ApplicationEventRow, JobRow

TRANSITIONS = {
    "to_apply": {"applied"},
    "approved": {"applied"},          # packet built in CORTEX, now applied by hand
    "pending_approval": {"applied"},  # applied without waiting for the packet: allowed, it's his call
    "applied": {"interview", "rejected"},
    "interview": {"interview", "offer", "rejected"},   # interview -> interview = another round
    "offer": set(),
    "rejected": set(),
}
ACTIVE = ("applied", "interview")


class TrackerError(ValueError):
    pass


def _job(session, user_id: str, job_id: int) -> JobRow:
    job = session.scalar(select(JobRow).where(JobRow.id == job_id, JobRow.user_id == user_id))
    if job is None:
        raise TrackerError("job not found for this user")
    return job


def update_status(session, user_id: str, job_id: int, new_status: str, *, note: str | None = None,
                  on: date | None = None, audit: AuditLog | None = None, now: datetime | None = None) -> JobRow:
    job = _job(session, user_id, job_id)
    allowed = TRANSITIONS.get(job.status, set())
    if new_status not in allowed:
        raise TrackerError(f"can't go from {job.status!r} to {new_status!r}"
                           + (f" (allowed: {', '.join(sorted(allowed))})" if allowed else " (final status)"))
    if note is not None and len(note) > 300:
        raise TrackerError("note too long (max 300 chars)")
    at = now or datetime.now(UTC)
    if new_status == "applied":
        job.applied_date = on or at.date()
    job.status = new_status
    session.add(ApplicationEventRow(user_id=user_id, job_id=job_id, status=new_status, note=note, at=at))
    session.commit()
    if audit:
        audit.log("tracker", "status", user_id=user_id, job_id=job_id, status=new_status)
    return job


def mark_applied(session, user_id: str, job_id: int, **kw) -> JobRow:
    return update_status(session, user_id, job_id, "applied", **kw)


def history(session, user_id: str, job_id: int) -> list[ApplicationEventRow]:
    return list(session.scalars(select(ApplicationEventRow).where(
        ApplicationEventRow.user_id == user_id, ApplicationEventRow.job_id == job_id).order_by(ApplicationEventRow.at)))


def _last_activity(session, user_id: str, job: JobRow) -> date | None:
    ev = session.scalar(select(ApplicationEventRow.at).where(
        ApplicationEventRow.user_id == user_id, ApplicationEventRow.job_id == job.id)
        .order_by(ApplicationEventRow.at.desc()))
    return ev.date() if ev else job.applied_date   # imported tracker rows have a date but no events


def followups(session, user_id: str, today: date, days: int = 7) -> list[dict]:
    """Active applications with no update for `days` or more, oldest first."""
    out = []
    for job in session.scalars(select(JobRow).where(JobRow.user_id == user_id, JobRow.status.in_(ACTIVE),
                                                    JobRow.dup_of_id.is_(None))):
        last = _last_activity(session, user_id, job)
        if last is not None and (today - last).days >= days:
            out.append({"job_id": job.id, "company": job.company, "role": job.title, "status": job.status,
                        "last_update": last, "days_quiet": (today - last).days, "url": job.url})
    return sorted(out, key=lambda r: -r["days_quiet"])


@dataclass
class TrackerReport:
    by_status: dict[str, int]
    applied_total: int          # everything that ever reached "applied"
    responses: int              # moved past applied: interview or offer
    response_rate: float | None
    per_week: list[tuple[str, int]]   # (ISO week starting Monday, applications) for the last N weeks


def report(session, user_id: str, today: date, weeks: int = 6) -> TrackerReport:
    jobs = list(session.scalars(select(JobRow).where(JobRow.user_id == user_id, JobRow.dup_of_id.is_(None))))
    funnel = ("applied", "interview", "offer", "rejected")
    by_status = Counter(j.status for j in jobs if j.status in funnel)
    reached = [j for j in jobs if j.status in funnel]
    interviewed = {e.job_id for e in session.scalars(select(ApplicationEventRow).where(
        ApplicationEventRow.user_id == user_id, ApplicationEventRow.status.in_(("interview", "offer"))))}
    responses = sum(1 for j in reached if j.status in ("interview", "offer") or j.id in interviewed)
    start = today - timedelta(days=today.weekday()) - timedelta(weeks=weeks - 1)
    weeks_list = [start + timedelta(weeks=i) for i in range(weeks)]
    per_week = [(w.isoformat(), sum(1 for j in reached if j.applied_date and w <= j.applied_date < w + timedelta(days=7)))
                for w in weeks_list]
    return TrackerReport(dict(by_status), len(reached), responses,
                         responses / len(reached) if reached else None, per_week)


def format_report(r: TrackerReport, fups: list[dict]) -> str:
    lines = [f"applications: {r.applied_total}  " + "  ".join(f"{k}={v}" for k, v in sorted(r.by_status.items())),
             "response rate (interview or offer / applied): "
             + (f"{r.response_rate:.0%} ({r.responses}/{r.applied_total})" if r.response_rate is not None else "n/a"),
             "applications per week: " + "  ".join(f"{w[5:]}:{n}" for w, n in r.per_week)]
    lines.append(f"follow-ups due: {len(fups)}")
    lines += [f"  {f['days_quiet']:>3}d quiet  [{f['job_id']}] {f['company']} | {f['role']} ({f['status']})" for f in fups[:15]]
    return "\n".join(lines)
