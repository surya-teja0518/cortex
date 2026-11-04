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
    "applied": {"interview", "rejected", "no_response"},
    "interview": {"interview", "offer", "rejected", "no_response"},   # interview -> interview = another round
    "offer": set(),
    "rejected": set(),
    "no_response": set(),            # written off by Surya; final
}
ACTIVE = ("applied", "interview")
FUNNEL = ("applied", "interview", "offer", "rejected", "no_response")
UNDO = "undo"   # event status for a correction; note starts with "reverted #<event id>"


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


def _voided(session, user_id: str) -> set[int]:
    """Event ids that a later undo reverted (events stay append-only; undos point at what they revert)."""
    out = set()
    for note in session.scalars(select(ApplicationEventRow.note).where(
            ApplicationEventRow.user_id == user_id, ApplicationEventRow.status == UNDO)):
        if note and note.startswith("reverted #"):
            out.add(int(note[len("reverted #"):].split(":", 1)[0]))
    return out


def undo(session, user_id: str, job_id: int, *, note: str | None = None, audit: AuditLog | None = None,
         now: datetime | None = None) -> JobRow:
    """Revert the last status change CORTEX recorded for this job (e.g. 'applied' entered by mistake).
    Appends an 'undo' event instead of deleting anything, so the history shows the correction."""
    job = _job(session, user_id, job_id)
    voided = _voided(session, user_id)
    live = [e for e in history(session, user_id, job_id) if e.status != UNDO and e.id not in voided]
    if not live:
        raise TrackerError("nothing to undo: CORTEX has no recorded status change for this job "
                           "(statuses imported from the tracker sheet aren't undoable here)")
    target = live[-1]
    if job.status != target.status:
        raise TrackerError(f"job is {job.status!r} but the last recorded change is {target.status!r}; not undoing")
    if len(live) > 1:
        prev = live[-2].status
    else:   # first change: go back to where it came from
        from cortex.db.schema import TailoredResumeRow
        approved = session.scalar(select(TailoredResumeRow.id).where(
            TailoredResumeRow.user_id == user_id, TailoredResumeRow.job_id == job_id,
            TailoredResumeRow.status == "approved"))
        prev = "approved" if approved else "to_apply"
    if target.status == "applied":
        job.applied_date = None
    job.status = prev
    if note is not None and len(note) > 200:
        raise TrackerError("note too long (max 200 chars)")
    session.add(ApplicationEventRow(user_id=user_id, job_id=job_id, status=UNDO, at=now or datetime.now(UTC),
                                    note=f"reverted #{target.id}: {target.status} -> {prev}" + (f"; {note}" if note else "")))
    session.commit()
    if audit:
        audit.log("tracker", "undo", user_id=user_id, job_id=job_id, reverted=target.status, now_status=prev)
    return job


def close_stale(session, user_id: str, today: date, days: int = 30, *, dry_run: bool = True,
                audit: AuditLog | None = None) -> list[dict]:
    """Applications quiet for `days`+ -> no_response. dry_run lists them without changing anything."""
    stale = [f for f in followups(session, user_id, today, days, max_days=None) if f["status"] == "applied"]
    if not dry_run:
        for f in stale:
            update_status(session, user_id, f["job_id"], "no_response", note=f"no reply for {f['days_quiet']} days",
                          audit=audit)
    return stale


def history(session, user_id: str, job_id: int) -> list[ApplicationEventRow]:
    return list(session.scalars(select(ApplicationEventRow).where(
        ApplicationEventRow.user_id == user_id, ApplicationEventRow.job_id == job_id)
        .order_by(ApplicationEventRow.id)))   # append order; `at` can be backdated


def _last_activity(session, user_id: str, job: JobRow) -> date | None:
    ev = session.scalar(select(ApplicationEventRow.at).where(
        ApplicationEventRow.user_id == user_id, ApplicationEventRow.job_id == job.id)
        .order_by(ApplicationEventRow.at.desc()))
    return ev.date() if ev else job.applied_date   # imported tracker rows have a date but no events


def followups(session, user_id: str, today: date, days: int = 7, max_days: int | None = 30) -> list[dict]:
    """Active applications quiet for `days`..`max_days`: the window where a nudge still makes sense.
    Older ones are stale: close them with close_stale / no_response instead. max_days=None: no upper bound."""
    out = []
    for job in session.scalars(select(JobRow).where(JobRow.user_id == user_id, JobRow.status.in_(ACTIVE),
                                                    JobRow.dup_of_id.is_(None))):
        last = _last_activity(session, user_id, job)
        quiet = (today - last).days if last is not None else None
        if quiet is not None and quiet >= days and (max_days is None or quiet <= max_days):
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
    by_status = Counter(j.status for j in jobs if j.status in FUNNEL)
    reached = [j for j in jobs if j.status in FUNNEL]
    voided = _voided(session, user_id)
    interviewed = {e.job_id for e in session.scalars(select(ApplicationEventRow).where(
        ApplicationEventRow.user_id == user_id, ApplicationEventRow.status.in_(("interview", "offer"))))
        if e.id not in voided}
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
