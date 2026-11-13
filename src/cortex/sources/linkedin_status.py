"""Is a LinkedIn job still open? Reads LinkedIn's PUBLIC, no-login job page for each candidate job.

This is a deliberate, narrow exception to "no direct LinkedIn scraping", made by Surya on 2026-11-13 (see
AGENTS.md, LINKEDIN OPEN CHECK). The limits are what make it acceptable, so they are enforced in code:
- Off unless config `linkedin_check.enabled: true`. Never part of `cortex daily`.
- No login, no cookies, no account: plain anonymous GET of the public guest endpoint. Nothing is posted.
- Only to_apply jobs from source linkedin, posted (else added) within `max_age_days`, newest first,
  at most `max_per_run` per run, one request every `delay_seconds` (+ jitter).
- The FIRST sign of throttling (429, 999, 403, any redirect such as an authwall) stops the whole run.
- Only a clear "No longer accepting applications" or a 404/410 marks a job closed; anything unclear stays
  to_apply. Closed jobs become not_accepting with status_raw "auto: ..." (same status the sheet uses).
- Audit log: counts only, never page content.
Page markers were NOT verifiable from the build environment (LinkedIn is unreachable there and was not to be
hit from a cloud machine anyway): first real run is `cortex check-open --dry-run` on the Mac.
"""
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
from sqlalchemy import func, select

from cortex.core.audit import AuditLog
from cortex.db.schema import JobRow

GUEST_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
MAX_BYTES = 3_000_000
_CLOSED = re.compile(r"no longer accepting applications", re.IGNORECASE)
_OPEN_MARKERS = re.compile(r"show-more-less-html|top-card-layout|description__text", re.IGNORECASE)
_ID = re.compile(r"^\d{6,15}$")


class Throttled(Exception):
    """LinkedIn pushed back. Stop now; trying harder is exactly what gets an IP blocked."""


def classify(status: int, body: str) -> str:
    """'open' | 'closed' | 'unknown'. Raises Throttled on any push-back."""
    if status in (404, 410):
        return "closed"                       # posting removed
    if status in (429, 999, 403) or 300 <= status < 400:
        raise Throttled(f"HTTP {status}")
    if status != 200:
        return "unknown"
    if _CLOSED.search(body):
        return "closed"
    return "open" if _OPEN_MARKERS.search(body) else "unknown"


def check_one(client: httpx.Client, job_id: str) -> str:
    if not _ID.match(job_id or ""):
        return "unknown"
    try:
        r = client.get(GUEST_URL.format(job_id=job_id), follow_redirects=False, timeout=20,
                       headers={"Accept": "text/html", "Accept-Language": "en-US,en;q=0.8"})
    except httpx.HTTPError:
        return "unknown"
    if len(r.content) > MAX_BYTES:
        return "unknown"
    return classify(r.status_code, r.text)


@dataclass
class CheckResult:
    candidates: int = 0
    checked: int = 0
    open: int = 0
    closed: int = 0
    unknown: int = 0
    stopped: str | None = None
    closed_jobs: list[tuple[int, str, str]] = field(default_factory=list)   # id, company, title


def candidates(session, user_id: str, today: date, max_age_days: int, limit: int) -> list[JobRow]:
    cutoff = today - timedelta(days=max_age_days)
    age = func.coalesce(JobRow.posted, JobRow.date_added)
    q = (select(JobRow).where(JobRow.user_id == user_id, JobRow.source == "linkedin", JobRow.status == "to_apply",
                              JobRow.dup_of_id.is_(None), age.is_not(None), age >= cutoff)
         .order_by(age.desc(), JobRow.id.desc()).limit(limit))
    return list(session.scalars(q))


def check_open(session, cfg: dict, user_id: str, *, today: date | None = None, client: httpx.Client | None = None,
               sleep: Callable[[float], None] = time.sleep, dry_run: bool = False, limit: int | None = None,
               audit: AuditLog | None = None) -> CheckResult:
    c = cfg.get("linkedin_check") or {}
    if not c.get("enabled", False):
        raise PermissionError("linkedin_check is disabled: set linkedin_check.enabled: true in config.yaml "
                              "(read the trade-off in AGENTS.md first)")
    today = today or date.today()  # noqa: DTZ011 (local date)
    rows = candidates(session, user_id, today, int(c.get("max_age_days", 30)),
                      min(int(limit or c.get("max_per_run", 60)), int(c.get("max_per_run", 60))))
    delay = max(float(c.get("delay_seconds", 6)), 3.0)          # never faster than one request per 3s
    res = CheckResult(candidates=len(rows))
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    try:
        for i, r in enumerate(rows):
            if i:
                sleep(delay + random.uniform(0, delay / 2))
            try:
                verdict = check_one(client, r.external_id)
            except Throttled as e:
                res.stopped = f"LinkedIn pushed back ({e}); stopped after {res.checked} checks. Try again tomorrow."
                break
            res.checked += 1
            setattr(res, verdict, getattr(res, verdict) + 1)
            if verdict == "closed":
                res.closed_jobs.append((r.id, r.company, r.title))
                if not dry_run:
                    r.status, r.status_raw = "not_accepting", f"auto: LinkedIn says closed {today.isoformat()}"
    finally:
        if own:
            client.close()
    if dry_run:
        session.rollback()
    else:
        session.commit()
    if audit:
        audit.log("linkedin_check", "run", user_id=user_id, candidates=res.candidates, checked=res.checked,
                  open=res.open, closed=res.closed, unknown=res.unknown, stopped=bool(res.stopped), dry_run=dry_run)
    return res
