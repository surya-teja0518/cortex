"""`cortex daily`: fetch -> LLM-score new jobs -> email digest. Meant to run once a day from launchd.

Each step is isolated: a failing step becomes a one-line note in the digest and the next step still runs,
so a Groq outage or a dead ATS board never costs him the day's email. Notes carry counts and error types
only (never job text, never secrets). Nothing here tailors, approves or applies: those stay human-driven.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from cortex.core.audit import AuditLog


@dataclass
class DailyResult:
    notes: list[str] = field(default_factory=list)
    digest_status: str = "not run"
    failed_steps: list[str] = field(default_factory=list)


def run_daily(session, cfg: dict, user_id: str, today: date, *, fetch: Callable | None, score: Callable | None,
              digest: Callable, audit: AuditLog | None = None, pull: Callable | None = None,
              sync: Callable | None = None, inbox: Callable | None = None) -> DailyResult:
    """pull(session) -> list[ActorResult]; fetch(session) -> list[BoardResult]; score(session) -> BatchResult;
    digest(session, notes) -> status. Pass None to skip a step."""
    res = DailyResult()
    if inbox is not None:   # rejection / interview emails -> tracker (unambiguous ones only)
        try:
            r = inbox(session)
            for jid, verdict, company, title in r.applied:
                res.notes.append(f"gmail: marked {verdict}: {company} | {title} (undo: cortex undo {jid})")
            for verdict, companies, why in r.to_check:
                res.notes.append(f"gmail: check yourself: possible {verdict} from {companies} ({why})")
            if not r.applied and not r.to_check:
                res.notes.append(f"gmail: {r.scanned} new emails, nothing about your applications"
                                 + (f" ({r.skipped} skipped, retried tomorrow)" if getattr(r, "skipped", 0) else ""))
        except Exception as e:  # noqa: BLE001 — one step failing must not cost the digest
            session.rollback()
            res.failed_steps.append("inbox")
            res.notes.append(f"gmail FAILED: {type(e).__name__}"
                             + (f" ({e})" if type(e).__name__ == "GmailError" else ""))
    if sync is not None:   # after inbox, so today's email updates reach the sheet today; then bring in what SYNAPSE / Surya put in the sheet, write back known statuses
        try:
            r = sync(session)
            res.notes.append(f"sheet sync: {r.read.inserted} new rows, {r.written} status cells written back"
                             + (f" ({len(r.to_write)} would be written; dry run)" if r.to_write and not r.written else ""))
        except Exception as e:  # noqa: BLE001 — one step failing must not cost the digest
            session.rollback()
            res.failed_steps.append("sync")
            res.notes.append(f"sheet sync FAILED: {type(e).__name__}"
                             + (f" ({e})" if type(e).__name__ == "SheetsError" else ""))
    if pull is not None:
        try:
            actors = pull(session)
            bad = [f"{a.actor} ({a.error})" for a in actors if a.error]
            res.notes.append(f"apify: {sum(a.datasets for a in actors)} datasets, {sum(a.inserted for a in actors)} new"
                             f" jobs" + (f"; failed: {', '.join(bad)}" if bad else ""))
        except Exception as e:  # noqa: BLE001 — one step failing must not cost the digest
            session.rollback()
            res.failed_steps.append("pull")
            res.notes.append(f"apify FAILED: {type(e).__name__}"
                             + (f" ({e})" if type(e).__name__ == "ApifyError" else ""))
    if fetch is not None:
        try:
            boards = fetch(session)
            bad = [f"{b.ats}:{b.board} ({b.error})" for b in boards if b.error]
            res.notes.append(f"fetch: {len(boards)} boards, {sum(b.inserted for b in boards)} new jobs"
                             + (f"; failed: {', '.join(bad)}" if bad else ""))
        except Exception as e:  # noqa: BLE001 — one step failing must not cost the digest
            session.rollback()
            res.failed_steps.append("fetch")
            res.notes.append(f"fetch FAILED: {type(e).__name__}")
    if score is not None:
        try:
            b = score(session)
            if getattr(b, "dry_run", False):
                res.notes.append(f"LLM scoring: dry run, would score {b.candidates} jobs (no calls made)")
            else:
                res.notes.append(f"LLM scoring: {b.scored} scored, {b.unscored} kept rule scores"
                                 + (f" ({b.stopped_early})" if b.stopped_early else ""))
        except Exception as e:  # noqa: BLE001
            session.rollback()
            res.failed_steps.append("score")
            res.notes.append(f"LLM scoring FAILED: {type(e).__name__} (rule scores used)")
    try:
        res.digest_status = digest(session, res.notes)
    except Exception as e:  # noqa: BLE001
        res.failed_steps.append("digest")
        res.digest_status = f"FAILED: {e}" if type(e).__name__ == "DigestError" else f"FAILED: {type(e).__name__}"
    if audit:
        audit.log("daily", "run", user_id=user_id, day=today.isoformat(), failed=res.failed_steps,
                  digest=res.digest_status[:60])
    return res
