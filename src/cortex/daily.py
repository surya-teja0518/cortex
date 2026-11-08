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
              digest: Callable, audit: AuditLog | None = None) -> DailyResult:
    """fetch(session) -> list[BoardResult]; score(session) -> BatchResult; digest(session, notes) -> status.
    Pass None to skip a step."""
    res = DailyResult()
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
