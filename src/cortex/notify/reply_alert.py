"""Reply-needed alert: a short email to Surya's OWN address as soon as the Gmail scan finds emails that wait on
his reply (2026-11-24: an opportunity was lost to an unanswered email). Separate from the daily digest so it
isn't buried. Same rules as the digest: only to GMAIL_ADDRESS, no links, sender DOMAIN + subject only, never the
email body or the sender address. Each email is alerted once; "Done" in the dashboard clears it.
"""
from datetime import UTC, datetime

from sqlalchemy import select

from cortex.db.schema import ReplyAlertRow
from cortex.notify.digest import safe, send_text


def open_replies(session, user_id: str) -> list[ReplyAlertRow]:
    return list(session.scalars(select(ReplyAlertRow).where(ReplyAlertRow.user_id == user_id,
                                                            ReplyAlertRow.done_at.is_(None))
                                .order_by(ReplyAlertRow.found_at.desc())))


def render(rows: list[ReplyAlertRow], now: datetime) -> tuple[str, str]:
    subject = f"CORTEX: {len(rows)} email{'s' if len(rows) != 1 else ''} waiting for your reply"
    out = ["These emails look like they need YOUR reply or action. Answer them in Gmail today:", ""]
    for r in rows:
        hours = max(0, int((now - r.found_at.replace(tzinfo=r.found_at.tzinfo or UTC)).total_seconds() // 3600))
        who = safe(r.company or r.sender_domain, 60)
        out.append(f"  - {who} | {safe(r.subject, 120)}  ({r.cue}; found {hours} h ago)")
    out += ["", "Mark each one Done in the dashboard once you've replied.",
            "This email has no links on purpose. CORTEX never replies for you."]
    return subject, "\n".join(out) + "\n"


def send_new_alerts(session, user_id: str, *, address: str | None, password: str | None, dry_run: bool = False,
                    smtp_factory=None, now: datetime | None = None, covered_by_digest: bool = False) -> int:
    """Email the replies not alerted yet (once each). Returns how many were alerted (0 = nothing new).
    covered_by_digest: the daily digest just listed them at the top, so mark them alerted without a 2nd email."""
    now = now or datetime.now(UTC)
    new = [r for r in open_replies(session, user_id) if r.alerted_at is None]
    if not new or dry_run:
        return len(new)
    if not covered_by_digest:
        subject, body = render(new, now)
        send_text(subject, body, address=address, password=password, smtp_factory=smtp_factory)
    for r in new:
        r.alerted_at = now.replace(tzinfo=None)
    session.commit()
    return len(new)


def mark_done(session, user_id: str, alert_id: int) -> bool:
    row = session.scalar(select(ReplyAlertRow).where(ReplyAlertRow.id == alert_id, ReplyAlertRow.user_id == user_id))
    if row is None or row.done_at is not None:
        return False
    row.done_at = datetime.now(UTC).replace(tzinfo=None)
    session.commit()
    return True
