"""Email digest: a once-a-day summary sent to Surya's own Gmail. Output only.

Rules (AGENTS.md):
- Recipient is ALWAYS the sender (GMAIL_ADDRESS). There is no recipient parameter anywhere.
- Counts + company/title only. Never resume content, never job descriptions.
- No links at all: no approve/act links (approval stays in the dashboard) and no posting URLs either,
  so a hostile posting can't put a phishing link in front of him through CORTEX.
- Company/title are untrusted: control chars and newlines removed (no header/body injection), length-capped.
- Plain text. The SMTP password is read from .env at send time and never printed or logged.
"""
import re
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import date
from email.message import EmailMessage

from sqlalchemy import func, select

from cortex.agents.batch_score import latest_llm_scores
from cortex.agents.tracker import followups, report
from cortex.core.audit import AuditLog
from cortex.db.schema import DigestRunRow, JobRow
from cortex.review import pending_approvals, ready_to_apply

SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
_CTRL = re.compile(r"[\x00-\x1f\x7f  ]+")
_URLISH = re.compile(r"(https?://|www\.)\S*", re.IGNORECASE)
TOP_BANDS = ("P0", "P1")


def safe(text: str | None, cap: int = 80) -> str:
    t = _URLISH.sub("[link removed]", _CTRL.sub(" ", text or "")).strip()
    return t if len(t) <= cap else t[: cap - 1] + "…"


@dataclass
class DigestLine:
    job_id: int
    company: str
    title: str
    band: str
    score: int
    llm: bool


@dataclass
class Digest:
    day: date
    new_total: int = 0
    new_by_band: dict[str, int] = field(default_factory=dict)
    new_top: list[DigestLine] = field(default_factory=list)
    new_flagged: int = 0          # injection-flagged: manual review in the dashboard
    new_gated: int = 0
    pending: list[tuple[str, str]] = field(default_factory=list)       # (company, title) awaiting approval
    ready: list[tuple[str, str]] = field(default_factory=list)         # approved, not applied yet
    followups: list[tuple[int, str, str, int]] = field(default_factory=list)   # id, company, title, days quiet
    applied_total: int = 0
    response_rate: float | None = None
    notes: list[str] = field(default_factory=list)   # pipeline status lines from `cortex daily`
    max_job_id: int = 0

    @property
    def empty(self) -> bool:
        failed = any("FAILED" in n or "failed:" in n for n in self.notes)   # a broken pipeline is news
        return not (self.new_total or self.pending or self.ready or self.followups or failed)


def last_watermark(session, user_id: str) -> int:
    return session.scalar(select(func.max(DigestRunRow.max_job_id)).where(DigestRunRow.user_id == user_id)) or 0


def build_digest(session, user_id: str, today: date, *, top_n: int = 15, followup_days: int = 7,
                 stale_days: int = 30, notes: list[str] | None = None) -> Digest:
    since = last_watermark(session, user_id)
    d = Digest(day=today, notes=list(notes or []))
    d.max_job_id = session.scalar(select(func.max(JobRow.id)).where(JobRow.user_id == user_id)) or since
    new = list(session.scalars(select(JobRow).where(JobRow.user_id == user_id, JobRow.id > since,
                                                    JobRow.dup_of_id.is_(None), JobRow.status == "to_apply")))
    d.new_gated = sum(1 for r in new if r.gate_reason)
    d.new_flagged = sum(1 for r in new if r.sanitizer_flags and not r.gate_reason)
    live = [r for r in new if not r.gate_reason and not r.sanitizer_flags]
    llm = latest_llm_scores(session, user_id, [r.id for r in live])
    lines = []
    for r in live:
        o = llm.get(r.id)
        lines.append(DigestLine(r.id, safe(r.company, 40), safe(r.title), o.band if o else r.band,
                                o.score if o else r.score, o is not None))
    d.new_total = len(live)
    for ln in lines:
        d.new_by_band[ln.band] = d.new_by_band.get(ln.band, 0) + 1
    d.new_top = sorted(lines, key=lambda x: (-x.score, -x.job_id))[:top_n]
    d.pending = [(safe(p["company"], 40), safe(p["role"])) for p in pending_approvals(session, user_id)]
    d.ready = [(safe(p["company"], 40), safe(p["role"])) for p in ready_to_apply(session, user_id)]
    d.followups = [(f["job_id"], safe(f["company"], 40), safe(f["role"]), f["days_quiet"])
                   for f in followups(session, user_id, today, followup_days, stale_days)]
    rep = report(session, user_id, today)
    d.applied_total, d.response_rate = rep.applied_total, rep.response_rate
    return d


def subject(d: Digest) -> str:
    top = sum(v for k, v in d.new_by_band.items() if k in TOP_BANDS)
    return f"CORTEX {d.day.isoformat()}: {d.new_total} new ({top} P0/P1), {len(d.followups)} follow-ups"


def render(d: Digest) -> str:
    out = [f"CORTEX daily digest, {d.day.isoformat()}", ""]
    bands = "  ".join(f"{b}={d.new_by_band.get(b, 0)}" for b in ("P0", "P1", "P2", "P3"))
    out.append(f"NEW JOBS: {d.new_total}  ({bands})")
    if d.new_gated or d.new_flagged:
        out.append(f"  also {d.new_gated} gated out, {d.new_flagged} flagged for manual review (dashboard)")
    for ln in d.new_top:
        out.append(f"  [{ln.band} {ln.score}{'' if ln.llm else ' rules'}] #{ln.job_id} {ln.company} | {ln.title}")
    if d.new_total > len(d.new_top):
        out.append(f"  ... {d.new_total - len(d.new_top)} more in the dashboard")
    if d.pending:
        out += ["", f"WAITING FOR YOUR APPROVAL (dashboard): {len(d.pending)}"]
        out += [f"  {c} | {t}" for c, t in d.pending[:10]]
    if d.ready:
        out += ["", f"APPROVED, NOT APPLIED YET: {len(d.ready)}"]
        out += [f"  {c} | {t}" for c, t in d.ready[:10]]
    if d.followups:
        out += ["", f"FOLLOW-UPS DUE: {len(d.followups)}"]
        out += [f"  {q}d quiet  #{i} {c} | {t}" for i, c, t, q in d.followups[:10]]
    rate = f"{d.response_rate:.0%}" if d.response_rate is not None else "n/a"
    out += ["", f"Applications so far: {d.applied_total}, response rate {rate}"]
    if d.notes:
        out += ["", "PIPELINE:"] + [f"  {safe(n, 200)}" for n in d.notes]
    out += ["", "Review, approve and apply in the dashboard on your Mac. CORTEX never applies for you.",
            "This email has no links on purpose."]
    return "\n".join(out) + "\n"


class DigestError(RuntimeError):
    pass


def send(d: Digest, *, address: str | None, password: str | None, smtp_factory=None) -> None:
    """Send the digest to `address` FROM `address`. Raises DigestError with a message that never
    contains the password."""
    if not address or not password:
        raise DigestError("GMAIL_ADDRESS / GMAIL_APP_PASSWORD missing in .env")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address):
        raise DigestError("GMAIL_ADDRESS doesn't look like an email address")
    msg = EmailMessage()
    msg["From"] = msg["To"] = address
    msg["Subject"] = subject(d)
    msg.set_content(render(d))
    factory = smtp_factory or (lambda: smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(),
                                                        timeout=30))
    try:
        with factory() as s:
            s.login(address, password)
            s.send_message(msg, from_addr=address, to_addrs=[address])
    except smtplib.SMTPAuthenticationError:
        raise DigestError("Gmail rejected the login: check GMAIL_APP_PASSWORD (an App Password, not your "
                          "Google password; needs 2-step verification)") from None
    except (smtplib.SMTPException, OSError) as e:
        raise DigestError(f"sending failed: {type(e).__name__}") from None


def run_digest(session, user_id: str, today: date, *, cfg: dict, address: str | None, password: str | None,
               dry_run: bool = False, notes: list[str] | None = None, audit: AuditLog | None = None,
               smtp_factory=None) -> tuple[Digest, str]:
    """Build, then send (unless dry_run or nothing to say). Returns (digest, what happened)."""
    dcfg, tcfg = cfg.get("digest", {}), cfg.get("tracker", {})
    d = build_digest(session, user_id, today, top_n=dcfg.get("top_n", 15),
                     followup_days=tcfg.get("followup_days", 7), stale_days=tcfg.get("stale_days", 30), notes=notes)
    if dry_run:
        return d, "dry run: not sent"
    if d.empty and not dcfg.get("send_when_empty", False):
        return d, "nothing new: not sent"
    send(d, address=address, password=password, smtp_factory=smtp_factory)
    session.add(DigestRunRow(user_id=user_id, max_job_id=d.max_job_id, new_jobs=d.new_total))
    session.commit()
    if audit:
        audit.log("digest", "sent", user_id=user_id, new_jobs=d.new_total, followups=len(d.followups),
                  pending=len(d.pending))
    return d, "sent"
