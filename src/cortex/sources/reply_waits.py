"""Reply reminders: emails where a recruiter or employer is WAITING ON SURYA.

Why (2026-11-24): he lost an opportunity because a recruiter's "please share your CTC / notice period /
availability" email sat unanswered until the regret mail came. CORTEX only looked for rejections and
interviews, so nothing nagged him. Now every daily run:

1. `needs_reply(mail)`: LOCAL rules, no LLM (email text never leaves the run, never reaches a prompt). An email
   waits on him when it ASKS for something (CTC, notice period, availability, "kindly share", "are you
   interested", an online test, a portal form) and is not a job-board blast, a newsletter or a rejection.
   kind "reply" = a person wants an answer (closed automatically when his Sent mail shows a reply);
   kind "action" = a test / portal form (closed when he marks it done: `cortex reply-done <id>` or the dashboard).
2. `refresh(...)`: re-checks every open reminder against his Sent mail and closes the answered ones; old ones
   expire (config gmail.reply_expire_days, default 21) so the list can't rot.
3. The digest puts open reminders FIRST, with how many days each has waited, and its subject line starts with
   "⚠ N waiting on you", every day until they're closed. The dashboard shows the same list.

Stored: message/thread id, `who` (display name or sender domain, never the address), subject (<= 120), what
it asks for, dates. Never the body.
"""
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from cortex.db.schema import ReplyWaitRow

# (label shown to him, pattern, kind)
ASKS: list[tuple[str, str, str]] = [
    ("CTC", r"\b(current|expected|present|e\.?)\s*ctc\b|\bctc\s*(details|expectations?)\b", "reply"),
    ("notice period", r"\bnotice\s+period\b|\blast\s+working\s+day\b", "reply"),
    ("availability", (r"\b(your|share|interview)\s+availability\b|\bwhen\s+(are|would)\s+you\s+be\s+(available|free)\b"
                     r"|\b(convenient|suitable)\s+(time|slot)\b"), "reply"),
    ("details", (r"\b(kindly|please|request\s+you\s+to)\s+(share|send|provide|confirm|fill)\b"
                r"|\bshare\s+(the|your)\s+(below\s+)?(details|updated\s+(resume|cv)|resume|cv|contact)\b"), "reply"),
    ("a reply", (r"\b(kindly|please)\s+(revert|reply|respond|let\s+(me|us)\s+know)\b"
                r"|\blooking\s+forward\s+to\s+(hearing\s+from\s+you|your\s+(response|reply))\b"
                r"|\bawaiting\s+your\s+(response|reply|confirmation)\b"
                 r"|\blet\s+(me|us)\s+know\s+(once|when|if|whether|your)\b"), "reply"),
    ("interest", r"\bare\s+you\s+(interested|open|looking)\b|\b(if|whether)\s+you\s+(are|would\s+be)\s+interested\b",
                 "reply"),
    ("online test", (r"\b(complete|take|attempt|start)\s+(the|your|this)\s+(following\s+)?(online\s+|technical\s+|coding\s+)?"
                    r"(assessment|test|challenge)\b|\btest\s+(link|credentials)\b"), "action"),
    ("portal form", (r"\badditional\s+information\s+(is\s+)?required\b|\bcomplete\s+your\s+(application|profile)\b"
                    r"|\b(pending|incomplete)\s+(application|documents?)\b"
                     r"|\bto\s+continue\s+with\s+your\s+(candidature|application)\b"
                     r"|\blink\s+(is|will\s+be)\s+valid\s+(until|till|for)\b"), "action"),
]
_ASKS = [(lbl, re.compile(p, re.IGNORECASE), kind) for lbl, p, kind in ASKS]

# Gmail-side keywords for the catch-up search (`cortex inbox --asks-days N`); local rules decide afterwards
GMAIL_ASK_QUERY = ('ctc OR "notice period" OR availability OR "kindly share" OR "please share" OR "kindly revert" '
                   'OR "let me know" OR "are you interested" OR assessment OR "test link" OR "additional information"')

# job boards, course sellers, newsletters: their "are you interested?" is a blast, not someone waiting on him
IGNORE_DOMAINS = ("naukri.com", "foundit.in", "monsterindia.com", "linkedin.com", "indeed.com", "glassdoor.com",
                  "instahyre.com", "iimjobs.com", "hirist.tech", "hirist.com", "cutshort.io", "ambitionbox.com",
                  "crio.in", "stratascratch.com", "substack.com", "apna.co", "shine.com", "timesjobs.com",
                  "wellfound.com", "unstop.com", "medium.com", "coursera.org", "udemy.com", "scaler.com", "talent500.co", "hackajob.com", "mail-hackajob.com")
# assessment platforms: often no-reply senders with "unsubscribe" footers, but the test link is the real thing
TEST_PLATFORMS = ("hirepro.in", "hackerrank.com", "mettl.com", "codility.com", "hackerearth.com", "glider.ai",
                  "imocha.io", "testgorilla.com", "codesignal.com", "talview.com", "mercer.com")
# the ask must be about a job: a card statement's "notice period for reporting a defaulter" is not (seen 2026-11-24)
JOB_CONTEXT = re.compile(r"\b(role|position|opening|opportunit(y|ies)|interview|applications?|applied|candidate|"
                         r"candidature|resume|cv|hiring|recruit\w*|talent|jobs?|vacanc(y|ies)|offers?|careers?)\b", re.IGNORECASE)
DO_NOT_REPLY = re.compile(r"\b(do\s+not|don'?t|please\s+do\s+not)\s+reply\b|notification[-\s]only|cannot\s+accept\s+"
                          r"incoming|system[-\s]generated", re.IGNORECASE)
NO_ACTION = re.compile(r"\bno\s+(further\s+)?action\s+(is\s+)?(required|needed)\b", re.IGNORECASE)
REJECTION = re.compile(r"\bregret\s+to\s+inform\b|\bnot\s+(be\s+)?moving\s+forward\b|\bwon'?t\s+be\s+moving\s+forward\b"
                       r"|\bunable\s+to\s+(offer|proceed)\b|\bnot\s+been\s+selected\b|\bposition\s+has\s+(now\s+)?been\s+filled\b",
                       re.IGNORECASE)


@dataclass
class Ask:
    kind: str          # reply | action
    asks: list[str]    # labels, e.g. ["CTC", "notice period"]
    who: str


def _domain(sender: str) -> str:
    m = re.search(r"@([A-Za-z0-9.-]+)", sender or "")
    return m.group(1).lower().rstrip(">.") if m else ""


def _on(domain: str, suffixes) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in suffixes)


def who_from(sender: str) -> str:
    """Display name ('Prachi Saxena', 'Zensar Technologies Limited') or else the sender's domain. Never the
    address. Control chars and links dropped, capped."""
    name = re.sub(r"<[^>]*>", "", sender or "").strip().strip('"').strip()
    if "@" in name:
        name = ""
    name = re.sub(r"[\x00-\x1f\x7f]|https?://\S+|www\.\S+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return (name or _domain(sender) or "unknown sender")[:80]


def needs_reply(sender: str, subject: str, text: str, ignore_domains=IGNORE_DOMAINS) -> Ask | None:
    domain = _domain(sender)
    if _on(domain, ignore_domains):
        return None
    body = f"{subject}\n{text}"
    if REJECTION.search(body) or NO_ACTION.search(body):
        return None
    hits = [(lbl, kind) for lbl, rx, kind in _ASKS if rx.search(body)]
    if not hits or not (JOB_CONTEXT.search(body) or _on(domain, TEST_PLATFORMS)):
        return None
    platform = _on(domain, TEST_PLATFORMS)
    automated = bool(re.search(r"no[-_.]?reply|do[-_.]?not[-_.]?reply|notifications?@|mailer", sender or "", re.IGNORECASE))
    newsletter = bool(re.search(r"\bunsubscribe\b", text or "", re.IGNORECASE))
    actions = [lbl for lbl, kind in hits if kind == "action"]
    if platform or automated or newsletter:
        # nobody reads a reply to a no-reply/newsletter sender: only a test or a portal form counts there,
        # and a newsletter only when it comes from a real assessment platform
        if not actions or (newsletter and not platform and not automated):
            return None
        return Ask("action", actions, who_from(sender))
    kind = "action" if actions else "reply"      # a test / portal form is done on their site, not by email
    return Ask(kind, list(dict.fromkeys(lbl for lbl, _ in hits)), who_from(sender))


# ---------- stored reminders ----------

def add(session, user_id: str, mail, ask: Ask) -> ReplyWaitRow | None:
    """Record a reminder for this message (once)."""
    if session.scalar(select(ReplyWaitRow.id).where(ReplyWaitRow.user_id == user_id,
                                                    ReplyWaitRow.message_id == mail.id)):
        return None
    row = ReplyWaitRow(user_id=user_id, message_id=mail.id, thread_id=mail.thread_id or None, who=ask.who,
                       subject=(mail.subject or "")[:120], kind=ask.kind, asks=", ".join(ask.asks)[:160],
                       asked_at=_naive(mail.at or datetime.now(UTC)))
    session.add(row)
    return row


def _naive(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def open_waits(session, user_id: str) -> list[ReplyWaitRow]:
    return list(session.scalars(select(ReplyWaitRow).where(ReplyWaitRow.user_id == user_id,
                                                           ReplyWaitRow.closed_at.is_(None))
                                .order_by(ReplyWaitRow.asked_at)))


def close(session, user_id: str, wait_id: int, how: str = "done") -> ReplyWaitRow:
    row = session.get(ReplyWaitRow, wait_id)
    if row is None or row.user_id != user_id:
        raise LookupError(f"no reminder #{wait_id}")
    if row.closed_at is None:
        row.closed_at, row.closed_how = datetime.now(UTC).replace(tzinfo=None), how
        session.commit()
    return row


@dataclass
class RefreshResult:
    replied: int = 0
    expired: int = 0
    still_open: int = 0
    unchecked: int = 0      # Gmail check failed for these: left open, retried next run


def refresh(session, user_id: str, client, *, now: datetime | None = None, expire_days: int = 21,
            dry_run: bool = False) -> RefreshResult:
    """Close reminders he has answered (any mail he sent in that thread or to that sender after the ask) and
    expire old ones. A failed Gmail check leaves the reminder open: better one nag too many than one too few."""
    now = _naive(now or datetime.now(UTC))
    res = RefreshResult()
    for w in open_waits(session, user_id):
        if now - w.asked_at > timedelta(days=expire_days):
            res.expired += 1
            if not dry_run:
                w.closed_at, w.closed_how = now, "expired"
            continue
        try:
            answered = client.replied_after(w.thread_id or "", w.message_id, w.asked_at.replace(tzinfo=UTC))
        except Exception as e:
            if "access denied" in str(e) or "rate limit" in str(e):
                raise
            res.unchecked += 1
            res.still_open += 1
            continue
        if answered:
            res.replied += 1
            if not dry_run:
                w.closed_at, w.closed_how = now, "replied"
        else:
            res.still_open += 1
    if dry_run:
        session.rollback()
    else:
        session.commit()
    return res


def days_waiting(w: ReplyWaitRow, now: datetime | None = None) -> int:
    return max(0, (_naive(now or datetime.now(UTC)) - w.asked_at).days)
