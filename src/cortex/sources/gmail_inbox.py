"""Gmail inbox watcher: spot rejection / interview emails for jobs Surya applied to and update the tracker.
Approved by Surya on 2026-11-15 (AGENTS.md, GMAIL INBOX), with these rules enforced in code:

- READ ONLY: Gmail API with the `gmail.readonly` scope (Google enforces it), recent days only, primary inbox.
- Classification is LOCAL, plain rules, NO LLM: email content never leaves his Mac and never reaches a prompt.
  Email text is untrusted input; it is only searched for phrases, never executed or followed.
- AUTO-UPDATE only when unambiguous: a STRONG phrase ("not moving forward", "schedule an interview") AND the email
  names exactly ONE company he has an active application at (applied / interview) AND that company has exactly one
  active application. Anything else (several applications at that company, only a weak word like
  "unfortunately", both rejection and interview language) is a suggestion in the digest, never a tracker change.
- Every auto-update is an ordinary tracker event (`cortex undo <id>` reverts it) with the note "auto from Gmail".
- Stored per message: message id, verdict, matched job ids, whether it was applied. Never the body, never the
  sender address; the subject is kept (truncated) only for the dashboard/CLI, never put in the email digest.
- Credentials: an OAuth client (gitignored) and a token file (gitignored, refresh token = secret, chmod 600).
"""
import base64
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import select

from cortex.agents.tracker import ACTIVE, TrackerError, update_status
from cortex.core.audit import AuditLog
from cortex.core.parsing import normalize_company
from cortex.core.sanitize import sanitize_jd
from cortex.db.schema import InboxFindingRow, JobRow
from cortex.sources.ats import html_to_text

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_TEXT = 50_000
_ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")

REJECT_STRONG = [
    r"not (be )?moving forward", r"move forward with other candidates", r"(decided|chosen) to (pursue|proceed with|move forward with) other",
    r"regret to inform", r"not been selected", r"were not selected", r"will not be progressing", r"not to proceed with your",
    r"position has (now )?been filled", r"decided not to (move forward|proceed)", r"no longer under consideration",
    r"other (applicants|candidates) whose (skills|experience|qualifications)", r"unable to offer you",
]
REJECT_WEAK = [r"\bunfortunately\b"]
INTERVIEW_STRONG = [
    r"schedule (a|an|your) (call|interview|chat|conversation|discussion)", r"invite you (to|for) (an? )?interview",
    r"interview (invitation|invite|slot|schedule)", r"(online|technical|coding) (assessment|test)",
    r"hackerrank|codility|hackerearth|codesignal|mettl", r"shortlisted", r"share your availability",
    r"next round", r"move (you )?forward to the next",
]
# job-alert senders: never about an application of his
SKIP_SENDERS = ("jobalerts-noreply@linkedin.com", "jobs-listings@linkedin.com", "alerts@naukri.com",
                "naukrialerts@naukri.com", "noreply@glassdoor.com")

_REJ_S = [re.compile(p, re.IGNORECASE) for p in REJECT_STRONG]
_REJ_W = [re.compile(p, re.IGNORECASE) for p in REJECT_WEAK]
_INT_S = [re.compile(p, re.IGNORECASE) for p in INTERVIEW_STRONG]


class GmailError(Exception):
    pass


@dataclass
class Mail:
    id: str
    sender: str
    subject: str
    text: str


@dataclass
class Finding:
    message_id: str
    verdict: str            # rejected | interview
    job_ids: list[int]
    auto: bool
    why: str
    subject: str = ""


# ---------- pure decisions ----------

def classify(text: str) -> tuple[str | None, bool]:
    """(verdict, strong). Both rejection and interview language -> (None, False): too mixed to call."""
    rej = any(r.search(text) for r in _REJ_S)
    weak_rej = any(r.search(text) for r in _REJ_W)
    inter = any(r.search(text) for r in _INT_S)
    if (rej or weak_rej) and inter:
        return None, False
    if rej:
        return "rejected", True
    if inter:
        return "interview", True
    if weak_rej:
        return "rejected", False
    return None, False


def _company_key(name: str) -> str | None:
    key = normalize_company(name)
    return key if len(key.replace(" ", "")) >= 3 else None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _named(key: str, raw_company: str, text: str) -> bool:
    """Company named in text: its words in order ('wells fargo'), or joined into one token as in a sender
    domain ('wellsfargo', 'betacorp')."""
    toks = _tokens(text)
    if f" {key} " in " " + " ".join(toks) + " ":
        return True
    compact = {key.replace(" ", ""), "".join(_tokens(raw_company))}
    return any(len(c) >= 5 and c in toks for c in compact)


def _tech_words() -> set[str]:
    from cortex.agents.truth_guard import TECH_TERMS
    return {" ".join(_tokens(t)) for t in TECH_TERMS} | {"oracle", "microsoft", "google", "amazon", "ibm", "sap"}


TECH_NAMED = _tech_words()   # companies whose name is also a technology / everyday tech word


def matching_jobs(mail: Mail, active: list[JobRow]) -> dict[str, list[JobRow]]:
    """{company key: active jobs at that company} for every company the email names (subject, sender, body).
    A company whose name is also a technology (Oracle, Snowflake, Databricks...) only counts when it is the
    SENDER: "your Oracle DBA application at Zensar" must not reject his Oracle application (seen 2026-11-15)."""
    hay = f"{mail.subject} {mail.sender} {mail.text}"
    out: dict[str, list[JobRow]] = {}
    for j in active:
        key = _company_key(j.company)
        if not key:
            continue
        where = mail.sender if key in TECH_NAMED else hay
        if _named(key, j.company, where):
            out.setdefault(key, []).append(j)
    return out


def decide(mail: Mail, active: list[JobRow]) -> Finding | None:
    if any(s in mail.sender.lower() for s in SKIP_SENDERS):
        return None
    verdict, strong = classify(f"{mail.subject}\n{mail.text}")
    if verdict is None:
        return None
    found = matching_jobs(mail, active)
    if not found:
        return None
    ids = sorted(j.id for jobs in found.values() for j in jobs)
    if not strong:
        return Finding(mail.id, verdict, ids, False, "only a weak signal (e.g. 'unfortunately')", mail.subject)
    if len(found) > 1:
        return Finding(mail.id, verdict, ids, False, "email names more than one company you applied to", mail.subject)
    if len(ids) > 1:
        return Finding(mail.id, verdict, ids, False, f"{len(ids)} active applications at this company", mail.subject)
    key, jobs = next(iter(found.items()))
    if not _named(key, jobs[0].company, f"{mail.subject} {mail.sender}"):   # only deep in the body: don't act
        return Finding(mail.id, verdict, ids, False, "company only mentioned in the body", mail.subject)
    return Finding(mail.id, verdict, ids, True, "clear match", mail.subject)


# ---------- Gmail API (read only) ----------

def load_token(token_file: str) -> str:
    """Access token from the saved OAuth token; refreshes it (and saves the refresh) when expired."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as e:
        raise GmailError('google-auth missing: pip install -e ".[gmail]"') from e
    p = Path(token_file)
    if not p.exists():
        raise GmailError(f"{token_file} not found: run `cortex gmail-auth` once")
    try:
        creds = Credentials.from_authorized_user_file(str(p), SCOPES)
        if not creds.valid:
            creds.refresh(Request())
            p.write_text(creds.to_json())
            p.chmod(0o600)
    except Exception as e:  # noqa: BLE001 — auth errors can echo request details: type only
        raise GmailError(f"Gmail auth failed ({type(e).__name__}): run `cortex gmail-auth` again") from None
    if set(creds.scopes or SCOPES) - set(SCOPES):
        raise GmailError("token has more than read-only Gmail access: delete it and run `cortex gmail-auth`")
    return creds.token


def authorize(client_file: str, token_file: str) -> None:
    """One-time browser sign-in (read-only scope). Saves the token with owner-only permissions."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise GmailError('google-auth-oauthlib missing: pip install -e ".[gmail]"') from e
    if not Path(client_file).exists():
        raise GmailError(f"{client_file} not found: download your OAuth client (Desktop app) JSON there")
    creds = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES).run_local_server(port=0)
    p = Path(token_file)
    p.write_text(creds.to_json())
    p.chmod(0o600)


def _header(msg: dict, name: str) -> str:
    for h in (msg.get("payload") or {}).get("headers") or []:
        if isinstance(h, dict) and str(h.get("name", "")).lower() == name:
            return str(h.get("value", ""))
    return ""


def _body_text(part: dict) -> str:
    """Plain text of a message: text/plain parts preferred, else HTML stripped. Attachments ignored."""
    plain, html = [], []

    def walk(p):
        mime = p.get("mimeType", "")
        data = (p.get("body") or {}).get("data")
        if data and not (p.get("filename") or ""):
            raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")
            (plain if mime == "text/plain" else html if mime == "text/html" else []).append(raw)
        for c in p.get("parts") or []:
            if isinstance(c, dict):
                walk(c)
    walk(part or {})
    text = "\n".join(plain) or html_to_text("\n".join(html))
    return sanitize_jd(text[:MAX_TEXT])[0]


RATE_LIMIT_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "concurrentLimitExceeded"}
_REASON = re.compile(r"^[A-Za-z]{3,40}$")


def _reason(r: httpx.Response) -> str:
    """Google's machine reason ('userRateLimitExceeded', 'insufficientPermissions'...). Only that one word
    is ever shown: the rest of the error body is not echoed."""
    try:
        err = r.json().get("error") or {}
        reason = ((err.get("errors") or [{}])[0].get("reason")
                  or next((d.get("reason") for d in err.get("details") or [] if d.get("reason")), ""))
    except Exception:  # noqa: BLE001 — not JSON: no reason
        return ""
    return reason if isinstance(reason, str) and _REASON.match(reason) else ""


def _retry_after(r: httpx.Response, default: float) -> float:
    try:
        return min(max(float(r.headers.get("retry-after", default)), 1.0), 60.0)
    except ValueError:
        return default


class GmailClient:
    def __init__(self, token: str, client: httpx.Client | None = None, sleep=None):
        import time
        self._token = token
        self._c = client or httpx.Client(timeout=30)
        self._sleep = sleep or time.sleep

    def __repr__(self) -> str:
        return "GmailClient(token=***)"

    def _get(self, url: str, **params):
        for rl_try in range(4):           # Gmail rate limits come back as 403/429 with a reason: wait, retry
            r = self._send(url, params)
            reason = _reason(r) if r.status_code in (403, 429) else ""
            if r.status_code == 429 or reason in RATE_LIMIT_REASONS:
                if rl_try == 3:
                    raise GmailError(f"HTTP {r.status_code}: Gmail rate limit ({reason or 'too many requests'}); "
                                     "try again in a few minutes")
                self._sleep(_retry_after(r, default=5 * 2 ** rl_try))
                continue
            break
        if r.status_code in (401, 403):
            raise GmailError(f"HTTP {r.status_code}: Gmail access denied"
                             + (f" ({reason})" if reason else "")
                             + " (re-run `cortex gmail-auth`; is the Gmail API enabled in your Cortex Cloud project?)")
        if r.status_code != 200:
            raise GmailError(f"HTTP {r.status_code}")
        return r.json()

    def _send(self, url: str, params: dict) -> httpx.Response:
        for attempt in range(3):          # Gmail is occasionally slow on one message: retry, don't give up
            try:
                return self._c.get(url, params=params, headers={"Authorization": f"Bearer {self._token}"},
                                   follow_redirects=False)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt == 2:
                    raise GmailError(type(e).__name__) from None
                self._sleep(2 * (attempt + 1))
            except httpx.HTTPError as e:
                raise GmailError(type(e).__name__) from None
        raise AssertionError("unreachable")

    def recent_ids(self, days: int, limit: int = 200) -> list[str]:
        q = f"newer_than:{int(days)}d -category:promotions -category:social -category:forums"
        data = self._get(f"{API}/messages", q=q, maxResults=min(limit, 500))
        return [m["id"] for m in (data.get("messages") or []) if isinstance(m, dict) and _ID.match(str(m.get("id", "")))]

    def message(self, mid: str) -> Mail:
        if not _ID.match(mid):
            raise GmailError("invalid message id")
        m = self._get(f"{API}/messages/{mid}", format="full")
        return Mail(mid, _header(m, "from"), sanitize_jd(_header(m, "subject"))[0][:300], _body_text(m.get("payload")))


# ---------- the step ----------

@dataclass
class InboxResult:
    scanned: int = 0
    skipped: int = 0         # messages that kept failing to download: not recorded, so tried again next run
    applied: list[tuple[int, str, str, str]] = field(default_factory=list)    # job id, verdict, company, title
    to_check: list[tuple[str, str, str]] = field(default_factory=list)        # verdict, company(s), why


def scan(session, cfg: dict, user_id: str, client: GmailClient, *, dry_run: bool = False,
         audit: AuditLog | None = None) -> InboxResult:
    gc = cfg.get("gmail") or {}
    seen = set(session.scalars(select(InboxFindingRow.message_id).where(InboxFindingRow.user_id == user_id)))
    active = list(session.scalars(select(JobRow).where(JobRow.user_id == user_id, JobRow.status.in_(ACTIVE),
                                                       JobRow.dup_of_id.is_(None))))
    res = InboxResult()
    stop: GmailError | None = None
    for mid in client.recent_ids(int(gc.get("lookback_days", 3))):
        if mid in seen:
            continue
        try:
            mail = client.message(mid)
        except GmailError as e:
            if "access denied" in str(e) or "rate limit" in str(e):
                stop = e                       # auth / quota problems stop the run (every message would fail);
                break                          # what was read so far is kept, the rest is read next run
            res.skipped += 1
            continue
        res.scanned += 1
        f = decide(mail, active)
        row = InboxFindingRow(user_id=user_id, message_id=mid, found_at=datetime.now(UTC))
        if f is not None:
            row.verdict, row.job_ids, row.auto, row.subject = f.verdict, json.dumps(f.job_ids), f.auto, f.subject[:120]
            jobs = [j for j in active if j.id in f.job_ids]
            if f.auto:
                job = jobs[0]
                if not dry_run:
                    try:
                        update_status(session, user_id, job.id, f.verdict, audit=audit,
                                      note=f"auto from Gmail ({datetime.now(UTC).date().isoformat()})")
                        row.applied = True
                        res.applied.append((job.id, f.verdict, job.company, job.title))
                    except TrackerError:
                        row.applied = False      # already further along: nothing to do
                else:
                    res.applied.append((job.id, f.verdict, job.company, job.title))
            else:
                res.to_check.append((f.verdict, ", ".join(sorted({j.company for j in jobs})), f.why))
        if not dry_run:
            session.add(row)
    if dry_run:
        session.rollback()
    else:
        session.commit()
    if audit:
        audit.log("gmail_inbox", "scan", user_id=user_id, scanned=res.scanned, skipped=res.skipped,
                  applied=len(res.applied),
                  to_check=len(res.to_check), dry_run=dry_run, stopped=bool(stop))
    if stop is not None:
        raise stop
    return res


def client_from_config(cfg: dict) -> GmailClient:
    gc = cfg.get("gmail") or {}
    if not gc.get("enabled"):
        raise GmailError("Gmail watcher is off: set gmail.enabled: true in config.local.yaml")
    return GmailClient(load_token(gc.get("token_file", "gmail-token.json")))
