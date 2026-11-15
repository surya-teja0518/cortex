import base64
from datetime import date

import httpx
import pytest
from sqlalchemy import select

from cortex.core.audit import AuditLog
from cortex.db.schema import ApplicationEventRow, InboxFindingRow, JobRow, get_session
from cortex.sources.gmail_inbox import (
    GmailClient,
    GmailError,
    Mail,
    classify,
    decide,
    scan,
)

TOKEN = "ya29.GMAIL-SECRET"
CFG = {"gmail": {"enabled": True, "lookback_days": 3}}


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


def job(ses, n, company, status="applied", title="Data Engineer"):
    j = JobRow(user_id="u1", source="linkedin", external_id=str(n), company=company, title=title, url="u",
               status=status)
    ses.add(j)
    ses.commit()
    return j


def mail(subject, body, sender="Talent Team <careers@acme.com>", mid="m0000001"):
    return Mail(mid, sender, subject, body)


# ---------- classification ----------

@pytest.mark.parametrize("text,want", [
    ("We have decided not to move forward with your application", ("rejected", True)),
    ("we will not be moving forward at this time", ("rejected", True)),
    ("After careful review, we regret to inform you", ("rejected", True)),
    ("Unfortunately the role is on hold", ("rejected", False)),
    ("We'd like to schedule a call with you next week", ("interview", True)),
    ("Please complete the HackerRank assessment", ("interview", True)),
    ("Unfortunately we can't do Tuesday, could we schedule an interview Wednesday?", (None, False)),   # mixed
    ("Your weekly newsletter", (None, False)),
])
def test_classify(text, want):
    assert classify(text) == want


# ---------- decide: auto only when unambiguous ----------

def test_clear_rejection_is_auto(ses):
    acme = job(ses, 1, "Acme Technologies Pvt Ltd")
    f = decide(mail("Your application to Acme Technologies", "we will not be moving forward"), [acme])
    assert f.auto and f.verdict == "rejected" and f.job_ids == [acme.id]


def test_ambiguous_cases_are_suggestions(ses):
    wf1, wf2 = job(ses, 1, "Wells Fargo"), job(ses, 2, "Wells Fargo", title="Data Engineer II")
    beta = job(ses, 3, "Beta Corp")
    f = decide(mail("Wells Fargo update", "not moving forward", sender="wf@wellsfargo.com"), [wf1, wf2])
    assert not f.auto and "2 active applications" in f.why
    f = decide(mail("Update", "Unfortunately the role is filled", sender="x@betacorp.com"), [beta])
    assert not f.auto and "weak" in f.why
    f = decide(mail("Update from Beta Corp and Wells Fargo", "not moving forward"), [wf1, beta])
    assert not f.auto and "more than one company" in f.why
    f = decide(mail("Update on your application", "not moving forward.\n-- sent via Beta Corp mail", sender="hr@x.com"),
               [beta])
    assert not f.auto and "only mentioned in the body" in f.why


def test_irrelevant_mail_and_job_alerts_ignored(ses):
    acme = job(ses, 1, "Acme")
    assert decide(mail("Acme team lunch", "see you there"), [acme]) is None                 # no signal
    assert decide(mail("Rejection", "not moving forward"), []) is None                      # no application named
    assert decide(mail("Acme is hiring", "not moving forward",
                       sender="LinkedIn <jobalerts-noreply@linkedin.com>"), [acme]) is None  # job alert


# ---------- Gmail API client (read only) ----------

def b64(s):
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def gmail_message(mid, subject, body, sender="Acme Careers <careers@acme.com>", html=False):
    return {"id": mid, "payload": {
        "headers": [{"name": "From", "value": sender}, {"name": "Subject", "value": subject}],
        "mimeType": "multipart/alternative",
        "parts": [{"mimeType": "text/html" if html else "text/plain", "filename": "", "body": {"data": b64(body)}},
                  {"mimeType": "application/pdf", "filename": "offer.pdf", "body": {"data": b64("IGNORE ME")}}]}}


class FakeGmail:
    def __init__(self, messages):
        self.messages, self.calls = {m["id"]: m for m in messages}, []

    def handler(self, req: httpx.Request):
        self.calls.append(req)
        if req.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": i} for i in self.messages]})
        mid = req.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=self.messages[mid])

    def client(self):
        return GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(self.handler)))


def test_client_reads_text_ignores_attachments_and_keeps_token_in_header():
    f = FakeGmail([gmail_message("m0000001", "Hi", "<p>We will <b>not be moving forward</b></p>", html=True)])
    c = f.client()
    m = c.message("m0000001")
    assert "not be moving forward" in m.text and "IGNORE ME" not in m.text and "<b>" not in m.text
    req = f.calls[0]
    assert req.headers["Authorization"] == f"Bearer {TOKEN}" and TOKEN not in str(req.url) and TOKEN not in repr(c)
    with pytest.raises(GmailError, match="invalid message id"):
        c.message("../../settings")


def test_denied_access_is_explained():
    c = GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403))))
    with pytest.raises(GmailError, match="Gmail access denied") as e:
        c.recent_ids(3)
    assert TOKEN not in str(e.value)


# ---------- scan: records, idempotent, private ----------

def test_scan_records_clear_ones_lists_the_rest_and_never_twice(ses, tmp_path):
    acme = job(ses, 1, "Acme")
    wf1, wf2 = job(ses, 2, "Wells Fargo"), job(ses, 3, "Wells Fargo", title="DE II")
    gamma = job(ses, 4, "Gamma Labs")
    f = FakeGmail([
        gmail_message("m0000001", "Your Acme application", "We will not be moving forward. Secret body text."),
        gmail_message("m0000002", "Wells Fargo", "not moving forward", sender="wf@wellsfargo.com"),
        gmail_message("m0000003", "Gamma Labs: next steps", "We'd like to schedule a call", sender="hr@gammalabs.io"),
        gmail_message("m0000004", "Newsletter", "hello"),
    ])
    a = AuditLog(str(tmp_path / "a.log"))
    r = scan(ses, CFG, "u1", f.client(), audit=a)
    assert r.scanned == 4
    assert sorted((jid, v) for jid, v, _, _ in r.applied) == [(acme.id, "rejected"), (gamma.id, "interview")]
    assert [(v, c) for v, c, _ in r.to_check] == [("rejected", "Wells Fargo")]
    ses.refresh(acme)
    ses.refresh(gamma)
    assert (acme.status, gamma.status, wf1.status, wf2.status) == ("rejected", "interview", "applied", "applied")
    ev = ses.scalars(select(ApplicationEventRow).where(ApplicationEventRow.job_id == acme.id)).all()
    assert ev[-1].note.startswith("auto from Gmail")
    rows = ses.scalars(select(InboxFindingRow)).all()
    assert len(rows) == 4 and all("Secret body" not in (x.subject or "") for x in rows)
    assert "Secret body" not in a.path.read_text() and "Acme" not in a.path.read_text()
    f.calls.clear()
    r2 = scan(ses, CFG, "u1", f.client())
    assert r2.scanned == 0 and r2.applied == [] and len(f.calls) == 1                         # list only


def test_scan_dry_run_records_nothing(ses):
    acme = job(ses, 1, "Acme")
    f = FakeGmail([gmail_message("m0000001", "Your Acme application", "not moving forward")])
    r = scan(ses, CFG, "u1", f.client(), dry_run=True)
    ses.refresh(acme)
    assert r.applied and acme.status == "applied" and ses.query(InboxFindingRow).count() == 0


def test_already_rejected_job_is_left_alone(ses):
    acme = job(ses, 1, "Acme", status="interview")
    f = FakeGmail([gmail_message("m0000001", "Acme", "Please complete the HackerRank test")])
    r = scan(ses, CFG, "u1", f.client())                                  # interview -> interview: another round
    assert [v for _, v, _, _ in r.applied] == ["interview"]
    acme.status = "offer"
    ses.commit()
    f2 = FakeGmail([gmail_message("m0000002", "Acme", "regret to inform you")])
    assert scan(ses, CFG, "u1", f2.client()).applied == []                # offer is final: no change, no crash


def test_daily_runs_inbox_before_sheet_sync(ses):
    from types import SimpleNamespace

    from cortex.daily import run_daily
    order = []
    inbox = lambda s: order.append("inbox") or SimpleNamespace(applied=[], to_check=[], scanned=0)
    sync = lambda s: order.append("sync") or SimpleNamespace(read=SimpleNamespace(inserted=0), written=0,
                                                              to_write=[])
    run_daily(ses, {}, "u1", date(2026, 11, 15), inbox=inbox, sync=sync, pull=None, fetch=None, score=None,
              digest=lambda s, n: "sent")
    assert order == ["inbox", "sync"]


def test_slow_message_is_retried_then_skipped_not_fatal(ses):
    acme = job(ses, 1, "Acme")
    good = gmail_message("m0000001", "Your Acme application", "not moving forward")
    calls = {"slow": 0}

    def handler(req):
        if req.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": "m0000001"}, {"id": "m0000002"}]})
        if req.url.path.endswith("m0000002"):
            calls["slow"] += 1
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=good)
    c = GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    r = scan(ses, CFG, "u1", c)
    assert calls["slow"] == 3 and r.skipped == 1 and r.scanned == 1                 # 3 tries, then skipped
    assert [jid for jid, *_ in r.applied] == [acme.id]                              # the rest still processed
    assert ses.query(InboxFindingRow).filter_by(message_id="m0000002").count() == 0  # retried next run


def test_auth_failure_still_stops_the_run(ses):
    job(ses, 1, "Acme")

    def handler(req):
        if req.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": "m0000001"}]})
        return httpx.Response(401)
    c = GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(GmailError, match="401"):
        scan(ses, CFG, "u1", c)


def test_tech_word_company_only_matches_by_sender(ses):
    oracle = job(ses, 1, "Oracle", title="Database Engineer")
    zensar = job(ses, 2, "Zensar Technologies", title="Oracle DBA")
    m = mail("Your application for Oracle DBA", "we will not be moving forward", sender="Zensar Careers <hr@zensar.com>")
    f = decide(m, [oracle, zensar])
    assert f.auto and f.job_ids == [zensar.id]                                   # not Oracle
    only_oracle = decide(m, [oracle])
    assert only_oracle is None                                                  # Zensar email can't touch Oracle
    real = decide(mail("Your Oracle application", "regret to inform you", sender="Oracle Talent <talent@oracle.com>"),
                  [oracle, zensar])
    assert real.auto and real.job_ids == [oracle.id]


def _err(status, reason, headers=None):
    return httpx.Response(status, headers=headers or {}, json={"error": {"code": status, "message": "secret-ish detail",
                                                                         "errors": [{"reason": reason}]}})


def test_rate_limit_403_waits_and_retries_not_access_denied(ses):
    acme = job(ses, 1, "Acme")
    good = gmail_message("m0000001", "Your Acme application", "not moving forward")
    hits, slept = {"n": 0}, []

    def handler(req):
        if req.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": "m0000001"}]})
        hits["n"] += 1
        return _err(403, "userRateLimitExceeded", {"retry-after": "3"}) if hits["n"] < 3 else httpx.Response(200, json=good)
    c = GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=slept.append)
    r = scan(ses, CFG, "u1", c)
    assert [jid for jid, *_ in r.applied] == [acme.id] and slept == [3.0, 3.0]


def test_persistent_rate_limit_stops_but_keeps_what_was_read(ses):
    job(ses, 1, "Acme")
    good = gmail_message("m0000001", "Acme newsletter", "hello")

    def handler(req):
        if req.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": "m0000001"}, {"id": "m0000002"}]})
        return httpx.Response(200, json=good) if req.url.path.endswith("m0000001") else _err(429, "rateLimitExceeded")
    c = GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(GmailError, match="rate limit") as e:
        scan(ses, CFG, "u1", c)
    assert "secret-ish" not in str(e.value)
    assert ses.query(InboxFindingRow).filter_by(message_id="m0000001").count() == 1      # kept
    assert ses.query(InboxFindingRow).filter_by(message_id="m0000002").count() == 0      # read next run


def test_real_permission_403_names_googles_reason():
    c = GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(
        lambda r: _err(403, "insufficientPermissions"))), sleep=lambda s: None)
    with pytest.raises(GmailError, match=r"access denied \(insufficientPermissions\)"):
        c.recent_ids(3)
