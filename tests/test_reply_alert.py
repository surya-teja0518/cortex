"""Reply-needed alerts (2026-11-24): emails waiting on Surya's reply are detected in the Gmail scan, stored with the
sender DOMAIN + subject only, emailed to his own address once, listed in the digest and the dashboard."""
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from sqlalchemy import select

from cortex.db.schema import InboxFindingRow, JobRow, ReplyAlertRow, get_session
from cortex.notify.reply_alert import mark_done, open_replies, render, send_new_alerts
from cortex.sources.gmail_inbox import Mail, needs_reply, scan, sender_domain
from tests.test_gmail_inbox import CFG, FakeGmail, gmail_message


def mail(subject, body, sender="Recruiter <priya@acme.com>"):
    return Mail("m0000001", sender, subject, body)


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/r.db")


# ---------- detection (shapes of the real emails found on 2026-11-24) ----------

@pytest.mark.parametrize("subject,body,sender", [
    ("Interview Link - EY GDS", "Please find the interview link below and join on time.", "EY <talent@gds.ey.com>"),
    ("Accordion : Online Test Credentials", "Kindly complete the online assessment before Friday.", "HirePro <no-reply@hirepro.in>"),
    ("Accordion Recruitment Process - details required", "Please share your current CTC and notice period.",
     "Accordion HR <hr@accordion.com>"),
    ("Exciting Opportunity | Senior Oracle/PostgreSQL DBA", "I came across your profile. Are you interested?",
     "Recruiter <ta@ncratleos.com>"),
    ("Job Opportunity With Tier 1 MNC", "Please share your updated resume and expected CTC.", "Naukri <info@naukri.com>"),
    ("Quick call?", "Could you share your availability for a call this week?", "Hiring <x@startup.io>"),
])
def test_emails_waiting_on_a_reply_are_caught(subject, body, sender):
    assert needs_reply(mail(subject, body, sender)) is not None


@pytest.mark.parametrize("subject,body,sender", [
    ("Your application", "We regret to inform you... kindly note this decision is final.", "HR <hr@acme.com>"),
    ("New jobs for you", "Are you interested in these 20 jobs?", "LinkedIn <jobalerts-noreply@linkedin.com>"),
    ("Weekly digest", "Share your updated resume to get more views", "Naukri <newsletter@naukri.com>"),
    ("Team lunch", "see you at 1", "Friend <a@b.com>"),
])
def test_rejections_alerts_and_chatter_are_not(subject, body, sender):
    assert needs_reply(mail(subject, body, sender)) is None


def test_sender_domain_never_keeps_the_address():
    assert sender_domain("Priya Sharma <priya.s@talent.acme.co.in>") == "talent.acme.co.in"
    assert sender_domain("bare@example.com") == "example.com"
    assert sender_domain("no address here") == "unknown"


# ---------- scan stores domain + subject only, once ----------

def test_scan_records_reply_alerts_without_body_or_address(ses):
    f = FakeGmail([gmail_message("m0000001", "Interview Link - EY GDS", "Please find the interview link. SECRET BODY",
                                 sender="Priya <priya@gds.ey.com>"),
                   gmail_message("m0000002", "Lunch", "see you", sender="Friend <a@b.com>")])
    r = scan(ses, CFG, "u1", f.client())
    assert [(d, s) for d, s, _ in r.reply_needed] == [("gds.ey.com", "Interview Link - EY GDS")]
    rows = ses.scalars(select(ReplyAlertRow)).all()
    assert len(rows) == 1 and rows[0].sender_domain == "gds.ey.com" and rows[0].alerted_at is None
    stored = " ".join(str(getattr(rows[0], c.name)) for c in ReplyAlertRow.__table__.columns)
    assert "priya@" not in stored and "SECRET BODY" not in stored
    assert ses.query(InboxFindingRow).count() == 2
    assert scan(ses, CFG, "u1", f.client()).reply_needed == []                 # never read twice


def test_reply_links_the_company_when_it_matches_an_application(ses):
    job = JobRow(user_id="u1", source="linkedin", external_id="1", company="Accordion", title="Data Engineer",
                 url="u", status="applied")
    ses.add(job)
    ses.commit()
    f = FakeGmail([gmail_message("m0000001", "Accordion Recruitment Process - details required",
                                 "Please share your current CTC for the Data Engineer role.",
                                 sender="Accordion HR <hr@accordion.com>")])
    scan(ses, CFG, "u1", f.client())
    assert ses.scalar(select(ReplyAlertRow.company)) == "Accordion"


def test_dry_run_records_no_alerts(ses):
    f = FakeGmail([gmail_message("m0000001", "Interview Link", "Please find the interview link", sender="a@ey.com")])
    assert scan(ses, CFG, "u1", f.client(), dry_run=True).reply_needed
    assert ses.query(ReplyAlertRow).count() == 0


# ---------- alert email: once, own address, no links ----------

class FakeSMTP:
    sent: ClassVar[list] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, a, p):
        pass

    def send_message(self, msg, from_addr, to_addrs):
        FakeSMTP.sent.append((msg["Subject"], msg.get_content(), to_addrs))


def _alert(ses, n=1, user="u1", **kw):
    for i in range(n):
        ses.add(ReplyAlertRow(user_id=user, message_id=f"m{user}{i:07d}", sender_domain="gds.ey.com",
                              subject=f"Interview Link {i}", cue="assessment / interview link", **kw))
    ses.commit()


def test_alert_sent_once_to_own_address_without_links(ses):
    FakeSMTP.sent = []
    _alert(ses, 2)
    n = send_new_alerts(ses, "u1", address="me@example.com", password="pw", smtp_factory=FakeSMTP)
    assert n == 2 and len(FakeSMTP.sent) == 1
    subject, body, to = FakeSMTP.sent[0]
    assert "2 emails waiting for your reply" in subject and to == ["me@example.com"]
    assert "Interview Link 0" in body and "http" not in body and "@" not in body.replace("me@example.com", "")
    assert send_new_alerts(ses, "u1", address="me@example.com", password="pw", smtp_factory=FakeSMTP) == 0
    assert len(FakeSMTP.sent) == 1                                                # never twice


def test_digest_already_covering_them_sends_no_second_email(ses):
    FakeSMTP.sent = []
    _alert(ses, 1)
    assert send_new_alerts(ses, "u1", address="me@example.com", password="pw", smtp_factory=FakeSMTP,
                           covered_by_digest=True) == 1
    assert FakeSMTP.sent == [] and ses.scalar(select(ReplyAlertRow.alerted_at)) is not None


def test_done_clears_and_is_per_user(ses):
    _alert(ses, 1)
    _alert(ses, 1, user="other")
    mine = ses.scalar(select(ReplyAlertRow.id).where(ReplyAlertRow.user_id == "u1"))
    theirs = ses.scalar(select(ReplyAlertRow.id).where(ReplyAlertRow.user_id == "other"))
    assert not mark_done(ses, "u1", theirs) and mark_done(ses, "u1", mine) and not mark_done(ses, "u1", mine)
    assert open_replies(ses, "u1") == [] and len(open_replies(ses, "other")) == 1


def test_render_shows_age():
    r = ReplyAlertRow(user_id="u1", message_id="m1", sender_domain="acme.com", subject="Call?", cue="asks a convenient time",
                      found_at=datetime(2026, 11, 24, 6, 0))
    _, body = render([r], datetime(2026, 11, 24, 9, 0, tzinfo=UTC))
    assert "acme.com | Call?" in body and "found 3 h ago" in body


def test_digest_lists_open_replies_first(ses):
    from cortex.notify.digest import build_digest, subject
    from cortex.notify.digest import render as render_digest
    _alert(ses, 1)
    d = build_digest(ses, "u1", date(2026, 11, 24))
    text = render_digest(d)
    assert text.splitlines()[2].startswith("WAITING FOR YOUR REPLY: 1") and "Interview Link 0" in text
    assert "1 WAITING FOR YOUR REPLY" in subject(d)


# ---------- dashboard ----------

def test_dashboard_lists_replies_and_done_needs_csrf(tmp_path):
    from fastapi.testclient import TestClient

    from cortex.core.audit import AuditLog
    from cortex.web import auth
    from cortex.web.app import create_app
    from tests.test_web import BASE, OWNER, settings
    url = f"sqlite:///{tmp_path}/w.db"
    ses = get_session(url)
    _alert(ses, 1)
    s = settings(db_url=url)
    c = TestClient(create_app(s, session_factory=lambda: get_session(url),
                              audit_factory=lambda: AuditLog(str(tmp_path / "a.log"))), base_url=BASE)
    c.cookies.set(auth.SESSION_COOKIE, auth.new_session(s, OWNER))
    html = c.get("/").text
    assert "Waiting for your reply" in html and "Interview Link 0" in html
    rid = ses.scalar(select(ReplyAlertRow.id))
    assert c.post(f"/replies/{rid}/done").status_code == 403
    token = auth.read_session(s, c.cookies.get(auth.SESSION_COOKIE))["csrf"]
    assert c.post(f"/replies/{rid}/done", headers={"X-CSRF-Token": token}).text == ""
    assert "Waiting for your reply" not in c.get("/").text


# ---------- workflows ----------

def test_only_the_reply_check_workflow_reads_gmail():
    root = Path(__file__).parent.parent
    daily = (root / ".github/workflows/daily.yml").read_text()
    assert "cortex daily --no-inbox" in daily
    rc_text = (root / ".github/workflows/reply-check.yml").read_text()
    rc = yaml.safe_load(rc_text)
    assert set(rc.get("on", rc.get(True))) == {"schedule", "workflow_dispatch"}
    assert rc["permissions"] == {"contents": "read"} and rc["jobs"]["reply-check"]["env"]["CORTEX_PUBLIC_LOG"] == "1"
    assert "upload-artifact" not in rc_text and "pull_request" not in rc_text and "cat " not in rc_text
    for secret in ("GROQ", "APIFY", "SKILLS", "SERVICE_ACCOUNT"):                   # least privilege
        assert secret not in rc_text


def test_cli_reply_check_dry_run(tmp_path, monkeypatch, capsys):
    from cortex import cli
    monkeypatch.setenv("CORTEX_DB_URL", f"sqlite:///{tmp_path}/c.db")
    monkeypatch.setenv("CORTEX_USER_ID", "u1")
    f = FakeGmail([gmail_message("m0000001", "Interview Link - EY", "Please find the interview link", sender="a@ey.com")])
    monkeypatch.setattr("cortex.sources.gmail_inbox.client_from_config", lambda cfg: f.client())
    cli.main(["reply-check", "--dry-run"])
    out = capsys.readouterr().out
    assert "new needing a reply: 1" in out and "would alert: 0" in out and "ey.com | Interview Link - EY" in out
    assert get_session(f"sqlite:///{tmp_path}/c.db").query(ReplyAlertRow).count() == 0


# ---------- tuning on the real 14-day inbox (2026-11-24): 7 flagged, all real, 0 noise ----------

@pytest.mark.parametrize("subject,body,sender", [
    ("EY GDS - Interview Scheduled | Senior Data Integration Consultant | Round 1",
     "Your interview has been scheduled.", "EY GDS <noreply@glider.ai>"),          # the one that was missed
    ("Accordion Recruitment Process - details required",
     "Please fill in the form with the following details.", "Recruiter <ta@accordion.com>"),
])
def test_previously_missed_emails_are_caught(subject, body, sender):
    assert needs_reply(mail(subject, body, sender)) is not None


def test_bulk_mail_with_list_unsubscribe_is_ignored():
    m = Mail("m1", "Crio <hello@crio.in>", "NVIDIA has a job opportunity", "Are you interested? Take the test now!",
             bulk=True)
    assert needs_reply(m) is None


def test_job_board_relays_need_a_specific_ask():
    vague = mail("Job | Production Engineer", "We came across your profile. Are you interested?",
                 "foundit <jobs@foundit.in>")
    assert needs_reply(vague) is None
    specific = mail("Job | AWS Data Engineer", "Please share your updated resume and current CTC.",
                    "Naukri <recruiter@naukri.com>")
    assert needs_reply(specific) is not None
    direct = mail("Opportunity", "I came across your profile and wanted to connect.", "Ravi <ravi@ncratleos.com>")
    assert needs_reply(direct) == "recruiter reaching out"                    # a person, not a job board


def test_list_unsubscribe_header_marks_mail_as_bulk():
    from tests.test_gmail_inbox import b64
    msg = {"id": "m0000009", "payload": {"headers": [
        {"name": "From", "value": "Crio <hello@crio.in>"}, {"name": "Subject", "value": "Hiring!"},
        {"name": "List-Unsubscribe", "value": "<mailto:u@crio.in>"}],
        "mimeType": "text/plain", "body": {"data": b64("Are you interested?")}}}
    f = FakeGmail([msg])
    assert f.client().message("m0000009").bulk is True
    assert FakeGmail([gmail_message("m0000001", "Hi", "x")]).client().message("m0000001").bulk is False
