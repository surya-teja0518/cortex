import base64
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from cortex.db.schema import JobRow, ReplyWaitRow, get_session
from cortex.notify.digest import build_digest, render, subject
from cortex.sources import reply_waits as RW
from cortex.sources.gmail_inbox import GmailClient, scan

TOKEN = "ya29.GMAIL-SECRET"
CFG = {"gmail": {"enabled": True, "lookback_days": 3}}

# shapes of the real emails of 2026-11-08..20 (rewritten, no real addresses)
NCR = ("Prachi Saxena <prachi.saxena@ncr-example.com>", "Exciting Opportunity | Senior Oracle/PostgreSQL DBA",
       ("Dear Surya, your profile caught my attention. Request you to kindly share the details below for further "
        "process. Name Email ID Contact No Exp CTC E CTC Company Notice Period Interview Availability"))
TEST = ("admin@hirepro.in", "Accordion : Online Test Credentials",
        "Thank you for pursuing a career with Accordion. As the next step, we request you to complete the following online assessment. Test Link : [link]")
PORTAL = ("Zensar Technologies Limited <zenorc@oracle.com>", "Reminder: Additional information required",
          ("We are awaiting additional information required for your job application. Please provide the "
           "requested details using this link at your earliest convenience."))
REJECT = ("cisco@myworkday.com", "Thanks for Considering Cisco",
          "regret to inform you that we won't be moving forward with you. Please let us know if you have questions.")
NAUKRI = ("recruiter123@naukri.com", "Job | Data Engineer in Hyderabad", "Are you interested? Kindly share your CV")
NEWSLETTER = ("team@courses.example", "Your invitation is still waiting",
              "Are you interested in our bootcamp? Please share with friends. Unsubscribe")
PORTAL_FYI = ("noreply@workday.example", "Application update",
              "You moved to the next step. No action is required. Please complete your profile if you wish.")


@pytest.mark.parametrize("mail,kind,asks", [
    (NCR, "reply", {"CTC", "notice period", "availability", "details"}),
    (TEST, "action", {"online test"}),
    (PORTAL, "action", {"portal form"}),
    (REJECT, None, None),       # a rejection is not waiting on him
    (NAUKRI, None, None),       # job-board blast
    (NEWSLETTER, None, None),   # marketing
    (PORTAL_FYI, None, None),   # says no action needed
])
def test_needs_reply(mail, kind, asks):
    a = RW.needs_reply(*mail)
    if kind is None:
        assert a is None
    else:
        assert a.kind == kind and asks <= set(a.asks)


def test_who_never_keeps_the_address():
    assert RW.who_from("Prachi Saxena <prachi.saxena@ncr-example.com>") == "Prachi Saxena"
    assert RW.who_from("admin@hirepro.in") == "hirepro.in"
    assert RW.who_from('"x@y.com" <x@y.com>') == "y.com"
    assert "\n" not in RW.who_from("Evil\nBcc: a@b.c <e@x.com>")


# ---------- through the inbox scan, with a fake Gmail ----------

def b64(s):
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def msg(mid, mail, when, thread="t0000001", labels=("INBOX",)):
    sender, subj, body = mail
    return {"id": mid, "threadId": thread, "internalDate": str(int(when.timestamp() * 1000)), "labelIds": list(labels),
            "payload": {"headers": [{"name": "From", "value": sender}, {"name": "Subject", "value": subj}],
                        "mimeType": "text/plain", "body": {"data": b64(body)}}}


class FakeGmail:
    def __init__(self, messages):
        self.messages = {m["id"]: m for m in messages}
        self.sent: list[dict] = []          # his replies: {"thread": id, "to": addr, "at": datetime}
        self.queries = []

    def handler(self, req: httpx.Request):
        path = req.url.path
        if path.endswith("/messages"):
            q = req.url.params.get("q", "")
            self.queries.append(q)
            if q.startswith("in:sent to:"):
                to, after = q.split("to:")[1].split(" after:")
                hit = [s for s in self.sent if s.get("to") == to and s["at"].timestamp() > int(after)]
                return httpx.Response(200, json={"messages": [{"id": "s0000001"}] if hit else []})
            return httpx.Response(200, json={"messages": [{"id": i} for i in self.messages]})
        if "/threads/" in path:
            tid = path.rsplit("/", 1)[-1]
            ms = [{"id": m["id"], "labelIds": m["labelIds"], "internalDate": m["internalDate"]}
                  for m in self.messages.values() if m["threadId"] == tid]
            ms += [{"id": "s", "labelIds": ["SENT"], "internalDate": str(int(s["at"].timestamp() * 1000))}
                   for s in self.sent if s.get("thread") == tid]
            return httpx.Response(200, json={"messages": ms})
        return httpx.Response(200, json=self.messages[path.rsplit("/", 1)[-1]])

    def client(self):
        return GmailClient(TOKEN, client=httpx.Client(transport=httpx.MockTransport(self.handler)), sleep=lambda s: None)


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


ASKED = datetime.now(UTC) - timedelta(days=4)


def test_scan_adds_reminders_nags_until_he_replies(ses):
    g = FakeGmail([msg("m0000001", NCR, ASKED, thread="t0000001"), msg("m0000002", TEST, ASKED, thread="t0000002"),
                   msg("m0000003", REJECT, ASKED, thread="t0000003")])
    r = scan(ses, CFG, "u1", g.client())
    assert [w[0] for w in r.waiting_new] == ["Prachi Saxena", "hirepro.in"] and r.waiting_open == 2
    rows = RW.open_waits(ses, "u1")
    assert rows[0].asked_at.date() == ASKED.date() and "notice period" in rows[0].asks
    assert all(TOKEN not in q for q in g.queries)
    # next day: nothing new, he still hasn't replied -> still open (the digest keeps nagging)
    assert scan(ses, CFG, "u1", g.client()).waiting_open == 2
    # he replies in the thread -> closed automatically; the test stays until he marks it done
    g.sent.append({"thread": "t0000001", "at": datetime.now(UTC)})
    r = scan(ses, CFG, "u1", g.client())
    assert r.replied == 1 and r.waiting_open == 1
    w = RW.open_waits(ses, "u1")[0]
    RW.close(ses, "u1", w.id)
    assert RW.open_waits(ses, "u1") == []
    with pytest.raises(LookupError):
        RW.close(ses, "someone-else", w.id)


def test_reply_as_a_new_email_to_the_sender_also_counts(ses):
    g = FakeGmail([msg("m0000001", NCR, ASKED)])
    scan(ses, CFG, "u1", g.client())
    g.sent.append({"to": "prachi.saxena@ncr-example.com", "at": datetime.now(UTC)})      # not in the thread
    assert scan(ses, CFG, "u1", g.client()).replied == 1
    assert ses.query(ReplyWaitRow).one().closed_how == "replied"


def test_old_reply_before_the_ask_does_not_close_it(ses):
    g = FakeGmail([msg("m0000001", NCR, ASKED)])
    g.sent.append({"thread": "t0000001", "at": ASKED - timedelta(days=1)})
    assert scan(ses, CFG, "u1", g.client()).waiting_open == 1


def test_his_own_mail_and_the_digest_never_become_reminders(ses):
    digest = ("me@example.com", "⚠ 1 waiting on your reply · CORTEX", "asks for: CTC, notice period, availability")
    g = FakeGmail([msg("m0000001", digest, ASKED, labels=("INBOX", "SENT"))])
    assert scan(ses, CFG, "u1", g.client()).waiting_new == []


def test_old_reminders_expire_and_dry_run_records_nothing(ses):
    g = FakeGmail([msg("m0000001", NCR, ASKED)])
    r = scan(ses, CFG, "u1", g.client(), dry_run=True)
    assert r.waiting_new and ses.query(ReplyWaitRow).count() == 0
    old = FakeGmail([msg("m0000002", NCR, datetime.now(UTC) - timedelta(days=30), thread="t0000009")])
    scan(ses, CFG, "u1", old.client())
    assert ses.query(ReplyWaitRow).one().closed_how == "expired"


def test_catch_up_finds_asks_in_mail_already_read(ses):
    from cortex.db.schema import InboxFindingRow
    ses.add(InboxFindingRow(user_id="u1", message_id="m0000001"))      # read before reminders existed
    ses.commit()
    g = FakeGmail([msg("m0000001", NCR, ASKED)])
    assert scan(ses, CFG, "u1", g.client(), asks_days=0).waiting_new == []
    assert len(scan(ses, CFG, "u1", g.client()).waiting_new) == 1        # default: 14-day catch-up every run
    assert any("notice period" in q for q in g.queries)


def test_reminders_can_be_switched_off(ses):
    g = FakeGmail([msg("m0000001", NCR, ASKED)])
    r = scan(ses, {"gmail": {"reply_reminders": False}}, "u1", g.client())
    assert r.waiting_new == [] and ses.query(ReplyWaitRow).count() == 0


# ---------- digest ----------

def test_digest_puts_waiting_first_and_in_the_subject(ses):
    ses.add(JobRow(user_id="u1", source="t", external_id="1", company="Acme", title="DE", url="u",
                   status="to_apply", score=70, band="P2", score_reasons="[]"))
    ses.add(ReplyWaitRow(user_id="u1", message_id="m0000001", who="Prachi Saxena",
                         subject="Senior DBA https://evil.example/x", kind="reply", asks="CTC, notice period",
                         asked_at=(datetime.now(UTC) - timedelta(days=4)).replace(tzinfo=None)))
    ses.commit()
    d = build_digest(ses, "u1", datetime.now(UTC).date())
    text = render(d)
    assert subject(d).startswith("⚠ 1 waiting on your reply")
    assert text.index("WAITING ON YOU") < text.index("NEW JOBS")
    assert "4d  Prachi Saxena | Senior DBA" in text and "evil.example" not in text
    assert "cortex reply-done" in text
    d.new_total, d.followups, d.pending, d.ready = 0, [], [], []
    assert not d.empty                       # a waiting recruiter alone is reason enough to send the digest


# ---------- regression set from his real inbox (2026-11-24), paraphrased, no real addresses ----------

CARD = ("credit_cards@bank.example", "Credit Card Statement",
        ("Payment due by November 20. The time period between the payment due date and the billing date is "
        "considered as the notice period for reporting a Card Member as a defaulter. Please notify the sender. "
        "This is a system generated e-mail. Please do not reply."))
EY_APPLY = ("Gokul V <gokul.v@gds.ey-example.com>", "Application Link - Data Engineer",
            ("It was nice speaking with you. As discussed, I'm sharing the application link for the Data Engineer "
            "role. Please apply using the link and let me know once you have completed the application."))
GLIDER = ("EY GDS <assistant@glider.ai>", "EY GDS invitation for Senior Data Integration Consultant",
          ("Thank you for your interest in opportunities with EY GDS. If you agree, click here to continue with your "
          "candidature. You will be asked to complete a pre-screening assessment. The link is valid until Thursday."))
ARCHER = ("Archer <archer@mail-hackajob.com>", "Accenture and 21 others want you to apply",
          "I found a role at Accenture. Just hit reply. Are you interested? Unsubscribe")
SHOP = ("return@shop.example", "Your return", ("Prices and availability are subject to change. This email was sent "
        "from a notification-only address that cannot accept incoming email. Please do not reply."))


@pytest.mark.parametrize("mail,kind", [(CARD, None), (EY_APPLY, "reply"), (GLIDER, "action"), (ARCHER, None),
                                       (SHOP, None)])
def test_real_inbox_regressions(mail, kind):
    a = RW.needs_reply(*mail)
    assert (a.kind if a else None) == kind
