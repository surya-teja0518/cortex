"""Hosted dashboard: Google sign-in for one address, CSRF + origin checks, headers, escaping, and that every
action goes through the same code paths as the CLI."""
import base64
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from cortex.core.audit import AuditLog
from cortex.db.schema import ApplicationEventRow, JobRow, SkipRow, get_session
from cortex.web import auth
from cortex.web.app import create_app, safe_url
from cortex.web.settings import Settings, SettingsError, load_settings

OWNER = "owner@example.com"
BASE = "https://cortex.test"


def settings(**kw) -> Settings:
    d = {"db_url": "sqlite://", "user_id": "u1", "allowed_email": OWNER, "client_id": "cid.apps.googleusercontent.com",
         "client_secret": "client-secret", "session_secret": "x" * 40, "public_url": BASE}
    return Settings(**{**d, **kw})


def id_token(**claims) -> str:
    base = {"iss": "https://accounts.google.com", "aud": "cid.apps.googleusercontent.com", "exp": time.time() + 600,
            "email": OWNER, "email_verified": True}
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{enc({'alg': 'RS256'})}.{enc({**base, **claims})}.sig"


class Google:
    """Fake token endpoint. `claims` go into the returned ID token; the nonce is copied from the flow."""
    def __init__(self):
        self.claims, self.calls, self.nonce = {}, [], None

    def handler(self, req: httpx.Request):
        self.calls.append(req)
        return httpx.Response(200, json={"id_token": id_token(nonce=self.nonce, **self.claims)})

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def env(tmp_path):
    url = f"sqlite:///{tmp_path}/w.db"
    ses = get_session(url)
    jobs = [JobRow(user_id="u1", source="linkedin", external_id=str(i), company=f"Co{i}", title="Data Engineer",
                   url=f"https://www.linkedin.com/jobs/view/{i}", status="to_apply", score=70, band="P2",
                   score_reasons="[]") for i in range(3)]
    jobs.append(JobRow(user_id="u1", source="linkedin", external_id="x", company="<script>alert(1)</script>",
                       title="Engineer <img src=x onerror=alert(2)>", url="javascript:alert(3)",
                       status="to_apply", score=70, band="P2", score_reasons="[]"))
    jobs.append(JobRow(user_id="other", source="linkedin", external_id="o", company="NotYours", title="DE",
                       url="u", status="to_apply"))
    ses.add_all(jobs)
    ses.commit()
    s = settings(db_url=url)
    g = Google()
    audit_path = str(tmp_path / "audit.log")
    app = create_app(s, session_factory=lambda: get_session(url), http_client=g.client(),
                     checkpointer=None, audit_factory=lambda: AuditLog(audit_path))
    return {"s": s, "app": app, "google": g, "url": url, "ids": [j.id for j in jobs], "audit": audit_path}


def client(env, signed_in=True) -> TestClient:
    c = TestClient(env["app"], base_url=BASE, follow_redirects=False)
    if signed_in:
        c.cookies.set(auth.SESSION_COOKIE, auth.new_session(env["s"], OWNER))
    return c


def csrf(c, env) -> dict:
    return {"X-CSRF-Token": auth.read_session(env["s"], c.cookies.get(auth.SESSION_COOKIE))["csrf"]}


# ---------- settings ----------

def test_settings_refuse_weak_or_missing_config():
    ok = {"CORTEX_DB_URL": "sqlite://", "CORTEX_ALLOWED_EMAIL": OWNER, "GOOGLE_OAUTH_CLIENT_ID": "a",
          "GOOGLE_OAUTH_CLIENT_SECRET": "b", "CORTEX_SESSION_SECRET": "y" * 40, "CORTEX_PUBLIC_URL": BASE}
    assert load_settings(ok).allowed_email == OWNER
    for bad, msg in [({"CORTEX_SESSION_SECRET": "short"}, "32"), ({"CORTEX_PUBLIC_URL": "http://cortex.test"}, "https"),
                     ({"GOOGLE_OAUTH_CLIENT_SECRET": ""}, "missing"), ({"CORTEX_ALLOWED_EMAIL": "nope"}, "email")]:
        with pytest.raises(SettingsError, match=msg):
            load_settings({**ok, **bad})
    assert load_settings({**ok, "CORTEX_PUBLIC_URL": "http://127.0.0.1:8000"}).secure is False   # local only


# ---------- sign-in ----------

def test_pages_need_a_session(env):
    c = client(env, signed_in=False)
    for path in ("/", "/approvals", "/applications", "/report"):
        r = c.get(path)
        assert r.status_code == 303 and r.headers["location"] == "/login"
    r = c.post("/jobs/1/skip", headers={"HX-Request": "true"})
    assert r.status_code == 401 and r.headers["HX-Redirect"] == "/login"
    assert c.get("/login").status_code == 200


def _start(c):
    r = c.get("/auth/google")
    q = parse_qs(urlsplit(r.headers["location"]).query)
    flow = r.cookies.get(auth.FLOW_COOKIE) or c.cookies.get(auth.FLOW_COOKIE)
    return q, flow


def test_google_sign_in_happy_path(env):
    c = client(env, signed_in=False)
    q, _ = _start(c)
    assert q["scope"] == ["openid email"] and q["code_challenge_method"] == ["S256"]
    assert q["redirect_uri"] == [f"{BASE}/auth/callback"] and q["login_hint"] == [OWNER]
    env["google"].nonce = q["nonce"][0]
    r = c.get("/auth/callback", params={"state": q["state"][0], "code": "abc"})
    assert r.status_code == 303 and r.headers["location"] == "/"
    sent = parse_qs(env["google"].calls[0].content.decode())
    assert sent["code_verifier"][0] and sent["client_secret"] == ["client-secret"]
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie.replace("Lax", "lax")
    assert c.get("/").status_code == 200


@pytest.mark.parametrize("claims,why", [
    ({"email": "someone@else.com"}, "not+allowed"),
    ({"email_verified": False}, "not+allowed"),
    ({"aud": "other-app"}, "not+issued"),
    ({"iss": "https://evil.example"}, "not+issued"),
    ({"exp": time.time() - 10}, "expired"),
])
def test_google_sign_in_refusals(env, claims, why):
    c = client(env, signed_in=False)
    q, _ = _start(c)
    env["google"].nonce = q["nonce"][0]
    env["google"].claims = claims
    r = c.get("/auth/callback", params={"state": q["state"][0], "code": "abc"})
    assert r.status_code == 303 and why in r.headers["location"] and auth.SESSION_COOKIE not in r.headers.get("set-cookie", "").split("=")[0]
    assert c.get("/").status_code == 303                                            # still signed out


def test_state_and_nonce_must_match(env):
    c = client(env, signed_in=False)
    q, _ = _start(c)
    env["google"].nonce = q["nonce"][0]
    assert "state+mismatch" in c.get("/auth/callback", params={"state": "forged", "code": "abc"}).headers["location"]
    q, _ = _start(c)
    env["google"].nonce = "replayed-nonce"
    assert "nonce+mismatch" in c.get("/auth/callback", params={"state": q["state"][0], "code": "abc"}).headers["location"]
    fresh = client(env, signed_in=False)                                              # no flow cookie at all
    assert "expired" in fresh.get("/auth/callback", params={"state": "s", "code": "c"}).headers["location"]


def test_forged_or_foreign_session_cookies_rejected(env):
    c = client(env, signed_in=False)
    c.cookies.set(auth.SESSION_COOKIE, "eyJlbWFpbCI6Im93bmVyQGV4YW1wbGUuY29tIn0.forged")
    assert c.get("/").status_code == 303
    other = settings(session_secret="z" * 40)                                        # signed with another secret
    c.cookies.set(auth.SESSION_COOKIE, auth.new_session(other, OWNER))
    assert c.get("/").status_code == 303
    c.cookies.set(auth.SESSION_COOKIE, auth.new_session(env["s"], "someone@else.com"))  # right key, wrong person
    assert c.get("/").status_code == 303


# ---------- CSRF / origin / headers ----------

def test_post_needs_csrf_token_and_same_origin(env):
    c = client(env)
    jid = env["ids"][0]
    assert c.post(f"/jobs/{jid}/skip").status_code == 403                             # no token
    assert c.post(f"/jobs/{jid}/skip", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert c.post(f"/jobs/{jid}/skip", headers={**csrf(c, env), "Origin": "https://evil.example"}).status_code == 403
    assert c.post(f"/jobs/{jid}/skip", headers={**csrf(c, env), "Origin": BASE}).status_code == 200
    assert get_session(env["url"]).scalar(select(SkipRow.job_id)) == jid


def test_security_headers_on_every_response(env):
    for c, path in ((client(env), "/"), (client(env, signed_in=False), "/login"), (client(env, signed_in=False), "/")):
        h = c.get(path).headers
        assert "script-src 'self'" in h["content-security-policy"] and "frame-ancestors 'none'" in h["content-security-policy"]
        assert "unsafe-inline" not in h["content-security-policy"]
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
        assert h["referrer-policy"] == "no-referrer" and h["cache-control"] == "no-store"
        assert h["strict-transport-security"].startswith("max-age=")
    assert client(env).get("/docs").status_code == 404                                # no API explorer


def test_job_text_is_escaped_and_only_http_links_render(env):
    html = client(env).get("/").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "<img src=x" not in html and "javascript:" not in html
    assert "onerror" not in html.replace("onerror=alert(2)&gt;", "") or "&lt;img" in html
    assert 'href="https://www.linkedin.com/jobs/view/0"' in html and 'rel="noopener noreferrer"' in html
    assert "NotYours" not in html                                                     # other users' jobs never show
    assert safe_url("javascript:alert(1)") is None and safe_url("https://x.test/j") == "https://x.test/j"
    assert safe_url("//evil.example") is None and safe_url("data:text/html,x") is None


def test_no_inline_script_or_handlers_in_templates():
    import re
    from pathlib import Path
    for f in (Path(__file__).parent.parent / "src/cortex/web/templates").glob("*.html"):
        t = f.read_text()
        assert not re.search(r"<script(?![^>]*\bsrc=)", t), f.name                     # only external scripts
        assert not re.search(r"\son[a-z]+=", t), f.name                                 # no onclick/onchange...
        assert not re.search(r"\|\s*safe\b", t), f.name                                                # never bypass autoescape


# ---------- actions use the CLI code paths ----------

def test_skip_applied_status_undo(env, tmp_path):
    c = client(env)
    h = csrf(c, env)
    a, b = env["ids"][0], env["ids"][1]
    assert c.post(f"/jobs/{a}/skip", headers=h).text == ""
    assert "Already skipped" in c.post(f"/jobs/{a}/skip", headers=h).text
    assert "Recorded: applied" in c.post(f"/jobs/{b}/applied", headers=h).text
    assert "is now shortlisted" in c.post(f"/jobs/{b}/status", headers=h, data={"status": "shortlisted"}).text
    assert "can&#39;t go from" in c.post(f"/jobs/{b}/status", headers=h, data={"status": "applied"}).text
    assert "Unknown status" in c.post(f"/jobs/{b}/status", headers=h, data={"status": "approved"}).text
    assert "back to applied" in c.post(f"/jobs/{b}/undo", headers=h).text
    ses = get_session(env["url"])
    assert ses.get(JobRow, b).status == "applied"
    assert [e.status for e in ses.scalars(select(ApplicationEventRow).where(ApplicationEventRow.job_id == b))] == \
        ["applied", "shortlisted", "undo"]
    other = env["ids"][-1]
    assert "not your job" in c.post(f"/jobs/{other}/skip", headers=h).text
    assert "not found" in c.post(f"/jobs/{other}/applied", headers=h).text
    assert AuditLog(env["audit"]).verify()


def test_approval_records_decision_and_waits_for_packet(env, tmp_path):
    from cortex.agents.track_tailor import TrackTailor
    from cortex.agents.truth_guard import TruthGuard
    from cortex.graph import Deps, build_graph, open_checkpointer, start_tailoring
    from tests.test_track_tailor import MASTER
    ckpt = str(tmp_path / "ck.db")
    ses = get_session(env["url"])
    jid = env["ids"][2]
    start = build_graph(Deps(lambda: ses, AuditLog(env["audit"]), TruthGuard(MASTER), TrackTailor(MASTER)),
                        open_checkpointer(ckpt))
    assert start_tailoring(start, ses, "u1", jid)["status"] == "pending_approval"
    app = create_app(env["s"], session_factory=lambda: get_session(env["url"]), checkpointer=open_checkpointer(ckpt),
                     audit_factory=lambda: AuditLog(env["audit"]))
    c = TestClient(app, base_url=BASE, follow_redirects=False)
    c.cookies.set(auth.SESSION_COOKIE, auth.new_session(env["s"], OWNER))
    assert "Co2" in c.get("/approvals").text
    h = csrf(c, env)
    assert "Unknown decision" in c.post(f"/approvals/{jid}", headers=h, data={"decision": "maybe"}).text
    r = c.post(f"/approvals/{jid}", headers=h, data={"decision": "approve", "note": "good"})
    assert f"cortex packet {jid}" in r.text
    assert get_session(env["url"]).get(JobRow, jid).status == "approved"
    assert "Not recorded" in c.post(f"/approvals/{jid}", headers=h, data={"decision": "approve"}).text  # once only


def test_pages_render(env):
    c = client(env)
    assert "Co0" in c.get("/").text and "Approvals" in c.get("/approvals").text
    assert "Applications" in c.get("/applications").text and "response rate" in c.get("/report").text
    assert c.get("/static/htmx.min.js").status_code == 200
    r = c.post("/logout", headers=csrf(c, env))
    assert r.status_code == 303 and 'cortex_session=""' in r.headers["set-cookie"]


def test_form_field_csrf_works_and_body_still_reaches_the_route(env):
    c = client(env)
    token = csrf(c, env)["X-CSRF-Token"]
    assert c.post("/logout", data={"csrf": "wrong"}).status_code == 403
    r = c.post("/logout", data={"csrf": token})
    assert r.status_code == 303 and r.headers["location"] == "/login"
    c2 = client(env)
    b = env["ids"][1]
    r = c2.post(f"/jobs/{b}/applied", headers=csrf(c2, env))
    r = c2.post(f"/jobs/{b}/status", data={"status": "interview", "note": "call on Monday",
                                            "csrf": csrf(c2, env)["X-CSRF-Token"]})
    assert "is now interview" in r.text                                       # form read twice: still intact
    ev = get_session(env["url"]).scalars(select(ApplicationEventRow).where(ApplicationEventRow.job_id == b)).all()
    assert ev[-1].note == "call on Monday"


def test_reply_reminders_on_dashboard_escaped_and_closable(env):
    from datetime import datetime

    from cortex.db.schema import ReplyWaitRow
    ses = get_session(env["url"])
    mine = ReplyWaitRow(user_id="u1", message_id="m0000001", who="Recruiter <b>X</b>", subject="Share CTC",
                        kind="reply", asks="CTC, notice period", asked_at=datetime(2026, 11, 20))
    theirs = ReplyWaitRow(user_id="other", message_id="m0000002", who="NotYoursEither", subject="s", kind="reply",
                          asked_at=datetime(2026, 11, 20))
    ses.add_all([mine, theirs])
    ses.commit()
    c = client(env)
    assert "1 email waiting on your reply" in c.get("/").text
    html = c.get("/applications").text
    assert "Waiting on your reply" in html and "&lt;b&gt;X&lt;/b&gt;" in html and "NotYoursEither" not in html
    h = csrf(c, env)
    assert c.post(f"/waiting/{mine.id}/done").status_code == 403                       # CSRF still required
    assert "No such reminder" in c.post(f"/waiting/{theirs.id}/done", headers=h).text  # other user's: refused
    assert "won&#39;t be listed" in c.post(f"/waiting/{mine.id}/done", headers=h).text
    assert "waiting on your reply" not in c.get("/").text
