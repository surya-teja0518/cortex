"""The hosted dashboard. `create_app(settings)` builds it; `api/index.py` (Vercel) and `cortex web` (local) call it.

Security (AGENTS.md, Phase 3):
- Every page needs a valid signed session for the ONE allowed Google address (auth.py); otherwise /login.
- Every POST needs the session's CSRF token (X-CSRF-Token header from HTMX, or a form field) AND, when the
  browser sends one, an Origin/Referer equal to CORTEX_PUBLIC_URL.
- Strict headers: CSP without inline script/style, no framing, no referrer, no caching of pages, HSTS on https.
- Jinja2 autoescape for all job text (untrusted); job links rendered only when http(s).
- Approvals run graph.submit_decision with packet=None: the decision is recorded and the thread waits at
  `packet`; the PDF is built later on a computer (`cortex packet <id>`), because resume files are never hosted.
"""
import hmac
from collections.abc import Callable
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cortex.web import auth
from cortex.web.settings import Settings

HERE = Path(__file__).parent
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
       "form-action 'self' https://accounts.google.com; frame-ancestors 'none'; base-uri 'none'")
STATUS_LABEL = {"applied": "Applied", "shortlisted": "Shortlisted", "interview": "Interview", "offer": "Offer",
                "rejected": "Rejected", "no_response": "No response"}


def safe_url(u: str | None) -> str | None:
    """Job links come from scraped postings: only plain http(s) links are ever rendered."""
    return u if u and urlsplit(u).scheme in ("http", "https") and urlsplit(u).netloc else None


def create_app(settings: Settings, *, session_factory: Callable | None = None, checkpointer=None,
               http_client=None, audit_factory: Callable | None = None) -> FastAPI:
    from cortex.agents import tracker
    from cortex.core.audit import AuditLog
    from cortex.core.config import load_config
    from cortex.db.schema import get_session
    from cortex.review import (
        applications,
        pending_approvals,
        ready_to_apply,
        record_skip,
        review_queue,
    )

    s = settings
    new_session = session_factory or (lambda: get_session(s.db_url))
    new_audit = audit_factory or (lambda: AuditLog(db_url=s.db_url))
    state = {"ckpt": checkpointer}
    async def post_guard(request: Request):
        """CSRF + origin for every POST. A dependency, not middleware: it shares FastAPI's parsed form with the
        route. (Read in middleware, the form body was consumed and the route got an empty form: caught by a
        test, 2026-11-19.)"""
        if request.method != "POST":
            return
        if not _same_origin(request) or not await _csrf_ok(request, request.state.session):
            raise HTTPException(403, "Request refused (security check failed). Reload the page.")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, dependencies=[Depends(post_guard)])
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    tpl = Jinja2Templates(directory=HERE / "templates")
    tpl.env.filters["http_link"] = safe_url
    tpl.env.globals["STATUS_LABEL"] = STATUS_LABEL

    @contextmanager
    def db():
        ses = new_session()
        try:
            yield ses
        finally:
            ses.close()

    def gate(ses):
        from cortex.graph import Deps, build_graph, checkpoint_target, open_checkpointer
        if state["ckpt"] is None:
            state["ckpt"] = open_checkpointer(checkpoint_target(s.db_url))
        return build_graph(Deps(session_factory=lambda: ses, audit=new_audit(), packet=None), state["ckpt"])

    # ---------- middleware: auth + CSRF + headers ----------

    OPEN = ("/login", "/auth/google", "/auth/callback", "/static/", "/healthz")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        sess = auth.read_session(s, request.cookies.get(auth.SESSION_COOKIE))
        request.state.session = sess
        if not path.startswith(OPEN) and sess is None:
            if request.headers.get("HX-Request"):
                resp = Response(status_code=401, headers={"HX-Redirect": "/login"})
            else:
                resp = RedirectResponse("/login", status_code=303)
            return _headers(resp)
        return _headers(await call_next(request))

    def _same_origin(request: Request) -> bool:
        origin = request.headers.get("origin") or request.headers.get("referer")
        if not origin:
            return True                     # some browsers omit both on same-origin POSTs; CSRF token still required
        u = urlsplit(origin)
        return f"{u.scheme}://{u.netloc}" == s.origin

    async def _csrf_ok(request: Request, sess: dict | None) -> bool:
        if sess is None:
            return False
        sent = request.headers.get("x-csrf-token")
        if sent is None:
            form = await request.form()
            sent = form.get("csrf")
        return isinstance(sent, str) and hmac.compare_digest(sent, sess["csrf"])

    def _headers(resp: Response) -> Response:
        h = resp.headers
        h["Content-Security-Policy"] = CSP
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "no-referrer"
        h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        h.setdefault("Cache-Control", "no-store")
        if s.secure:
            h["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return resp

    def page(request: Request, name: str, **ctx) -> HTMLResponse:
        return tpl.TemplateResponse(request, name, {"csrf": request.state.session["csrf"], "nav": name, **ctx})

    def partial(request: Request, name: str, **ctx) -> HTMLResponse:
        return tpl.TemplateResponse(request, name, {"csrf": request.state.session["csrf"], **ctx})

    def notice(msg: str, ok: bool = True) -> HTMLResponse:
        cls = "notice" if ok else "notice error"
        from markupsafe import escape
        return HTMLResponse(f'<div class="{cls}" role="status">{escape(msg)}</div>')

    # ---------- sign-in ----------

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/login", response_class=HTMLResponse)
    def login(request: Request, error: str | None = None):
        return tpl.TemplateResponse(request, "login.html", {"error": error})

    @app.get("/auth/google")
    def auth_google():
        url, flow = auth.start(s)
        resp = RedirectResponse(url, status_code=303)
        resp.set_cookie(auth.FLOW_COOKIE, flow, max_age=auth.FLOW_SECONDS, httponly=True, secure=s.secure,
                        samesite="lax", path="/auth")
        return resp

    @app.get("/auth/callback")
    def auth_callback(request: Request, state: str = "", code: str = "", error: str = ""):
        if error:
            return RedirectResponse("/login?error=Google+sign-in+was+cancelled", status_code=303)
        try:
            email = auth.finish(s, request.cookies.get(auth.FLOW_COOKIE), state, code, client=http_client)
        except auth.AuthError as e:
            new_audit().log("web", "login_refused", reason=str(e)[:60])
            resp = RedirectResponse("/login?error=" + str(e).replace(" ", "+")[:80], status_code=303)
            resp.delete_cookie(auth.FLOW_COOKIE, path="/auth")
            return resp
        new_audit().log("web", "login", user_id=s.user_id)
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(auth.FLOW_COOKIE, path="/auth")
        resp.set_cookie(auth.SESSION_COOKIE, auth.new_session(s, email), max_age=s.session_days * 86400,
                        httponly=True, secure=s.secure, samesite="lax", path="/")
        return resp

    @app.post("/logout")
    def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(auth.SESSION_COOKIE, path="/")
        return resp

    # ---------- queue ----------

    @app.get("/", response_class=HTMLResponse)
    def queue(request: Request, track: str = "", show: int = 50):
        cfg = load_config()
        tracks = [track] if track in ("data", "database", "ai") else None
        with db() as ses:
            rows = review_queue(ses, s.user_id, tracks=tracks, limit=2000,
                                max_age_days=(cfg.get("freshness") or {}).get("max_age_days", 30))
        show = max(10, min(int(show), 500))
        return page(request, "queue.html", rows=rows[:show], total=len(rows), track=track, show=show)

    @app.post("/jobs/{job_id}/skip", response_class=HTMLResponse)
    def skip(job_id: int):
        with db() as ses:
            ok = record_skip(ses, s.user_id, job_id, new_audit())
        return HTMLResponse("") if ok else notice("Already skipped, or not your job.", ok=False)

    @app.post("/jobs/{job_id}/applied", response_class=HTMLResponse)
    def applied(job_id: int):
        with db() as ses:
            try:
                job = tracker.update_status(ses, s.user_id, job_id, "applied", audit=new_audit(),
                                            note="from the dashboard")
            except tracker.TrackerError as e:
                return notice(f"Not recorded: {e}", ok=False)
            return notice(f"Recorded: applied to {job.company} | {job.title}. Apply on their site yourself.")

    # ---------- approvals ----------

    @app.get("/approvals", response_class=HTMLResponse)
    def approvals(request: Request):
        with db() as ses:
            items = pending_approvals(ses, s.user_id)
        return page(request, "approvals.html", items=items)

    @app.post("/approvals/{job_id}", response_class=HTMLResponse)
    def decide(job_id: int, decision: str = Form(...), note: str = Form("")):
        from cortex.graph import submit_decision
        if decision not in ("approve", "reject"):
            return notice("Unknown decision.", ok=False)
        with db() as ses:
            try:
                out = submit_decision(gate(ses), ses, s.user_id, job_id,
                                      {"decision": decision, "note": note.strip()[:500] or None})
            except (ValueError, LookupError, RuntimeError) as e:
                return notice(f"Not recorded: {e}", ok=False)
        if out.get("status") == "approved":
            return notice(f"Approved. Build the packet on your computer: cortex packet {job_id}")
        return notice("Rejected: the job is back in the queue.")

    # ---------- applications ----------

    @app.get("/applications", response_class=HTMLResponse)
    def apps(request: Request):
        cfg = (load_config().get("tracker") or {})
        with db() as ses:
            ready = ready_to_apply(ses, s.user_id)
            fups = tracker.followups(ses, s.user_id, date.today(), cfg.get("followup_days", 7),
                                     cfg.get("stale_days", 30))
            rows = applications(ses, s.user_id)
        for r in rows:
            r["next"] = sorted(tracker.TRANSITIONS.get(r["status"], set()) - {r["status"]})
        return page(request, "applications.html", ready=ready, fups=fups, rows=rows)

    @app.post("/jobs/{job_id}/status", response_class=HTMLResponse)
    def set_status(job_id: int, status: str = Form(...), note: str = Form("")):
        if status not in STATUS_LABEL:
            return notice("Unknown status.", ok=False)
        with db() as ses:
            try:
                job = tracker.update_status(ses, s.user_id, job_id, status, audit=new_audit(),
                                            note=(note.strip()[:300] or None))
            except tracker.TrackerError as e:
                return notice(f"Not recorded: {e}", ok=False)
            return notice(f"{job.company} | {job.title} is now {STATUS_LABEL[status].lower()}.")

    @app.post("/jobs/{job_id}/undo", response_class=HTMLResponse)
    def undo(job_id: int):
        with db() as ses:
            try:
                job = tracker.undo(ses, s.user_id, job_id, audit=new_audit(), note="from the dashboard")
            except tracker.TrackerError as e:
                return notice(f"Nothing undone: {e}", ok=False)
            return notice(f"Undone: {job.company} | {job.title} is back to {job.status.replace('_', ' ')}.")

    # ---------- report ----------

    @app.get("/report", response_class=HTMLResponse)
    def report(request: Request):
        cfg = (load_config().get("tracker") or {})
        with db() as ses:
            r = tracker.report(ses, s.user_id, date.today())
            fups = tracker.followups(ses, s.user_id, date.today(), cfg.get("followup_days", 7),
                                     cfg.get("stale_days", 30))
        return page(request, "report.html", r=r, fups=fups)

    return app
