"""cortex import <csv...> | stats | compare | eval | tailor <job_id> | packet <job_id> | llm-check | bench"""
import argparse
import os
import statistics
import sys

from sqlalchemy import func, select

from cortex.core.audit import AuditLog
from cortex.core.config import load_config
from cortex.core.secrets import get_secret
from cortex.db.schema import JobRow, get_session
from cortex.sources.synapse_import import import_files


def _session():
    return get_session(get_secret("CORTEX_DB_URL", required=False) or "sqlite:///data/cortex.db")


def cmd_import(a, uid):
    s = import_files(a.files, _session(), load_config(a.config), uid, AuditLog())
    print(f"read={s.rows_read} inserted={s.inserted} updated={s.updated} blank={s.skipped_blank} "
          f"gated={s.gated} flagged={s.flagged} possible_dups={s.duplicates}")


def cmd_fetch(a, uid):
    """Pull postings from the public ATS boards in config.yaml (sources.ats.boards)."""
    from cortex.sources.ats import fetch_all
    cfg = load_config(a.config)
    if not (cfg.get("sources") or {}).get("ats", {}).get("boards"):
        raise SystemExit("no boards configured: add them under sources.ats.boards in config.yaml")
    res = fetch_all(_session(), cfg, uid, audit=AuditLog(), only=a.board, dry_run=a.dry_run)
    if a.board and not res:
        raise SystemExit(f"{a.board!r} is not in sources.ats.boards (format ats:board, e.g. greenhouse:databricks)")
    for r in res:
        head = f"{r.ats}:{r.board} ({r.company})"
        if r.error:
            print(f"{head}: ERROR {r.error}")
            continue
        print(f"{head}: fetched={r.fetched} relevant={r.kept} new={r.inserted} updated={r.updated} "
              f"closed={r.closed} gated={r.gated} flagged={r.flagged}")
        for t, loc, band in r.new_jobs[:10]:
            print(f"    + [{band}] {t} | {loc}")
    total = sum(r.inserted for r in res)
    print(f"{total} new jobs" + (" (dry run: nothing saved)" if a.dry_run else "; review them in the dashboard"))


def cmd_pull(a, uid):
    """Read SYNAPSE's recent Apify datasets (LinkedIn/Naukri). Read only: never starts an actor run."""
    from cortex.sources.apify import ApifyError, pull_all
    cfg = load_config(a.config)
    try:
        res = pull_all(_session(), cfg, uid, audit=AuditLog(), dry_run=a.dry_run)
    except ApifyError as e:
        raise SystemExit(f"not pulled: {e}") from None
    for r in res:
        if r.error:
            print(f"{r.actor}: ERROR {r.error}")
            continue
        print(f"{r.actor}: datasets={r.datasets} items={r.items} relevant={r.kept} new={r.inserted} "
              f"updated={r.updated} enriched_sheet_rows={r.enriched} gated={r.gated} flagged={r.flagged}")
        for c, t, loc, band in r.new_jobs[:10]:
            print(f"    + [{band}] {c} | {t} | {loc}")
    total = sum(r.inserted for r in res)
    print(f"{total} new jobs" + (" (dry run: nothing saved)" if a.dry_run else ""))


def cmd_check_open(a, uid):
    """Check whether recent to_apply LinkedIn jobs are still open (public no-login page; opt-in, rate-limited)."""
    from cortex.sources.linkedin_status import check_open
    cfg = load_config(a.config)
    try:
        r = check_open(_session(), cfg, uid, dry_run=a.dry_run, limit=a.limit, audit=AuditLog())
    except PermissionError as e:
        raise SystemExit(str(e)) from None
    print(f"candidates={r.candidates} checked={r.checked} open={r.open} closed={r.closed} unclear={r.unknown}")
    for i, c, t in r.closed_jobs[:20]:
        print(f"  closed: [{i}] {c} | {t}")
    if r.stopped:
        print(f"STOPPED: {r.stopped}")
    print("dry run: nothing changed" if a.dry_run else f"{r.closed} marked not_accepting")


def cmd_stats(a, uid):
    ses = _session()
    for label, col in (("status", JobRow.status), ("track", JobRow.track), ("cortex band", JobRow.band)):
        print(f"-- {label}")
        for k, n in ses.execute(select(col, func.count()).where(JobRow.user_id == uid).group_by(col)):
            print(f"   {k}: {n}")


def cmd_compare(a, uid):
    """CORTEX deterministic score vs SYNAPSE's stored Match: the first eval-harness metric."""
    rows = _session().scalars(select(JobRow).where(
        JobRow.user_id == uid, JobRow.synapse_score.is_not(None), JobRow.score.is_not(None))).all()
    if len(rows) < 3:
        print("not enough rows to compare")
        return
    syn = [r.synapse_score for r in rows]
    cor = [r.score for r in rows]
    diffs = [c - s for c, s in zip(cor, syn)]
    try:
        r = statistics.correlation(syn, cor)
    except statistics.StatisticsError:
        r = float("nan")
    agree = sum(1 for x in rows if x.band == x.synapse_priority) / len(rows)
    print(f"n={len(rows)}  MAE={statistics.mean(abs(d) for d in diffs):.1f}  "
          f"mean(cortex-synapse)={statistics.mean(diffs):+.1f}  pearson_r={r:.2f}  band_agreement={agree:.0%}")
    for x in sorted(rows, key=lambda x: -abs(x.score - x.synapse_score))[:5]:
        print(f"   {x.company} | {x.title} | synapse={x.synapse_score} cortex={x.score}")


def cmd_eval(a, uid):
    """Deterministic vs LLM scorer against real decisions. LLM results are cached per job + provider."""
    from cortex.agents.scorer import LLMScore, LLMScorer
    from cortex.core.ratelimit import NoLimit
    from cortex.eval import EvalAborted, format_report, run_eval
    from cortex.llm.router import build_route

    cfg, audit = load_config(a.config), AuditLog()
    scorer = None
    if a.exclude_fields is not None and a.provider == "none":
        raise SystemExit("--exclude-fields only applies to an LLM run: add --provider")
    # Default comes from config (llm.scorer_exclude_fields); "none" means include every field.
    raw = a.exclude_fields if a.exclude_fields is not None else ",".join(cfg["llm"].get("scorer_exclude_fields", []))
    excl = frozenset(x.strip() for x in raw.split(",") if x.strip() and x.strip() != "none")
    if a.provider != "none":
        only = None if a.provider == "route" else a.provider
        router = build_route("scorer", cfg, audit, schema=LLMScore.model_json_schema(), only=only)
        # The router rate-limits per provider; a second limiter here would only slow every call down.
        scorer = LLMScorer(router, NoLimit(), audit, cfg["scoring"]["bands"], exclude_fields=excl)

    def progress(done, total, live, cached, unscored, secs):
        if done % 10 and done != total:
            return
        eta = f"~{(total - done) * secs / live / 60:.0f} min left if the rest is uncached" if live else "cache only so far"
        print(f"  {done}/{total}  live={live} cached={cached} unscored={unscored}  {secs / 60:.1f} min elapsed, {eta}",
              file=sys.stderr, flush=True)
    try:
        print(format_report(run_eval(_session(), uid, scorer, progress=progress if scorer else None)))
    except EvalAborted as e:
        raise SystemExit(f"eval stopped: {e}") from None


def _build_scorer(cfg: dict, audit, provider: str = "route"):
    from cortex.agents.scorer import LLMScore, LLMScorer
    from cortex.core.ratelimit import NoLimit
    from cortex.llm.router import build_route
    excl = frozenset(cfg["llm"].get("scorer_exclude_fields", []))
    router = build_route("scorer", cfg, audit, schema=LLMScore.model_json_schema(),
                         only=None if provider == "route" else provider)
    return LLMScorer(router, NoLimit(), audit, cfg["scoring"]["bands"], exclude_fields=excl)


def cmd_score(a, uid):
    """LLM-score new jobs (to_apply, ungated, not dup, not flagged, not scored yet). Newest first."""
    from cortex.agents.batch_score import score_new
    cfg, audit = load_config(a.config), AuditLog()
    scorer = _build_scorer(cfg, audit, a.provider)
    limit = a.limit or cfg.get("daily", {}).get("score_limit", 100)

    def progress(i, n):
        if i % 10 == 0 or i == n:
            print(f"  {i}/{n}", file=sys.stderr, flush=True)
    r = score_new(_session(), uid, scorer, limit=limit, progress=progress)
    print(f"candidates={r.candidates} scored={r.scored} kept_rule_score={r.unscored}"
          + (f"\nstopped early: {r.stopped_early}" if r.stopped_early else ""))


def _digest_fn(cfg, uid, audit, dry_run):
    from datetime import date

    from cortex.core.secrets import get_secret
    from cortex.notify.digest import render, run_digest, subject

    def digest(ses, notes):
        d, status = run_digest(ses, uid, date.today(), cfg=cfg, dry_run=dry_run, notes=notes, audit=audit,  # noqa: DTZ011
                               address=get_secret("GMAIL_ADDRESS", required=False),
                               password=get_secret("GMAIL_APP_PASSWORD", required=False))
        if dry_run:
            print(f"Subject: {subject(d)}\n\n{render(d)}")
        return status
    return digest


def cmd_digest(a, uid):
    """Build the email digest; send it to your own GMAIL_ADDRESS (or print it with --dry-run)."""
    from cortex.notify.digest import DigestError
    cfg, audit = load_config(a.config), AuditLog()
    try:
        print(_digest_fn(cfg, uid, audit, a.dry_run)(_session(), []))
    except DigestError as e:
        raise SystemExit(f"digest not sent: {e}") from None


def cmd_daily(a, uid):
    """fetch -> LLM-score new jobs -> digest. What launchd runs. One failing step never stops the rest."""
    from datetime import datetime

    from cortex.agents.batch_score import score_new
    from cortex.daily import run_daily
    from cortex.sources.ats import fetch_all
    cfg, audit, ses = load_config(a.config), AuditLog(), _session()
    dcfg = cfg.get("daily", {})
    has_boards = bool((cfg.get("sources") or {}).get("ats", {}).get("boards"))
    fetch = (lambda s: fetch_all(s, cfg, uid, audit=audit)) if has_boards and not a.no_fetch else None
    has_actors = bool((cfg.get("sources") or {}).get("apify", {}).get("actors"))
    pull = None
    if has_actors and not a.no_pull:
        from cortex.sources.apify import pull_all
        pull = lambda s: pull_all(s, cfg, uid, audit=audit)
    score = None
    if not a.no_score:
        score = lambda s: score_new(s, uid, _build_scorer(cfg, audit), limit=dcfg.get("score_limit", 100))
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] cortex daily")  # noqa: DTZ005 (local time for the log)
    res = run_daily(ses, cfg, uid, datetime.now().date(), fetch=fetch, score=score, pull=pull,  # noqa: DTZ005
                    digest=_digest_fn(cfg, uid, audit, a.dry_run), audit=audit)
    for n in res.notes:
        print(f"  {n}")
    print(f"  digest: {res.digest_status}")
    if res.failed_steps:
        raise SystemExit(f"failed steps: {', '.join(res.failed_steps)}")


def cmd_schedule(a, uid):
    """Print or install the launchd job that runs `cortex daily` every day (macOS)."""
    from pathlib import Path

    from cortex.schedule import LABEL, find_cortex_bin, install, render_plist
    cfg = load_config(a.config).get("daily", {})
    hour, minute = cfg.get("hour", 9), cfg.get("minute", 7)
    repo, exe = Path.cwd().resolve(), find_cortex_bin()
    if not a.install:
        sys.stdout.write(render_plist(repo, exe, hour, minute).decode())
        print(f"\n(preview: run `cortex schedule --install` from the repo folder to install for {hour:02d}:{minute:02d})")
        return
    path = install(repo, exe, hour, minute)
    print(f"wrote {path}\nnow load it (once):\n  launchctl unload {path} 2>/dev/null; launchctl load {path}\n"
          f"check: launchctl list | grep {LABEL}    log: {repo / 'data' / 'daily.log'}\n"
          f"stop:  launchctl unload {path}")


def cmd_tailor(a, uid):
    """Start one tailoring run. It pauses at the approval gate; approve or reject in the dashboard."""
    from cortex.agents.tailor import Tailor
    from cortex.agents.track_tailor import TrackError, TrackTailor
    from cortex.agents.truth_guard import TruthGuard
    from cortex.core.ratelimit import NoLimit
    from cortex.core.resume import TailoredResume, load_master
    from cortex.graph import Deps, build_graph, open_checkpointer, start_tailoring
    from cortex.llm.router import build_route

    cfg, master, audit, ses = load_config(a.config), load_master(a.master), AuditLog(), _session()
    tcfg = cfg.get("tailor", {})
    mode = a.mode or tcfg.get("mode", "track")
    router = None
    if mode in ("track", "original"):
        try:
            tailor = TrackTailor(master, tcfg.get("track_map"), tcfg.get("default_track", "data_engineer"),
                                 override=a.track, original=mode == "original")
        except TrackError as e:
            raise SystemExit(f"track mode: {e}. Add tracks, or run with --mode llm") from None
    else:
        router = build_route("tailor", cfg, audit, schema=TailoredResume.model_json_schema(), only=a.provider)
        tailor = Tailor(router, master, NoLimit(), audit,   # the router rate-limits per provider
                        cover_note=tcfg.get("cover_note", False))
    app = build_graph(Deps(session_factory=lambda: ses, audit=audit, guard=TruthGuard(master), tailor=tailor),
                      open_checkpointer())
    st = start_tailoring(app, ses, uid, a.job_id)
    used = (st.get("resume") or {}).get("based_on") or (router.served_by if router else None) or "-"
    print(f"job {a.job_id}: {st['status']} (attempts={st.get('attempts', 0)}, based on {used})")
    for v in st.get("violations") or []:
        print(f"   {v}")
    if st["status"] == "pending_approval":
        print("   open the dashboard to approve or reject")


def packet_builder(master_path: str = "master_resume.yaml", contact_path: str = "contact.yaml",
                   answers_path: str = "answers.yaml", originals_dir: str | None = None):
    """PacketBuilder from the local master + contact (+ optional answers) files, or (None, reason)."""
    from cortex.agents.packet import PacketBuilder, PacketError, load_contact
    from cortex.agents.workday import load_answers
    from cortex.core.resume import load_master
    try:
        odir = originals_dir or load_config().get("tailor", {}).get("originals_dir", "resumes")
        return PacketBuilder(load_master(master_path), load_contact(contact_path),
                             answers=load_answers(answers_path), originals_dir=odir), None
    except (OSError, ValueError, PacketError) as e:
        return None, str(e)


def cmd_packet(a, uid):
    """Build (or retry) the packet for an approved job: resume.pdf + job.md under packets/."""
    from cortex.graph import Deps, build_graph, build_packet, open_checkpointer

    builder, why = packet_builder(a.master, a.contact)
    if builder is None:
        raise SystemExit(f"cannot build packets: {why}")
    ses = _session()
    app = build_graph(Deps(session_factory=lambda: ses, audit=AuditLog(), packet=builder), open_checkpointer())
    st = build_packet(app, ses, uid, a.job_id)
    print(f"job {a.job_id}: packet at {st['packet_path']}  (CORTEX never submits: apply yourself)")


def cmd_workday_fill(a, uid):
    """Type your details into a Workday application you have open. Never clicks anything, never submits."""
    import json
    from pathlib import Path

    from cortex.agents.packet import load_contact
    from cortex.agents.workday import load_answers
    from cortex.agents.workday_browser import open_browser
    from cortex.agents.workday_fill import (
        build_plan,
        inspect,
        is_workday,
        load_fmap,
        run,
    )
    from cortex.core.resume import TailoredResume, load_master
    from cortex.db.schema import PacketRow, TailoredResumeRow

    ses = _session()
    job = ses.get(JobRow, a.job_id)
    if job is None or job.user_id != uid:
        raise SystemExit("job not found")
    start = a.url or (job.url if is_workday(job.url or "") else None)
    if a.url and not is_workday(a.url):
        raise SystemExit("--url must be an https Workday link (*.myworkdayjobs.com / *.myworkday.com / *.workday.com)")
    plan = None
    if not a.inspect:
        tr = ses.scalars(select(TailoredResumeRow).where(
            TailoredResumeRow.user_id == uid, TailoredResumeRow.job_id == a.job_id,
            TailoredResumeRow.status == "approved").order_by(TailoredResumeRow.id.desc())).first()
        if tr is None:
            raise SystemExit("no approved resume for this job: run `cortex tailor` and approve it in the dashboard first")
        pk = ses.scalars(select(PacketRow.path).where(PacketRow.user_id == uid, PacketRow.job_id == a.job_id)
                         .order_by(PacketRow.id.desc())).first()
        plan = build_plan(load_master(a.master), TailoredResume.model_validate(json.loads(tr.resume_json)),
                          load_contact(a.contact), load_answers(), Path(pk) / "resume.pdf" if pk else None)
    pw, browser, page = open_browser(start)
    try:
        if a.inspect:
            input("Open the Workday page you want to map, then press Enter here... ")
            print("\n".join(inspect(page)))
            return
        print("Log in and open the application in the browser window."
              + ("" if start else " (Paste the Workday apply link into its address bar.)"))
        print("CORTEX only types into EMPTY fields on Workday pages. It never clicks Next or Submit: review every "
              "page yourself. Close the browser window when you're done.")
        st = run(page, plan, load_fmap(a.fields), on_fill=lambda k: print(f"  filled {'/'.join(map(str, k))}"))
        AuditLog().log("workday_fill", "session", user_id=uid, job_id=a.job_id, filled=st.filled)
        print(f"done: {st.filled} fields filled. Submit was (and always is) yours.")
    finally:
        try:
            browser.close()
        finally:
            pw.stop()


def cmd_track(a, uid):
    """Record what happened: applied / interview / offer / rejected. Forward-only, audit-logged."""
    from datetime import date

    from cortex.agents.tracker import TrackerError, update_status
    ses = _session()   # keep a reference: the job is printed after the update
    try:
        job = update_status(ses, uid, a.job_id, a.status, note=a.note,
                            on=date.fromisoformat(a.date) if a.date else None, audit=AuditLog())
    except (TrackerError, ValueError) as e:
        raise SystemExit(f"not recorded: {e}") from None
    print(f"job {a.job_id}: {job.company} | {job.title} -> {job.status}"
          + (f" on {job.applied_date}" if a.status == "applied" else ""))


def cmd_report(a, uid):
    """Applications by status, response rate, applications per week, follow-ups due."""
    from datetime import date

    from cortex.agents.tracker import followups, format_report, report
    ses, today = _session(), date.today()  # noqa: DTZ011 (the user's local date is what's meant)
    tcfg = load_config(a.config).get("tracker", {})
    fups = followups(ses, uid, today, tcfg.get("followup_days", 7), tcfg.get("stale_days", 30))
    print(format_report(report(ses, uid, today), fups))


def cmd_undo(a, uid):
    """Revert the last status change CORTEX recorded for a job (history keeps an 'undo' event)."""
    from cortex.agents.tracker import TrackerError, undo
    ses = _session()
    try:
        job = undo(ses, uid, a.job_id, note=a.note, audit=AuditLog())
    except TrackerError as e:
        raise SystemExit(f"not undone: {e}") from None
    print(f"job {a.job_id}: {job.company} | {job.title} -> back to {job.status}")


def cmd_close_stale(a, uid):
    """Mark applications with no reply for N days as no_response. Dry run unless --yes."""
    from datetime import date

    from cortex.agents.tracker import close_stale
    days = a.days or load_config(a.config).get("tracker", {}).get("stale_days", 30)
    rows = close_stale(_session(), uid, date.today(), days, dry_run=not a.yes, audit=AuditLog())  # noqa: DTZ011
    for r in rows:
        print(f"  {r['days_quiet']:>3}d quiet  [{r['job_id']}] {r['company']} | {r['role']}")
    print(f"{len(rows)} applications quiet for {days}+ days: "
          + ("marked no_response" if a.yes else "dry run, nothing changed (add --yes to mark them no_response)"))


def cmd_llm_check(a, uid):
    """Ping every configured provider with a tiny JSON prompt. Prints OK/latency or the (redacted) error."""
    import time

    from cortex.llm.providers import ProviderError, build_provider

    prompt = 'Reply with only this JSON object: {"ok": true}'
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    for name, spec in load_config(a.config)["llm"]["providers"].items():
        try:
            p = build_provider(name, spec)
        except RuntimeError as e:          # missing key
            print(f"{name:<16} MISSING  {e}")
            continue
        t0 = time.monotonic()
        try:
            out = p.complete(prompt, schema=schema)
        except ProviderError as e:
            print(f"{name:<16} ERROR    {e}")
            continue
        print(f"{name:<16} OK       {time.monotonic() - t0:5.1f}s  model={p.model}  reply={out.strip()[:40]!r}")


def cmd_bench(a, uid):
    """Benchmark local models on the real tailor task (your master resume x N real jobs)."""
    from cortex.bench import format_bench, run_bench
    from cortex.core.resume import load_master
    from cortex.llm.providers import build_provider

    cfg = load_config(a.config)
    specs = cfg["llm"]["providers"]
    names = a.providers.split(",")
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise SystemExit(f"unknown provider(s) {unknown}; configured: {sorted(specs)}")
    providers = [build_provider(n, specs[n]) for n in names]
    runs = run_bench(_session(), uid, load_master(a.master), providers, AuditLog(), n_jobs=a.jobs)
    print(format_bench(runs))


def main(argv=None):
    p = argparse.ArgumentParser(prog="cortex")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("import", help="import SYNAPSE tracker CSV exports (master first, then batches)")
    i.add_argument("files", nargs="+")
    i.set_defaults(fn=cmd_import)
    co = sub.add_parser("check-open", help="are recent LinkedIn to_apply jobs still open? (opt-in, slow on purpose)")
    co.add_argument("--dry-run", action="store_true", help="check and report; change nothing")
    co.add_argument("--limit", type=int, default=None, help="max jobs this run (capped by config max_per_run)")
    co.set_defaults(fn=cmd_check_open)
    wf = sub.add_parser("workday-fill", help="type your details into an open Workday application (never submits)")
    wf.add_argument("job_id", type=int)
    wf.add_argument("--url", default=None, help="the Workday apply link (https://<company>.wd*.myworkdayjobs.com/...)")
    wf.add_argument("--inspect", action="store_true", help="print the page's field ids (no values) to fix the map")
    wf.add_argument("--fields", default="workday_fields.yaml")
    wf.add_argument("--master", default="master_resume.yaml")
    wf.add_argument("--contact", default="contact.yaml")
    wf.set_defaults(fn=cmd_workday_fill)
    pl = sub.add_parser("pull", help="read SYNAPSE's recent Apify datasets (APIFY_TOKEN in .env); never starts runs")
    pl.add_argument("--dry-run", action="store_true", help="show what would be added; save nothing")
    pl.set_defaults(fn=cmd_pull)
    f = sub.add_parser("fetch", help="pull postings from public ATS boards (Greenhouse/Lever/Ashby) in config.yaml")
    f.add_argument("--board", default=None, help="only this board, e.g. greenhouse:databricks")
    f.add_argument("--dry-run", action="store_true", help="fetch and show what would be added; save nothing")
    f.set_defaults(fn=cmd_fetch)
    sc = sub.add_parser("score", help="LLM-score new jobs and keep the results (digest + dashboard rank by them)")
    sc.add_argument("--limit", type=int, default=None, help="max jobs this run (default: config daily.score_limit)")
    sc.add_argument("--provider", default="route", help="route (default: fallback chain) or one provider name")
    sc.set_defaults(fn=cmd_score)
    dg = sub.add_parser("digest", help="email yourself today's digest (GMAIL_ADDRESS + GMAIL_APP_PASSWORD in .env)")
    dg.add_argument("--dry-run", action="store_true", help="print it instead of sending")
    dg.set_defaults(fn=cmd_digest)
    dl = sub.add_parser("daily", help="pull -> fetch -> score -> digest (what the schedule runs)")
    dl.add_argument("--dry-run", action="store_true", help="print the digest instead of sending it")
    dl.add_argument("--no-fetch", action="store_true")
    dl.add_argument("--no-pull", action="store_true")
    dl.add_argument("--no-score", action="store_true")
    dl.set_defaults(fn=cmd_daily)
    sh = sub.add_parser("schedule", help="show / install the macOS launchd job for `cortex daily`")
    sh.add_argument("--install", action="store_true", help="write ~/Library/LaunchAgents/com.cortex.daily.plist")
    sh.set_defaults(fn=cmd_schedule)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    sub.add_parser("compare").set_defaults(fn=cmd_compare)
    e = sub.add_parser("eval", help="deterministic vs LLM scorer against applied/interview/offer labels")
    e.add_argument("--provider", default="none",
                   help="none | route (the configured scorer fallback chain) | one provider name from config.yaml")
    e.add_argument("--exclude-fields", default=None,
                   help="job fields left out of the LLM prompt (default: llm.scorer_exclude_fields in config, "
                        "currently stack_fit); 'none' = include every field")
    e.set_defaults(fn=cmd_eval)
    t = sub.add_parser("tailor", help="tailor a resume for one job; pauses for approval in the dashboard")
    t.add_argument("job_id", type=int)
    t.add_argument("--mode", choices=["original", "track", "llm"], default=None,
                   help="track (default): pick one of your curated resumes and reorder bullets; llm: rewrite with Ollama")
    t.add_argument("--track", default=None, help="force a track, e.g. oracle_dba (default: from the job's track)")
    t.add_argument("--provider", default=None, help="llm mode: pin one provider (must be local)")
    t.add_argument("--master", default="master_resume.yaml")
    t.set_defaults(fn=cmd_tailor)
    k = sub.add_parser("packet", help="build/retry the PDF packet for an approved job")
    k.add_argument("job_id", type=int)
    k.add_argument("--master", default="master_resume.yaml")
    k.add_argument("--contact", default="contact.yaml")
    k.set_defaults(fn=cmd_packet)
    for status in ("applied", "interview", "offer", "rejected", "no_response"):
        s = sub.add_parser(status.replace("_", "-"), help=f"record: job is now {status}")
        s.add_argument("job_id", type=int)
        s.add_argument("--note", default=None)
        if status == "applied":
            s.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
        s.set_defaults(fn=cmd_track, status=status, date=None)
    sub.add_parser("report", help="applications, response rate, follow-ups due").set_defaults(fn=cmd_report)
    u = sub.add_parser("undo", help="revert the last status change recorded for a job (e.g. 'applied' by mistake)")
    u.add_argument("job_id", type=int)
    u.add_argument("--note", default=None)
    u.set_defaults(fn=cmd_undo)
    c = sub.add_parser("close-stale", help="mark applications with no reply for N days as no_response (dry run)")
    c.add_argument("--days", type=int, default=None, help="default: config tracker.stale_days (30)")
    c.add_argument("--yes", action="store_true", help="actually change them (default: preview only)")
    c.set_defaults(fn=cmd_close_stale)
    sub.add_parser("llm-check", help="ping every configured LLM provider").set_defaults(fn=cmd_llm_check)
    b = sub.add_parser("bench", help="benchmark local models on the real tailor task")
    b.add_argument("--providers", default="ollama_8b,ollama_granite", help="comma-separated provider names")
    b.add_argument("--jobs", type=int, default=5)
    b.add_argument("--master", default="master_resume.yaml")
    b.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv)
    a.fn(a, os.getenv("CORTEX_USER_ID", "surya"))


if __name__ == "__main__":
    sys.exit(main())
