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


def cmd_tailor(a, uid):
    """Start one tailoring run. It pauses at the approval gate; approve or reject in the dashboard."""
    from cortex.agents.tailor import Tailor
    from cortex.agents.truth_guard import TruthGuard
    from cortex.core.ratelimit import NoLimit
    from cortex.core.resume import TailoredResume, load_master
    from cortex.graph import Deps, build_graph, open_checkpointer, start_tailoring
    from cortex.llm.router import build_route

    cfg, master, audit, ses = load_config(a.config), load_master(a.master), AuditLog(), _session()
    router = build_route("tailor", cfg, audit, schema=TailoredResume.model_json_schema(), only=a.provider)
    tailor = Tailor(router, master, NoLimit(), audit)   # the router rate-limits per provider
    app = build_graph(Deps(session_factory=lambda: ses, audit=audit, guard=TruthGuard(master), tailor=tailor),
                      open_checkpointer())
    st = start_tailoring(app, ses, uid, a.job_id)
    print(f"job {a.job_id}: {st['status']} (attempts={st.get('attempts', 0)}, model={router.served_by or '-'})")
    for v in st.get("violations") or []:
        print(f"   {v}")
    if st["status"] == "pending_approval":
        print("   open the dashboard to approve or reject")


def packet_builder(master_path: str = "master_resume.yaml", contact_path: str = "contact.yaml",
                   answers_path: str = "answers.yaml"):
    """PacketBuilder from the local master + contact (+ optional answers) files, or (None, reason)."""
    from cortex.agents.packet import PacketBuilder, PacketError, load_contact
    from cortex.agents.workday import load_answers
    from cortex.core.resume import load_master
    try:
        return PacketBuilder(load_master(master_path), load_contact(contact_path),
                             answers=load_answers(answers_path)), None
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
    t.add_argument("--provider", default=None, help="pin one provider (must be local: the route refuses others)")
    t.add_argument("--master", default="master_resume.yaml")
    t.set_defaults(fn=cmd_tailor)
    k = sub.add_parser("packet", help="build/retry the PDF packet for an approved job")
    k.add_argument("job_id", type=int)
    k.add_argument("--master", default="master_resume.yaml")
    k.add_argument("--contact", default="contact.yaml")
    k.set_defaults(fn=cmd_packet)
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
