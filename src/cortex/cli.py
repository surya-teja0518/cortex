"""cortex import <csv...> | stats | compare | eval"""
import argparse
import os
import statistics
import sys

from sqlalchemy import func, select

from cortex.core.audit import AuditLog
from cortex.core.config import load_config
from cortex.core.ratelimit import RateLimiter
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
    """Deterministic vs LLM scorer against real apply decisions. LLM responses are cached by job."""
    from cortex.agents.scorer import GeminiProvider, GroqProvider, LLMScorer
    from cortex.eval import format_report, run_eval

    cfg = load_config(a.config)
    scorer = None
    if a.provider != "none":
        provider = {"gemini": GeminiProvider, "groq": GroqProvider}[a.provider]()
        scorer = LLMScorer(provider, RateLimiter(cfg["rate_limits"]["llm"]), AuditLog(),
                           cfg["scoring"]["bands"])
    print(format_report(run_eval(_session(), uid, scorer)))


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
    e.add_argument("--provider", choices=["none", "gemini", "groq"], default="none",
                   help="LLM provider (gemini/groq are stubs until wired; their jobs fall back)")
    e.set_defaults(fn=cmd_eval)
    a = p.parse_args(argv)
    a.fn(a, os.getenv("CORTEX_USER_ID", "surya"))


if __name__ == "__main__":
    sys.exit(main())
