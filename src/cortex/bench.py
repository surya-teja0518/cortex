"""Tailor benchmark: which local model can actually tailor Surya's resume?

For each provider x N real jobs, run the real tailor + Truth Guard loop (1 attempt + 2 retries) and
record: passed/rejected, attempts, wall-clock seconds, violation kinds. No DB writes, no approval
runs: this measures the model, not the pipeline. Personal-data routes only (local models), because
the prompt carries the full master resume.
"""
import time
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import select

from cortex.agents.scorer import ScoringInput, render_job_fields
from cortex.agents.tailor import Tailor
from cortex.agents.truth_guard import TruthGuard, guarded_tailor
from cortex.core.audit import AuditLog
from cortex.core.resume import MasterResume, TailoredResume
from cortex.db.schema import JobRow
from cortex.llm.router import Router


class _NoWait:
    def wait(self):
        pass


@dataclass
class BenchRun:
    provider: str
    job_id: int
    job: str
    status: str            # passed | rejected
    attempts: int
    seconds: float
    violation_kinds: list[str] = field(default_factory=list)


def pick_jobs(session, user_id: str, n: int) -> list[JobRow]:
    """Top-n jobs he would realistically tailor for: ungated, not duplicates, not injection-flagged."""
    q = (select(JobRow)
         .where(JobRow.user_id == user_id, JobRow.gate_reason.is_(None), JobRow.dup_of_id.is_(None),
                JobRow.sanitizer_flags.is_(None), JobRow.status.in_(("to_apply", "applied")))
         .order_by(JobRow.score.desc(), JobRow.id).limit(n))
    return list(session.scalars(q))


def run_bench(session, user_id: str, master: MasterResume, providers: list, audit: AuditLog,
              n_jobs: int = 5, clock=time.monotonic) -> list[BenchRun]:
    jobs = pick_jobs(session, user_id, n_jobs)
    guard = TruthGuard(master)
    runs = []
    for prov in providers:
        router = Router("tailor_bench", [prov], audit, personal_data=True,
                        schema=TailoredResume.model_json_schema())
        tailor = Tailor(router, master, _NoWait(), audit)
        for j in jobs:
            wrapped, flags = render_job_fields(ScoringInput.from_row(j))
            if flags:
                continue
            t0 = clock()
            out = guarded_tailor(tailor.for_job(wrapped, job_id=j.id), guard, audit, user_id=user_id, job_id=j.id)
            runs.append(BenchRun(provider=prov.name, job_id=j.id, job=f"{j.company} | {j.title}"[:60],
                                 status=out.status, attempts=out.attempts, seconds=round(clock() - t0, 1),
                                 violation_kinds=sorted({v.kind for v in out.violations})))
    return runs


def format_bench(runs: list[BenchRun]) -> str:
    if not runs:
        return "no runs (no eligible jobs, or every job was injection-flagged)"
    lines = [f"{'provider':<24}{'jobs':>5}{'passed':>9}{'avg tries':>11}{'avg s/job':>11}{'max s':>8}  top failures"]
    for prov in dict.fromkeys(r.provider for r in runs):
        rs = [r for r in runs if r.provider == prov]
        passed = sum(r.status == "passed" for r in rs)
        fails = Counter(k for r in rs for k in r.violation_kinds).most_common(3)
        lines.append(f"{prov:<24}{len(rs):>5}{passed / len(rs):>9.0%}{sum(r.attempts for r in rs) / len(rs):>11.1f}"
                     f"{sum(r.seconds for r in rs) / len(rs):>11.1f}{max(r.seconds for r in rs):>8.1f}  "
                     + (", ".join(f"{k}x{n}" for k, n in fails) or "-"))
    lines.append("")
    for r in runs:
        lines.append(f"  {r.provider:<22} {r.status:<9} tries={r.attempts} {r.seconds:>6.1f}s  {r.job}"
                     + (f"  [{', '.join(r.violation_kinds)}]" if r.violation_kinds else ""))
    return "\n".join(lines)
