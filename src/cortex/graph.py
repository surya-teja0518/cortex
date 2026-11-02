"""Per-job LangGraph: Tailor -> Truth Guard -> [HUMAN APPROVAL GATE] -> record -> Packet.

    START -> tailor_and_guard --passed--> approval (interrupt) -> record --approved--> packet -> END
                              \\-manual_review / guard_rejected-> END   \\-rejected-> END

One graph run = one LangGraph *thread* (thread_id), checkpointed to SQLite after every node.
- `interrupt()` in `approval` pauses the run and persists it; nothing downstream runs until a
  human resumes it with `Command(resume=decision)`. Only the Streamlit dashboard calls `submit_decision`.
- After a crash or restart, a new process compiling the same graph against the same checkpoint DB
  resumes from the last completed node. Completed nodes (the LLM tailoring) are NOT re-run.
- On resume, the interrupted node runs again from its first line, so `approval` has no side effects
  before `interrupt()`; all writes happen in `record`, which is idempotent.
- `packet` runs only on the approved branch. If it fails (e.g. contact.yaml missing), the approval is
  already recorded and checkpointed; the thread waits at `packet` and `build_packet` resumes it later
  without asking for approval again.
"""
import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from cortex.agents.packet import PacketBuilder, PacketError
from cortex.agents.scorer import ScoringInput, render_job_fields
from cortex.agents.tailor import Tailor
from cortex.agents.truth_guard import TruthGuard, guarded_tailor
from cortex.core.audit import AuditLog
from cortex.core.resume import TailoredResume
from cortex.db.schema import ApprovalRow, JobRow, PacketRow, TailoredResumeRow


class JobState(TypedDict, total=False):
    user_id: str
    job_id: int
    thread_id: str
    status: str            # manual_review | guard_rejected | pending_approval | approved | rejected
    resume: dict | None
    violations: list[str]
    attempts: int
    decision: dict
    packet_path: str


class ApprovalDecision(BaseModel):
    """The only shape a human decision may take. Validated before AND after resuming the graph."""
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=500)


@dataclass
class Deps:
    session_factory: Callable    # () -> SQLAlchemy session
    audit: AuditLog
    guard: TruthGuard | None = None
    tailor: Tailor | None = None  # None in the dashboard: it only resumes, never tailors
    packet: PacketBuilder | None = None   # None -> approved threads wait at `packet` until configured


def open_checkpointer(path: str = "data/checkpoints.db") -> SqliteSaver:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def _row(ses, thread_id: str) -> TailoredResumeRow | None:
    return ses.scalar(select(TailoredResumeRow).where(TailoredResumeRow.thread_id == thread_id))


def build_graph(deps: Deps, checkpointer):
    def tailor_and_guard(state: JobState) -> dict:
        uid, jid, tid = state["user_id"], state["job_id"], state["thread_id"]
        ses = deps.session_factory()
        job = ses.scalar(select(JobRow).where(JobRow.id == jid, JobRow.user_id == uid))
        if job is None:
            raise LookupError("job not found for this user")
        wrapped, flags = render_job_fields(ScoringInput.from_row(job))
        ref = {"user_id": uid, "job_id": jid}
        if flags:
            # Injection-flagged jobs never reach the tailor; they wait for manual review.
            out = {"status": "manual_review", "resume": None, "attempts": 0,
                   "violations": sorted({f.split(":", 1)[0] for f in flags})}
            deps.audit.log("graph", "manual_review", reason="injection_flagged", **ref)
        else:
            if deps.tailor is None or deps.guard is None:
                raise RuntimeError("graph built without tailor/guard cannot start a tailoring run")
            g = guarded_tailor(deps.tailor.for_job(wrapped, **ref), deps.guard, deps.audit, **ref)
            out = {"status": "pending_approval" if g.status == "passed" else "guard_rejected",
                   "resume": g.resume.model_dump() if g.resume else None,
                   "attempts": g.attempts, "violations": [v.feedback() for v in g.violations]}
        row = _row(ses, tid) or TailoredResumeRow(user_id=uid, job_id=jid, thread_id=tid)
        row.status, row.attempts = out["status"], out["attempts"]
        row.resume_json = json.dumps(out["resume"]) if out["resume"] else None
        row.violations_json = json.dumps(out["violations"]) if out["violations"] else None
        ses.add(row)
        if out["status"] == "pending_approval":
            job.status = "pending_approval"
        ses.commit()
        return out

    def approval(state: JobState) -> dict:
        # No side effects above interrupt(): this line re-runs on every resume.
        raw = interrupt({"job_id": state["job_id"], "resume": state["resume"]})
        return {"decision": ApprovalDecision.model_validate(raw).model_dump()}

    def record(state: JobState) -> dict:
        uid, jid, tid = state["user_id"], state["job_id"], state["thread_id"]
        d = ApprovalDecision.model_validate(state["decision"])
        status = "approved" if d.decision == "approve" else "rejected"
        ses = deps.session_factory()
        row = _row(ses, tid)
        if row is not None and row.status == "pending_approval":   # idempotent if re-run after a crash
            row.status, row.note, row.decided_at = status, d.note, datetime.now(UTC)
            ses.add(ApprovalRow(user_id=uid, job_id=jid, decision="approved" if status == "approved" else "rejected"))
            job = ses.scalar(select(JobRow).where(JobRow.id == jid, JobRow.user_id == uid))
            if job is not None:
                job.status = "approved" if status == "approved" else "to_apply"
            ses.commit()
            deps.audit.log("approval", status, user_id=uid, job_id=jid, thread_id=tid)
        return {"status": status}

    def packet(state: JobState) -> dict:
        uid, jid, tid = state["user_id"], state["job_id"], state["thread_id"]
        if deps.packet is None:
            raise PacketError("packet builder not configured: needs master_resume.yaml and contact.yaml")
        ses = deps.session_factory()
        job = ses.scalar(select(JobRow).where(JobRow.id == jid, JobRow.user_id == uid))
        if job is None:
            raise PacketError("job not found for this user")
        res = deps.packet.build(TailoredResume.model_validate(state["resume"]), user_id=uid, job_id=jid,
                                company=job.company, title=job.title, location=job.location or "", url=job.url)
        row = ses.scalar(select(PacketRow).where(PacketRow.thread_id == tid)) or \
            PacketRow(user_id=uid, job_id=jid, thread_id=tid)
        row.path = str(res.folder)
        ses.add(row)
        ses.commit()
        deps.audit.log("packet", "built", user_id=uid, job_id=jid, thread_id=tid)
        return {"packet_path": str(res.folder)}

    g = StateGraph(JobState)
    g.add_node("tailor_and_guard", tailor_and_guard)
    g.add_node("approval", approval)
    g.add_node("record", record)
    g.add_node("packet", packet)
    g.add_edge(START, "tailor_and_guard")
    g.add_conditional_edges("tailor_and_guard",
                            lambda s: "approval" if s["status"] == "pending_approval" else END,
                            ["approval", END])
    g.add_edge("approval", "record")
    g.add_conditional_edges("record", lambda s: "packet" if s["status"] == "approved" else END, ["packet", END])
    g.add_edge("packet", END)
    return g.compile(checkpointer=checkpointer)


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_tailoring(app, session, user_id: str, job_id: int) -> JobState:
    """Start a new tailoring run for one job. Refused if one is already waiting for approval."""
    if session.scalar(select(JobRow.id).where(JobRow.id == job_id, JobRow.user_id == user_id)) is None:
        raise LookupError("job not found for this user")
    if session.scalar(select(TailoredResumeRow.id).where(
            TailoredResumeRow.user_id == user_id, TailoredResumeRow.job_id == job_id,
            TailoredResumeRow.status == "pending_approval")):
        raise ValueError("a tailored resume for this job is already waiting for approval")
    tid = f"{user_id}:{job_id}:{uuid.uuid4().hex[:12]}"
    app.invoke({"user_id": user_id, "job_id": job_id, "thread_id": tid}, _cfg(tid))
    return app.get_state(_cfg(tid)).values


def submit_decision(app, session, user_id: str, job_id: int, decision: dict) -> JobState:
    """Human approval gate. The dashboard is the only caller in v1."""
    d = ApprovalDecision.model_validate(decision)          # reject junk before touching the graph
    row = session.scalar(select(TailoredResumeRow).where(
        TailoredResumeRow.user_id == user_id, TailoredResumeRow.job_id == job_id,
        TailoredResumeRow.status == "pending_approval"))
    if row is None:
        raise LookupError("nothing waiting for approval for this job")
    state = app.get_state(_cfg(row.thread_id))
    if state.next != ("approval",):
        raise RuntimeError("graph is not paused at the approval gate")
    try:
        app.invoke(Command(resume=d.model_dump()), _cfg(row.thread_id))
    except PacketError as e:   # decision is recorded; only the packet failed and can be retried
        return {**app.get_state(_cfg(row.thread_id)).values, "packet_error": str(e)}
    return app.get_state(_cfg(row.thread_id)).values


def build_packet(app, session, user_id: str, job_id: int) -> JobState:
    """(Re)run the packet step for the latest approved run of this job, resuming its graph thread."""
    row = session.scalar(select(TailoredResumeRow).where(
        TailoredResumeRow.user_id == user_id, TailoredResumeRow.job_id == job_id,
        TailoredResumeRow.status == "approved").order_by(TailoredResumeRow.id.desc()))
    if row is None:
        raise LookupError("no approved tailored resume for this job")
    state = app.get_state(_cfg(row.thread_id))
    if state.next == ("packet",):
        app.invoke(None, _cfg(row.thread_id))     # resume the checkpointed thread at the failed node
    elif "packet_path" not in state.values:
        raise RuntimeError("approved run has no pending packet step")
    return app.get_state(_cfg(row.thread_id)).values
