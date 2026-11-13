"""SQLAlchemy schema. user_id on every table from day one (future multi-user)."""
from datetime import UTC, date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("user_id", "source", "external_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(32))
    external_id: Mapped[str] = mapped_column(String(128))
    batch: Mapped[str | None] = mapped_column(String(64))
    # identity
    company: Mapped[str] = mapped_column(String(256))
    title: Mapped[str] = mapped_column(String(256))
    url: Mapped[str] = mapped_column(Text, default="")
    location: Mapped[str] = mapped_column(String(256), default="")
    work_model: Mapped[str | None] = mapped_column(String(64))
    comp: Mapped[str | None] = mapped_column(String(128))
    posted: Mapped[date | None] = mapped_column(Date)
    date_added: Mapped[date | None] = mapped_column(Date)
    # requirements
    exp_raw: Mapped[str | None] = mapped_column(String(128))
    min_yoe: Mapped[int | None] = mapped_column(Integer)
    max_yoe: Mapped[int | None] = mapped_column(Integer)
    stack_fit: Mapped[str | None] = mapped_column(String(128))
    resume_version: Mapped[str | None] = mapped_column(String(64))
    # SYNAPSE's own verdict (kept for evaluation; never overwritten)
    track_raw: Mapped[str | None] = mapped_column(String(32))
    synapse_score: Mapped[int | None] = mapped_column(Integer)
    synapse_priority: Mapped[str | None] = mapped_column(String(4))
    # CORTEX deterministic verdict
    track: Mapped[str | None] = mapped_column(String(16))
    score: Mapped[int | None] = mapped_column(Integer)
    band: Mapped[str | None] = mapped_column(String(2))
    score_reasons: Mapped[str | None] = mapped_column(Text)   # JSON list
    gate_reason: Mapped[str | None] = mapped_column(Text)
    sanitizer_flags: Mapped[str | None] = mapped_column(Text)  # JSON list; flagged => manual review
    dup_of_id: Mapped[int | None] = mapped_column(Integer, index=True)
    # lifecycle (tracker)
    status: Mapped[str] = mapped_column(String(24), default="to_apply", index=True)
    status_raw: Mapped[str | None] = mapped_column(String(64))
    applied_date: Mapped[date | None] = mapped_column(Date)
    oa: Mapped[str | None] = mapped_column(String(64))
    interview: Mapped[str | None] = mapped_column(String(64))
    hr: Mapped[str | None] = mapped_column(String(64))
    offer: Mapped[str | None] = mapped_column(String(64))
    followup_date: Mapped[str | None] = mapped_column(String(32))
    notes: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))


class ApprovalRow(Base):
    __tablename__ = "approvals"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    decision: Mapped[str] = mapped_column(String(12))  # approved | rejected
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class SkipRow(Base):
    """Explicit "not applying" decisions from the review UI: the negative labels for eval."""
    __tablename__ = "skips"
    __table_args__ = (UniqueConstraint("user_id", "job_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class LLMScoreCacheRow(Base):
    """Validated LLM scores, keyed by job + provider + prompt hash, so eval reruns make no calls.
    Stores only the parsed result (no prompt, no job text). Failures are never cached."""
    __tablename__ = "llm_score_cache"
    __table_args__ = (UniqueConstraint("user_id", "job_id", "provider", "prompt_sha256"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    provider: Mapped[str] = mapped_column(String(32))
    prompt_sha256: Mapped[str] = mapped_column(String(64))
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class TailoredResumeRow(Base):
    """One tailoring run per row (one LangGraph thread). The dashboard reads pending rows from here;
    the graph's own state lives in the LangGraph checkpointer, keyed by thread_id."""
    __tablename__ = "tailored_resumes"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    thread_id: Mapped[str] = mapped_column(String(128), unique=True)
    # manual_review | guard_rejected | pending_approval | approved | rejected
    status: Mapped[str] = mapped_column(String(24), index=True)
    resume_json: Mapped[str | None] = mapped_column(Text)
    violations_json: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime)


class ApplicationEventRow(Base):
    """Append-only history of what happened to an application (tracker agent). New table: no migration."""
    __tablename__ = "application_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(24))
    note: Mapped[str | None] = mapped_column(String(300))
    at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class PacketRow(Base):
    """Where the packet for an approved run was written (resume.pdf + job.md). New table, so existing
    databases pick it up via create_all without a migration."""
    __tablename__ = "packets"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    thread_id: Mapped[str] = mapped_column(String(128), unique=True)
    path: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class DigestRunRow(Base):
    """One row per digest actually sent. max_job_id is the watermark: the next digest's "new jobs" are the
    ones above it. Counts only, no job text. New table: no migration."""
    __tablename__ = "digest_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    max_job_id: Mapped[int] = mapped_column(Integer, default=0)
    new_jobs: Mapped[int] = mapped_column(Integer, default=0)


class JobCheckRow(Base):
    """Last open/closed check per job (cortex check-open), so a small daily budget rotates through the queue
    instead of re-checking the same jobs. One row per job, overwritten. New table: no migration."""
    __tablename__ = "job_checks"
    __table_args__ = (UniqueConstraint("user_id", "job_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    verdict: Mapped[str] = mapped_column(String(16))
    checked_at: Mapped[date] = mapped_column(Date)


def get_session(url: str = "sqlite:///data/cortex.db"):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine)()
