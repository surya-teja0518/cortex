"""SQLAlchemy schema. user_id on every table from day one (future multi-user)."""
from datetime import datetime
from sqlalchemy import String, Integer, Text, DateTime, UniqueConstraint, create_engine
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
    title: Mapped[str] = mapped_column(String(256))
    company: Mapped[str] = mapped_column(String(256))
    location: Mapped[str] = mapped_column(String(256), default="")
    url: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    track: Mapped[str | None] = mapped_column(String(16))
    score: Mapped[int | None] = mapped_column(Integer)
    band: Mapped[str | None] = mapped_column(String(2))
    status: Mapped[str] = mapped_column(String(24), default="new", index=True)
    gate_reason: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    batch: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ApprovalRow(Base):
    __tablename__ = "approvals"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    decision: Mapped[str] = mapped_column(String(12))  # approved | rejected
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_session(url: str = "sqlite:///data/cortex.db"):
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine)()
