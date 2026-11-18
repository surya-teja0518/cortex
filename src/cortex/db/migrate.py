"""One-time copy of a CORTEX database (SQLite on the Mac) into another (Neon Postgres), plus the file audit log.

- Every table in the schema, ids kept (events, approvals and findings point at job ids), so nothing is renumbered.
- Refuses a target that already has CORTEX rows: no merging, no duplicates. Dry run by default.
- After the copy: row counts compared table by table, Postgres id sequences moved past the copied ids
  (or the next insert would collide), and the audit file copied record by record into the hash chain
  and verified.
- LangGraph checkpoints are NOT copied: finished threads don't need them. A thread still waiting for approval
  is listed so it can be decided before switching (or re-tailored after).
"""
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select, text

from cortex.db.schema import AuditRow, Base, TailoredResumeRow, get_engine


@dataclass
class CopyReport:
    counts: dict[str, int] = field(default_factory=dict)          # table -> rows in the source
    copied: dict[str, int] = field(default_factory=dict)          # table -> rows in the target after the copy
    audit_records: int = 0
    pending_threads: list[int] = field(default_factory=list)      # job ids still waiting for approval
    done: bool = False

    @property
    def ok(self) -> bool:
        return self.done and self.counts == self.copied


class MigrateError(RuntimeError):
    pass


def _tables():
    return [t for t in Base.metadata.sorted_tables if t.name != AuditRow.__tablename__]


def copy_database(src_url: str, dst_url: str, *, audit_file: str | None = None, dry_run: bool = True,
                  batch: int = 500) -> CopyReport:
    src, dst = get_engine(src_url), get_engine(dst_url)        # get_engine creates the target tables
    rep = CopyReport()
    with src.connect() as s:
        for t in _tables():
            rep.counts[t.name] = s.execute(select(func.count()).select_from(t)).scalar_one()
        rep.pending_threads = list(s.execute(select(TailoredResumeRow.job_id).where(
            TailoredResumeRow.status == "pending_approval")).scalars())
    with dst.connect() as d:
        busy = {t.name: n for t in _tables() + [AuditRow.__table__]
                if (n := d.execute(select(func.count()).select_from(t)).scalar_one())}
    if busy:
        raise MigrateError("target already has CORTEX data (" + ", ".join(f"{k}={v}" for k, v in busy.items())
                           + "): refusing to merge")
    if dry_run:
        return rep
    with src.connect() as s, dst.begin() as d:
        for t in _tables():
            rows = [dict(r._mapping) for r in s.execute(select(t).order_by(*t.primary_key.columns))]
            for i in range(0, len(rows), batch):
                d.execute(t.insert(), rows[i:i + batch])
        if dst.dialect.name == "postgresql":
            for t in _tables() + [AuditRow.__table__]:
                pk = list(t.primary_key.columns)
                if len(pk) == 1 and pk[0].autoincrement in (True, "auto") and pk[0].type.python_type is int:
                    d.execute(text(f"SELECT setval(pg_get_serial_sequence('{t.name}', '{pk[0].name}'), "
                                   f"COALESCE((SELECT MAX({pk[0].name}) FROM {t.name}), 0) + 1, false)"))
    with dst.connect() as d:
        for t in _tables():
            rep.copied[t.name] = d.execute(select(func.count()).select_from(t)).scalar_one()
    if audit_file and Path(audit_file).exists():
        from cortex.core.audit import AuditLog, _DbStore
        f = AuditLog(audit_file)
        if not f.verify():
            raise MigrateError(f"{audit_file} fails verify(): not copying a broken chain")
        store = _DbStore(dst_url)
        rep.audit_records = store.append_many(f.records())
        db = AuditLog.__new__(AuditLog)
        db._db = store
        if not db.verify():
            raise MigrateError("audit chain failed verify() after the copy")
    rep.done = True
    return rep
