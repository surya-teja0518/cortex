"""Append-only, hash-chained audit log (tamper-evident).

Two stores, same records and hashes:
- a JSON-lines file (default `data/audit.log`): one machine.
- the `audit_log` table, when CORTEX_DB_URL is Postgres (Neon, 2026-11-19): Mac, Windows, the cloud daily job
  and the dashboard all append to ONE chain. A transaction-scoped advisory lock serialises appends so two
  devices can't fork the chain.
"""
import hashlib
import json
import os
import time
from pathlib import Path

_LOCK_KEY = 0x434F5254   # "CORT": the pg_advisory_xact_lock key for appends


def _hash(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | None = None, *, db_url: str | None = None):
        """path given -> that file. Otherwise the database when CORTEX_DB_URL (or db_url) is Postgres,
        else data/audit.log."""
        from cortex.db.schema import is_postgres
        url = None if path is not None else (db_url or os.getenv("CORTEX_DB_URL"))
        self._db = _DbStore(url) if url and is_postgres(url) else None
        if self._db is None:
            self.path = Path(path or "data/audit.log")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._prev = self._last_hash()

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "GENESIS"
        last = self.path.read_text().strip().splitlines()[-1]
        return json.loads(last)["hash"]

    def log(self, actor: str, action: str, **details) -> None:
        if self._db is not None:
            self._db.append(actor, action, details)
            return
        rec = {"ts": time.time(), "actor": actor, "action": action, "details": details, "prev": self._prev}
        rec["hash"] = _hash(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self._prev = rec["hash"]

    def records(self) -> list[dict]:
        if self._db is not None:
            return self._db.records()
        return [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []

    def verify(self) -> bool:
        prev = "GENESIS"
        for rec in self.records():
            h = rec.pop("hash")
            if rec["prev"] != prev or _hash(rec) != h:
                return False
            prev = h
        return True


class _DbStore:
    def __init__(self, url: str):
        from cortex.db.schema import get_engine
        self.engine = get_engine(url)

    def append(self, actor: str, action: str, details: dict, *, rec: dict | None = None) -> None:
        """rec: an existing record (file import) appended verbatim, after checking it continues the chain."""
        from sqlalchemy import select

        from cortex.db.schema import AuditRow
        with self.engine.begin() as c:
            if self.engine.dialect.name == "postgresql":
                c.exec_driver_sql(f"SELECT pg_advisory_xact_lock({_LOCK_KEY})")
            prev = c.execute(select(AuditRow.hash).order_by(AuditRow.id.desc()).limit(1)).scalar() or "GENESIS"
            if rec is None:
                rec = {"ts": time.time(), "actor": actor, "action": action, "details": details, "prev": prev}
                rec["hash"] = _hash(rec)
            elif rec["prev"] != prev:
                raise ValueError("record does not continue the chain")
            c.execute(AuditRow.__table__.insert().values(
                ts=rec["ts"], actor=rec["actor"], action=rec["action"],
                details=json.dumps(rec["details"]), prev=rec["prev"], hash=rec["hash"]))

    def records(self) -> list[dict]:
        from sqlalchemy import select

        from cortex.db.schema import AuditRow
        with self.engine.connect() as c:
            rows = c.execute(select(AuditRow).order_by(AuditRow.id)).all()
        return [{"ts": r.ts, "actor": r.actor, "action": r.action, "details": json.loads(r.details),
                 "prev": r.prev, "hash": r.hash} for r in rows]
