"""Audit log in the database (Postgres: one chain for every device). Runs on SQLite by default and on real
Postgres when CORTEX_TEST_PG_URL is set."""
import json
import os
import threading

import pytest
from sqlalchemy import update

from cortex.core.audit import AuditLog, _DbStore
from cortex.db.schema import AuditRow, get_engine, get_session


def db_log(tmp_path):
    url = f"sqlite:///{tmp_path}/a.db"
    get_session(url)
    log = AuditLog.__new__(AuditLog)
    log._db = _DbStore(url)
    return log, url


def test_db_chain_verifies_and_tamper_fails(tmp_path):
    log, url = db_log(tmp_path)
    for i in range(5):
        log.log("tracker", "status", job_id=i, status="applied")
    assert log.verify() and [r["details"]["job_id"] for r in log.records()] == [0, 1, 2, 3, 4]
    with get_engine(url).begin() as c:
        c.execute(update(AuditRow).where(AuditRow.id == 3).values(details=json.dumps({"job_id": 99})))
    assert not log.verify()


def test_file_log_copies_in_and_still_verifies(tmp_path):
    f = AuditLog(str(tmp_path / "audit.log"))
    f.log("importer", "synapse_import", rows=3)
    f.log("tracker", "status", job_id=1)
    log, _ = db_log(tmp_path)
    for rec in f.records():
        log._db.append("", "", {}, rec=rec)
    log.log("tracker", "status", job_id=2)                        # chain continues after the copy
    assert log.verify() and len(log.records()) == 3
    with pytest.raises(ValueError, match="continue the chain"):
        log._db.append("", "", {}, rec=f.records()[0])             # a replayed record is refused


def test_postgres_url_picks_the_database(monkeypatch, tmp_path):
    monkeypatch.setenv("CORTEX_DB_URL", f"sqlite:///{tmp_path}/x.db")
    monkeypatch.chdir(tmp_path)
    assert AuditLog()._db is None                                   # SQLite setup: file log as before
    assert AuditLog(str(tmp_path / "f.log"))._db is None            # an explicit path always means the file


@pytest.mark.skipif(not os.getenv("CORTEX_TEST_PG_URL"), reason="needs CORTEX_TEST_PG_URL")
def test_concurrent_devices_never_fork_the_chain(tmp_path):
    log, _ = db_log(tmp_path)

    def device(n):
        for i in range(10):
            log.log("device", "x", n=n, i=i)
    ts = [threading.Thread(target=device, args=(n,)) for n in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    recs = log.records()
    assert len(recs) == 40 and log.verify()
