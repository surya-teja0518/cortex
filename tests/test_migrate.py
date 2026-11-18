"""One-time copy SQLite -> shared database. The target is SQLite here, or real Postgres under CORTEX_TEST_PG_URL."""
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from cortex.core.audit import AuditLog, _DbStore
from cortex.db.migrate import MigrateError, copy_database
from cortex.db.schema import ApplicationEventRow, InboxFindingRow, JobRow, get_session


def seed(url):
    s = get_session(url)
    jobs = [JobRow(user_id="u1", source="linkedin", external_id=str(i), company=f"C{i}", title="Data Engineer",
                   url="u", status="applied", applied_date=date(2026, 11, i + 1)) for i in range(3)]
    s.add_all(jobs)
    s.commit()
    s.add(ApplicationEventRow(user_id="u1", job_id=jobs[2].id, status="rejected", note="auto from Gmail",
                              at=datetime(2026, 11, 17, 10, tzinfo=UTC)))
    s.add(InboxFindingRow(user_id="u1", message_id="m0000001", found_at=datetime(2026, 11, 17, tzinfo=UTC)))
    s.commit()
    return [j.id for j in jobs]


def test_dry_run_then_copy_keeps_ids_counts_and_audit_chain(tmp_path):
    src, dst = f"sqlite:///{tmp_path}/src.db", f"sqlite:///{tmp_path}/dst.db"
    ids = seed(src)
    f = AuditLog(str(tmp_path / "audit.log"))
    f.log("tracker", "status", job_id=ids[2], status="rejected")
    dry = copy_database(src, dst, audit_file=str(f.path))
    assert not dry.done and dry.counts["jobs"] == 3 and get_session(dst).scalars(select(JobRow)).all() == []
    r = copy_database(src, dst, audit_file=str(f.path), dry_run=False)
    assert r.ok and r.copied["jobs"] == 3 and r.copied["application_events"] == 1 and r.audit_records == 1
    d = get_session(dst)
    assert sorted(j.id for j in d.scalars(select(JobRow))) == ids                      # ids kept
    assert d.scalar(select(ApplicationEventRow.job_id)) == ids[2]
    new = JobRow(user_id="u1", source="t", external_id="new", company="N", title="T", url="u", status="to_apply")
    d.add(new)
    d.commit()
    assert new.id > max(ids)                                                           # no id collision
    db = AuditLog.__new__(AuditLog)
    db._db = _DbStore(dst)
    db.log("tracker", "status", job_id=new.id)
    assert db.verify() and len(db.records()) == 2                                      # chain continues


def test_refuses_a_target_with_data(tmp_path):
    src, dst = f"sqlite:///{tmp_path}/src.db", f"sqlite:///{tmp_path}/dst.db"
    seed(src)
    seed(dst)
    with pytest.raises(MigrateError, match="refusing to merge"):
        copy_database(src, dst, dry_run=False)


def test_broken_audit_file_is_not_copied(tmp_path):
    src, dst = f"sqlite:///{tmp_path}/src.db", f"sqlite:///{tmp_path}/dst.db"
    seed(src)
    f = AuditLog(str(tmp_path / "audit.log"))
    f.log("a", "b", x=1)
    f.path.write_text(f.path.read_text().replace('"x": 1', '"x": 2'))
    with pytest.raises(MigrateError, match="fails verify"):
        copy_database(src, dst, audit_file=str(f.path), dry_run=False)


def test_pending_approval_is_reported(tmp_path):
    from cortex.db.schema import TailoredResumeRow
    src, dst = f"sqlite:///{tmp_path}/src.db", f"sqlite:///{tmp_path}/dst.db"
    ids = seed(src)
    s = get_session(src)
    s.add(TailoredResumeRow(user_id="u1", job_id=ids[0], thread_id="t1", status="pending_approval"))
    s.commit()
    assert copy_database(src, dst).pending_threads == [ids[0]]


def test_cli_never_prints_the_database_url(tmp_path, monkeypatch, capsys):
    from cortex import cli
    monkeypatch.setenv("CORTEX_DB_URL", "postgresql://user:SUPERSECRET@host.neon.tech/db")
    monkeypatch.setattr("cortex.db.migrate.copy_database",
                        lambda *a, **k: (_ for _ in ()).throw(MigrateError("target already has CORTEX data")))
    with pytest.raises(SystemExit) as e:
        cli.main(["migrate-db"])
    out = capsys.readouterr()
    assert "SUPERSECRET" not in str(e.value) + out.out + out.err
