import json
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from cortex.agents.tracker import (
    TrackerError,
    followups,
    format_report,
    history,
    mark_applied,
    report,
    update_status,
)
from cortex.core.audit import AuditLog
from cortex.db.schema import ApplicationEventRow, JobRow, get_session
from cortex.review import applications, ready_to_apply

TODAY = date(2026, 11, 20)


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


def add(ses, n, status="to_apply", user="u1", **kw):
    j = JobRow(user_id=user, source="t", external_id=str(n), company=f"C{n}", title="Data Engineer", status=status, **kw)
    ses.add(j)
    ses.commit()
    return j


def at(d):
    return datetime(d.year, d.month, d.day, 10, tzinfo=UTC)


def test_mark_applied_sets_date_event_and_audit(ses, tmp_path):
    j = add(ses, 1, "approved")
    a = AuditLog(str(tmp_path / "a.log"))
    mark_applied(ses, "u1", j.id, note="via Workday", audit=a, now=at(date(2026, 11, 3)))
    assert (j.status, j.applied_date) == ("applied", date(2026, 11, 3))
    ev = history(ses, "u1", j.id)
    assert [(e.status, e.note) for e in ev] == [("applied", "via Workday")]
    rec = json.loads(a.path.read_text().splitlines()[-1])
    assert rec["actor"] == "tracker" and rec["details"]["status"] == "applied" and a.verify()


def test_explicit_applied_date(ses):
    j = add(ses, 1)
    mark_applied(ses, "u1", j.id, on=date(2026, 10, 30))
    assert j.applied_date == date(2026, 10, 30)


@pytest.mark.parametrize("start,new", [("to_apply", "interview"), ("applied", "offer"), ("rejected", "applied"),
                                       ("offer", "rejected"), ("interview", "applied")])
def test_forward_only(ses, start, new):
    j = add(ses, 1, start)
    with pytest.raises(TrackerError, match="can't go from"):
        update_status(ses, "u1", j.id, new)
    assert j.status == start and ses.scalars(select(ApplicationEventRow)).all() == []


def test_full_funnel_and_repeat_interviews(ses):
    j = add(ses, 1)
    for s in ("applied", "interview", "interview", "offer"):
        update_status(ses, "u1", j.id, s)
    assert [e.status for e in history(ses, "u1", j.id)] == ["applied", "interview", "interview", "offer"]


def test_other_users_job_refused_and_long_note(ses):
    j = add(ses, 1, user="someone_else")
    with pytest.raises(TrackerError, match="not found"):
        mark_applied(ses, "u1", j.id)
    mine = add(ses, 2)
    with pytest.raises(TrackerError, match="note too long"):
        mark_applied(ses, "u1", mine.id, note="x" * 301)


def test_followups(ses):
    quiet = add(ses, 1)
    mark_applied(ses, "u1", quiet.id, now=at(date(2026, 11, 1)))
    fresh = add(ses, 2)
    mark_applied(ses, "u1", fresh.id, now=at(date(2026, 11, 18)))
    imported = add(ses, 3, "applied", applied_date=date(2026, 10, 20))      # from the sheet: no events
    done = add(ses, 4)
    mark_applied(ses, "u1", done.id, now=at(date(2026, 10, 1)))
    update_status(ses, "u1", done.id, "rejected", now=at(date(2026, 10, 15)))
    f = followups(ses, "u1", TODAY, days=7)
    assert [(r["job_id"], r["days_quiet"]) for r in f] == [(imported.id, 31), (quiet.id, 19)]


def test_report_counts_rate_and_weeks(ses):
    for n, statuses, day in [(1, ["applied"], date(2026, 11, 17)), (2, ["applied", "interview"], date(2026, 11, 10)),
                             (3, ["applied", "interview", "rejected"], date(2026, 11, 3)),
                             (4, ["applied", "rejected"], date(2026, 11, 18))]:
        j = add(ses, n)
        for s in statuses:
            update_status(ses, "u1", j.id, s, now=at(day))
    add(ses, 5)                                                       # never applied: not counted
    r = report(ses, "u1", TODAY, weeks=3)
    assert r.applied_total == 4 and r.by_status == {"applied": 1, "interview": 1, "rejected": 2}
    assert r.responses == 2 and r.response_rate == 0.5                 # interviewed-then-rejected still counts
    assert r.per_week == [("2026-11-02", 1), ("2026-11-09", 1), ("2026-11-16", 2)]
    assert "response rate" in format_report(r, [])


def test_dashboard_queries(ses):
    a = add(ses, 1, "approved")
    add(ses, 2, "to_apply")
    b = add(ses, 3)
    mark_applied(ses, "u1", b.id, on=date(2026, 11, 5))
    assert [r["job_id"] for r in ready_to_apply(ses, "u1")] == [a.id]
    assert [r["job_id"] for r in applications(ses, "u1")] == [b.id]


def test_cli_track_and_report(ses, tmp_path, monkeypatch, capsys):
    from cortex.cli import main
    j = add(ses, 1, "approved")
    monkeypatch.setenv("CORTEX_DB_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("CORTEX_USER_ID", "u1")
    monkeypatch.chdir(tmp_path)                                         # audit log lands in tmp
    (tmp_path / "config.yaml").write_text("tracker: {followup_days: 7}\n")
    main(["applied", str(j.id), "--date", "2026-11-01", "--note", "Workday"])
    assert "-> applied on 2026-11-01" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not recorded"):
        main(["offer", str(j.id)])
    main(["report"])
    assert "applications: 1" in capsys.readouterr().out
