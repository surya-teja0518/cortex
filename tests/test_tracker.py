import json
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from cortex.agents.tracker import (
    TrackerError,
    close_stale,
    followups,
    format_report,
    history,
    mark_applied,
    report,
    undo,
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
    f = followups(ses, "u1", TODAY, days=7)                                   # default window 7..30 days
    assert [(r["job_id"], r["days_quiet"]) for r in f] == [(quiet.id, 19)]
    f_all = followups(ses, "u1", TODAY, days=7, max_days=None)
    assert [(r["job_id"], r["days_quiet"]) for r in f_all] == [(imported.id, 31), (quiet.id, 19)]


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


# ---------- undo, no_response, close_stale ----------

def test_undo_applied_back_to_approved(ses, tmp_path):
    from cortex.db.schema import TailoredResumeRow
    j = add(ses, 1, "approved")
    ses.add(TailoredResumeRow(user_id="u1", job_id=j.id, thread_id="t1", status="approved"))
    ses.commit()
    a = AuditLog(str(tmp_path / "a.log"))
    mark_applied(ses, "u1", j.id, now=at(date(2026, 11, 4)))
    undo(ses, "u1", j.id, note="not submitted yet", audit=a)
    assert (j.status, j.applied_date) == ("approved", None)
    ev = history(ses, "u1", j.id)
    assert [e.status for e in ev] == ["applied", "undo"] and ev[1].note.startswith(f"reverted #{ev[0].id}")
    assert json.loads(a.path.read_text().splitlines()[-1])["action"] == "undo"
    mark_applied(ses, "u1", j.id)                                     # can apply for real afterwards
    assert j.status == "applied"


def test_undo_walks_back_one_step_at_a_time(ses):
    j = add(ses, 1)
    for s_ in ("applied", "interview", "rejected"):
        update_status(ses, "u1", j.id, s_)
    undo(ses, "u1", j.id)
    assert j.status == "interview"
    undo(ses, "u1", j.id)
    assert j.status == "applied"
    undo(ses, "u1", j.id)
    assert j.status == "to_apply"
    with pytest.raises(TrackerError, match="nothing to undo"):
        undo(ses, "u1", j.id)


def test_undo_refuses_imported_status(ses):
    j = add(ses, 1, "applied", applied_date=date(2026, 9, 1))
    with pytest.raises(TrackerError, match="imported"):
        undo(ses, "u1", j.id)
    assert j.status == "applied"


def test_undone_interview_not_counted_as_response(ses):
    j = add(ses, 1)
    update_status(ses, "u1", j.id, "applied")
    update_status(ses, "u1", j.id, "interview")
    undo(ses, "u1", j.id)
    assert report(ses, "u1", TODAY).responses == 0


def test_no_response_is_final_and_still_a_positive_for_eval(ses):
    from cortex.eval import POSITIVE_STATUSES
    j = add(ses, 1)
    mark_applied(ses, "u1", j.id)
    update_status(ses, "u1", j.id, "no_response")
    with pytest.raises(TrackerError, match="final status"):
        update_status(ses, "u1", j.id, "interview")
    assert {"applied", "rejected", "no_response"} <= set(POSITIVE_STATUSES)   # he chose these jobs


def test_close_stale_dry_run_then_apply(ses):
    old = add(ses, 1, "applied", applied_date=date(2026, 9, 1))
    recent = add(ses, 2)
    mark_applied(ses, "u1", recent.id, now=at(date(2026, 11, 15)))
    interviewing = add(ses, 3)
    mark_applied(ses, "u1", interviewing.id, now=at(date(2026, 9, 1)))
    update_status(ses, "u1", interviewing.id, "interview", now=at(date(2026, 9, 10)))
    preview = close_stale(ses, "u1", TODAY, days=30)
    assert [r["job_id"] for r in preview] == [old.id] and old.status == "applied"     # dry run changes nothing
    close_stale(ses, "u1", TODAY, days=30, dry_run=False)
    assert (old.status, recent.status, interviewing.status) == ("no_response", "applied", "interview")
    assert history(ses, "u1", old.id)[-1].note == "no reply for 80 days"


def test_cli_undo_no_response_close_stale(ses, tmp_path, monkeypatch, capsys):
    from cortex.cli import main
    j = add(ses, 1, "approved")
    old = add(ses, 2, "applied", applied_date=date(2026, 8, 1))
    monkeypatch.setenv("CORTEX_DB_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("CORTEX_USER_ID", "u1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("tracker: {followup_days: 7, stale_days: 30}\n")
    main(["applied", str(j.id)])                                        # the job-225 mistake
    main(["undo", str(j.id), "--note", "not submitted yet"])
    assert "back to to_apply" in capsys.readouterr().out                # no tailored row here -> to_apply
    with pytest.raises(SystemExit, match="not undone"):
        main(["undo", str(j.id)])
    main(["close-stale"])
    out = capsys.readouterr().out
    assert f"[{old.id}]" in out and "dry run" in out
    ses.expire_all()
    assert ses.get(JobRow, old.id).status == "applied"
    main(["close-stale", "--yes"])
    ses.expire_all()
    assert ses.get(JobRow, old.id).status == "no_response"
    k = add(ses, 3, "to_apply")
    main(["applied", str(k.id)])
    main(["no-response", str(k.id)])
    assert "-> no_response" in capsys.readouterr().out
