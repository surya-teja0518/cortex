import json

import pytest
from sqlalchemy import select

from cortex.agents.scorer import FakeProvider, LLMScorer
from cortex.cli import main
from cortex.core.audit import AuditLog
from cortex.db.schema import JobRow, LLMScoreCacheRow, get_session
from cortex.eval import format_report, pct_rank, run_eval
from cortex.review import record_skip, review_queue, skipped_job_ids

BANDS = {"P0": 85, "P1": 78, "P2": 70}


class NoWait:
    def wait(self):
        pass


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


def add(ses, n, score, band, status="to_apply", user="u1", **kw):
    r = JobRow(user_id=user, source="t", external_id=str(n), company=f"C{n}", title="Data Engineer",
               track="data", score=score, band=band, score_reasons=json.dumps([f"base {score}"]),
               status=status, **kw)
    ses.add(r)
    ses.commit()
    return r


def llm(score):
    return json.dumps({"score": score, "reasons": ["fit"]})


def scorer(responses, tmp_path):
    p = FakeProvider(responses)
    return LLMScorer(p, NoWait(), AuditLog(str(tmp_path / "a.log")), BANDS), p


def test_pct_rank_midrank():
    assert pct_rank(80, [70, 80, 90, 100]) == 37.5
    assert pct_rank(50, [50, 50]) == 50.0


def test_eval_deterministic_vs_llm(ses, tmp_path):
    # Deterministic misses the applied job (P2); the LLM ranks it top.
    a = add(ses, 1, 72, "P2", status="applied")
    add(ses, 2, 86, "P0")
    add(ses, 3, 80, "P1")
    add(ses, 4, 60, "P3")
    sc, p = scorer([llm(90), llm(70), llm(65), llm(40)], tmp_path)
    rep = run_eval(ses, "u1", sc)
    det, lm = rep.metrics
    assert (rep.population, rep.positives) == (4, 1)
    assert (det.recall_top, det.mean_pct_rank) == (0.0, 37.5)
    assert (lm.recall_top, lm.mean_pct_rank, lm.unscored) == (1.0, 87.5, 0)
    assert len(p.prompts) == 4 and rep.llm_jobs_scored_live == 4
    assert a.id in {r.job_id for r in ses.scalars(select(LLMScoreCacheRow))}


def test_eval_cache_makes_rerun_free(ses, tmp_path):
    add(ses, 1, 72, "P2", status="interview")
    add(ses, 2, 86, "P0")
    sc, _ = scorer([llm(90), llm(70)], tmp_path)
    run_eval(ses, "u1", sc)
    sc2, p2 = scorer([], tmp_path)            # empty queue: any live call would raise
    rep = run_eval(ses, "u1", sc2)
    assert p2.prompts == [] and rep.llm_calls_cached == 2 and rep.llm_jobs_scored_live == 0
    assert rep.metrics[1].recall_top == 1.0


def test_eval_cache_invalidated_when_fields_change(ses, tmp_path):
    r = add(ses, 1, 72, "P2", status="applied")
    sc, _ = scorer([llm(90)], tmp_path)
    run_eval(ses, "u1", sc)
    r.notes = "now has notes"
    ses.commit()
    sc2, p2 = scorer([llm(50)], tmp_path)
    run_eval(ses, "u1", sc2)
    assert len(p2.prompts) == 1


def test_eval_failures_not_cached_and_counted_unscored(ses, tmp_path):
    add(ses, 1, 72, "P2", status="applied")
    sc, _ = scorer(["junk", "junk"], tmp_path)
    rep = run_eval(ses, "u1", sc)
    assert rep.metrics[1].unscored == 1 and rep.metrics[1].recall_top == 0.0  # used deterministic P2
    assert ses.scalars(select(LLMScoreCacheRow)).all() == []


def test_eval_excludes_gated_dups_other_users(ses, tmp_path):
    add(ses, 1, 80, "P1", status="applied")
    add(ses, 2, 90, "P0", status="applied", gate_reason="title:lead")
    add(ses, 3, 90, "P0", status="offer", dup_of_id=1)
    add(ses, 4, 90, "P0", status="applied", user="someone_else")
    rep = run_eval(ses, "u1")
    assert (rep.population, rep.positives, rep.positives_gated) == (1, 1, 1)
    assert len(rep.metrics) == 1


def test_report_states_precision_needs_negatives(ses):
    add(ses, 1, 80, "P1", status="applied")
    out = format_report(run_eval(ses, "u1"))
    assert "PRECISION NOT REPORTED" in out and "negatives" in out and "recall@P0-P1" in out


def test_no_positives_reports_na(ses):
    add(ses, 1, 80, "P1")
    out = format_report(run_eval(ses, "u1"))
    assert "n/a" in out


def test_cli_eval_runs(ses, tmp_path, monkeypatch, capsys):
    add(ses, 1, 80, "P1", status="applied")
    monkeypatch.setenv("CORTEX_DB_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("CORTEX_USER_ID", "u1")
    main(["eval"])
    assert "deterministic" in capsys.readouterr().out


# --- skip decisions (negative labels) ---

def test_record_skip_and_queue_hides_it(ses, tmp_path):
    a = add(ses, 1, 80, "P1")
    b = add(ses, 2, 75, "P2")
    audit = AuditLog(str(tmp_path / "a.log"))
    assert record_skip(ses, "u1", a.id, audit) is True
    assert record_skip(ses, "u1", a.id, audit) is False         # idempotent
    assert skipped_job_ids(ses, "u1") == {a.id}
    assert [r["id"] for r in review_queue(ses, "u1")] == [b.id]
    assert {r["id"] for r in review_queue(ses, "u1", include_skipped=True)} == {a.id, b.id}
    rec = json.loads(audit.path.read_text().splitlines()[-1])
    assert rec["action"] == "skip" and rec["details"]["job_id"] == a.id and audit.verify()


def test_skip_has_user_and_timestamp(ses):
    from cortex.db.schema import SkipRow
    a = add(ses, 1, 80, "P1")
    record_skip(ses, "u1", a.id)
    row = ses.scalars(select(SkipRow)).one()
    assert row.user_id == "u1" and row.decided_at is not None


def test_cannot_skip_other_users_or_missing_job(ses):
    other = add(ses, 1, 80, "P1", user="someone_else")
    assert record_skip(ses, "u1", other.id) is False
    assert record_skip(ses, "u1", 9999) is False
    assert skipped_job_ids(ses, "u1") == set() and skipped_job_ids(ses, "someone_else") == set()


def test_skips_counted_in_eval(ses):
    a = add(ses, 1, 80, "P1")
    record_skip(ses, "u1", a.id)
    assert run_eval(ses, "u1").skips == 1
