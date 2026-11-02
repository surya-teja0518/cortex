from pathlib import Path

import pytest

from cortex.bench import format_bench, pick_jobs, run_bench
from cortex.cli import main
from cortex.core.audit import AuditLog
from cortex.core.resume import load_master
from cortex.db.schema import JobRow, get_session
from cortex.llm.providers import FakeLLM
from cortex.llm.router import PrivacyViolation

MASTER = load_master(Path(__file__).parent.parent / "master_resume.example.yaml")
GOOD = ('{"summary": "Data engineer with 4+ years on Azure and Oracle.", "skills": ["Python"], "roles": '
        '[{"employer": "Example Systems", "title": "Data Engineer", "start": "2024-07", "end": "present", '
        '"bullets": ["Ran 6 production Oracle databases."]}]}')
BAD = GOOD.replace('"Python"', '"Kubernetes"')


@pytest.fixture
def ses(tmp_path):
    s = get_session(f"sqlite:///{tmp_path}/b.db")
    rows = [
        JobRow(user_id="u1", source="t", external_id="1", company="A", title="DE", score=80, band="P1", status="to_apply"),
        JobRow(user_id="u1", source="t", external_id="2", company="B", title="DE", score=75, band="P2", status="applied"),
        JobRow(user_id="u1", source="t", external_id="3", company="C", title="Lead DE", score=99, band="P0",
               status="to_apply", gate_reason="title:lead"),
        JobRow(user_id="u1", source="t", external_id="4", company="D", title="DE", score=98, band="P0",
               status="to_apply", sanitizer_flags='["x"]'),
        JobRow(user_id="u1", source="t", external_id="5", company="E", title="DE", score=97, band="P0", status="closed"),
        JobRow(user_id="other", source="t", external_id="6", company="F", title="DE", score=99, band="P0", status="to_apply"),
    ]
    s.add_all(rows)
    s.commit()
    return s


def test_pick_jobs_excludes_gated_flagged_closed_and_other_users(ses):
    assert [j.company for j in pick_jobs(ses, "u1", 10)] == ["A", "B"]
    assert [j.company for j in pick_jobs(ses, "u1", 1)] == ["A"]


def test_bench_measures_pass_rate_attempts_and_time(ses, tmp_path):
    fast = FakeLLM([GOOD, GOOD], name="ollama:good", personal_data=True)
    slow = FakeLLM([BAD, BAD, BAD, BAD, GOOD], name="ollama:bad", personal_data=True)
    ticks = iter(range(0, 1000, 10))                         # each clock() call advances 10s
    runs = run_bench(ses, "u1", MASTER, [fast, slow], AuditLog(str(tmp_path / "a.log")), n_jobs=2,
                     clock=lambda: next(ticks))
    by = {(r.provider, r.job_id): r for r in runs}
    assert len(runs) == 4
    assert all(r.status == "passed" and r.attempts == 1 for r in runs if r.provider == "ollama:good")
    bad = [r for r in runs if r.provider == "ollama:bad"]
    assert [r.status for r in bad] == ["rejected", "passed"] and [r.attempts for r in bad] == [3, 2]
    assert bad[0].violation_kinds == ["skill"]
    assert all(r.seconds == 10 for r in by.values())
    out = format_bench(runs)
    assert "ollama:good" in out and "100%" in out and "50%" in out and "skillx1" in out


def test_bench_refuses_cloud_providers(ses, tmp_path):
    with pytest.raises(PrivacyViolation):
        run_bench(ses, "u1", MASTER, [FakeLLM([], name="gemini")], AuditLog(str(tmp_path / "a.log")))


def test_bench_cli_rejects_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_DB_URL", f"sqlite:///{tmp_path}/c.db")
    with pytest.raises(SystemExit, match="unknown provider"):
        main(["bench", "--providers", "gpt5"])


def test_format_bench_empty():
    assert "no runs" in format_bench([])
