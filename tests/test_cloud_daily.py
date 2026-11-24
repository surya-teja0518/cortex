"""Cloud daily job (GitHub Actions on a PUBLIC repo): skills-only export instead of the master resume, and a
counts-only log."""
import json
import re
from pathlib import Path

import pytest
import yaml

from cortex.agents.scorer import candidate_profile
from cortex.core.skillfit import SkillMatcher
from cortex.daily import public_notes
from tests.test_track_tailor import MASTER

JOBS = [
    ("Data Engineer", "PySpark, Azure Databricks, ADF and Snowflake pipelines; Airflow; Kafka a plus"),
    ("Oracle DBA", "Oracle 19c RAC, RMAN backup and recovery, Data Guard, patching, AWR tuning, PL/SQL"),
    ("AI Engineer", "LLMs, RAG, LangGraph agents, Python, FastAPI, vector databases, evaluation"),
    ("Analyst", "Excel and stakeholder management"),
]


# ---------- skills export: the cloud scores exactly like the Mac, without the resume ----------

def test_exported_matcher_scores_and_picks_like_the_master():
    local = SkillMatcher(MASTER)
    cloud = SkillMatcher.from_export(json.loads(json.dumps(local.export())))     # through JSON, like the secret
    for title, desc in JOBS:
        assert cloud.fit(f"{title}\n{desc}") == local.fit(f"{title}\n{desc}")
        assert cloud.best_track(title, desc, "data_engineer") == local.best_track(title, desc, "data_engineer")


def test_export_holds_skills_only():
    blob = json.dumps({"profile": candidate_profile(MASTER), "matcher": SkillMatcher(MASTER).export()})
    forbidden = [r.employer for r in MASTER.roles] + [r.client for r in MASTER.roles if r.client] \
        + [p.name for p in MASTER.projects] + [b for r in MASTER.roles for b in r.bullets] \
        + [m for r in MASTER.roles for m in r.metrics]
    for f in forbidden:
        assert f not in blob, f
    assert not re.search(r"\d+\s*%|\d+\+?\s*(TB|GB|tables|databases)", blob)
    assert "@" not in blob


def test_cli_uses_the_skills_file_when_the_master_is_absent(tmp_path, monkeypatch):
    from cortex import cli
    from cortex.core.config import load_config
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORTEX_SKILLS_FILE", str(tmp_path / "skills.json"))
    cfg = load_config(str(Path(__file__).parent.parent / "config.yaml"))
    cfg["llm"]["scorer_profile"] = True
    assert cli._profile(cfg) is None                                              # nothing at all: generic
    (tmp_path / "skills.json").write_text(json.dumps({"profile": candidate_profile(MASTER),
                                                      "matcher": SkillMatcher(MASTER).export()}))
    assert cli._profile(cfg) == candidate_profile(MASTER)


def test_export_skills_command_writes_a_private_file(tmp_path, monkeypatch, capsys):
    from cortex import cli
    m = tmp_path / "master.yaml"
    m.write_text(yaml.safe_dump(MASTER.model_dump(mode="json")))
    out = tmp_path / "skills_profile.json"
    cli.main(["export-skills", "--master", str(m), "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["profile"] == candidate_profile(MASTER) and d["matcher"]["version"] == 1
    import os
    if os.name == "posix":          # Windows ignores chmod: the user-folder ACL protects the file there
        assert oct(out.stat().st_mode & 0o777) == "0o600"


# ---------- public log: counts only ----------

def test_public_notes_turn_gmail_details_into_counts():
    notes = ["gmail: marked rejected: Cisco | Database Administrator (undo: cortex undo 123)",
             "gmail: check yourself: possible rejected from Oaktree (email doesn't name the job title you applied for)",
             "gmail: check yourself: possible rejected from Amazon (only a weak signal)",
             "sheet sync: 0 new rows, 3 status cells written back", "LLM scoring: 98 scored, 2 kept rule scores"]
    out = public_notes(notes)
    assert out[0] == "gmail: 1 recorded, 2 to check yourself (details in your digest email)"
    assert not any(c in " ".join(out) for c in ("Cisco", "Oaktree", "Amazon", "Database Administrator", "123"))
    assert "sheet sync: 0 new rows, 3 status cells written back" in out


def test_public_mode_hides_tracebacks_and_details(monkeypatch, capsys):
    from cortex import cli
    monkeypatch.setenv("CORTEX_PUBLIC_LOG", "1")

    def boom(a, uid):
        raise RuntimeError("job 'Secret Corp | Staff Engineer' broke at https://x.test/job/1")
    monkeypatch.setattr(cli, "cmd_report", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["report"])
    err = capsys.readouterr().err
    assert e.value.code == 1 and err.strip() == "cortex failed: RuntimeError" and "Secret" not in err

    def stop(a, uid):
        raise SystemExit("not recorded: job 'Secret Corp' can't go from applied")
    monkeypatch.setattr(cli, "cmd_report", stop)
    with pytest.raises(SystemExit):
        cli.main(["report"])
    assert "Secret" not in capsys.readouterr().err


# ---------- the workflow itself ----------

def test_workflow_is_locked_down():
    wf = yaml.safe_load((Path(__file__).parent.parent / ".github/workflows/daily.yml").read_text())
    triggers = wf.get("on", wf.get(True))                                         # YAML 1.1 reads `on:` as True
    assert set(triggers) == {"schedule", "workflow_dispatch"}                     # no PR / fork triggers
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["daily"]
    assert job["env"]["CORTEX_PUBLIC_LOG"] == "1" and "surya-teja0518/cortex" in job["if"]
    text = (Path(__file__).parent.parent / ".github/workflows/daily.yml").read_text()
    assert "upload-artifact" not in text and "actions/cache" not in text and "pull_request" not in text
    assert "set -x" not in text and "cat " not in text                            # never echo the secret files


def test_vercel_serves_no_repo_files_statically():
    """2026-11-20: with the 'Other' preset Vercel published the whole repo (AGENTS.md, config.yaml, src/...) as
    static files in front of the app. Only public/ may be static, and it holds nothing but robots.txt."""
    root = Path(__file__).parent.parent
    cfg = json.loads((root / "vercel.json").read_text())
    assert cfg["outputDirectory"] == "public" and cfg["regions"] == ["cle1"]
    assert sorted(p.name for p in (root / "public").iterdir()) == ["robots.txt"]
    assert "Disallow: /" in (root / "public/robots.txt").read_text()


# ---------- once-per-day guard for the extra schedule slots (2026-11-20) ----------

def test_local_today_uses_ist_not_utc():
    from datetime import UTC, date, datetime

    from cortex.daily import local_today
    assert local_today({}, datetime(2026, 11, 20, 20, 0, tzinfo=UTC)) == date(2026, 11, 21)   # 01:30 IST next day
    assert local_today({"daily": {"timezone": "UTC"}}, datetime(2026, 11, 20, 20, 0, tzinfo=UTC)) == date(2026, 11, 20)


def test_guard_skips_done_days_retries_once_then_stops(tmp_path):
    from datetime import date

    from cortex.daily import already_done, record_run
    from cortex.db.schema import get_session
    ses = get_session(f"sqlite:///{tmp_path}/g.db")
    d = date(2026, 11, 20)
    assert already_done(ses, "u1", d) is None
    record_run(ses, "u1", d, ["apify"])
    assert already_done(ses, "u1", d) is None                       # one failure: a later slot retries
    record_run(ses, "u1", d, ["apify"])
    assert "failed runs today" in already_done(ses, "u1", d)        # two: stop, no email storm
    assert already_done(ses, "u2", d) is None and already_done(ses, "u1", date(2026, 11, 21)) is None
    record_run(ses, "u1", date(2026, 11, 21), [])
    assert "already finished" in already_done(ses, "u1", date(2026, 11, 21))


def test_cli_once_per_day_skips_second_slot_but_manual_runs_always(tmp_path, monkeypatch, capsys):
    import yaml
    from sqlalchemy import select

    from cortex.cli import main
    from cortex.db.schema import DailyRunRow, get_session
    monkeypatch.setenv("CORTEX_DB_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("CORTEX_USER_ID", "u1")
    monkeypatch.setenv("GMAIL_ADDRESS", "me@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "x")
    monkeypatch.chdir(tmp_path)
    cfg = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())
    cfg["digest"] = {**cfg.get("digest", {}), "send_when_empty": False}
    cfg["sources"] = {"ats": {"boards": []}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    args = ["daily", "--no-score", "--no-pull", "--no-sync", "--no-inbox", "--no-fetch"]
    main(args + ["--once-per-day"])                                 # first slot: runs (nothing new: no email)
    assert "cortex daily" in capsys.readouterr().out
    main(args + ["--once-per-day"])                                 # second slot: skipped
    assert "skipped (today's run already finished)" in capsys.readouterr().out
    main(args)                                                      # by hand: runs again
    assert "skipped" not in capsys.readouterr().out
    main(args + ["--dry-run", "--once-per-day"])                    # dry runs neither skip nor record
    rows = get_session(f"sqlite:///{tmp_path}/t.db").scalars(select(DailyRunRow)).all()
    assert len(rows) == 2 and all(r.ok for r in rows)


def test_empty_or_broken_skills_file_falls_back_quietly(tmp_path, monkeypatch, capsys):
    from cortex import cli
    from cortex.core.config import load_config
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "skills.json"
    monkeypatch.setenv("CORTEX_SKILLS_FILE", str(f))
    cfg = load_config(str(Path(__file__).parent.parent / "config.yaml"))
    cfg["llm"]["scorer_profile"] = True
    for text in ("", "not json", "[]"):
        f.write_text(text)
        assert cli._profile(cfg) is None                            # generic profile, no JSONDecodeError
    assert "empty or not JSON" in capsys.readouterr().err


def test_workflow_slots_use_the_guard_and_manual_runs_do_not():
    text = (Path(__file__).parent.parent / ".github/workflows/daily.yml").read_text()
    wf = yaml.safe_load(text)
    crons = [c["cron"] for c in wf.get("on", wf.get(True))["schedule"]]
    assert crons == ["43 5,8,12,16 * * *"]                          # 11:13 14:13 18:13 22:13 IST
    assert "(github.event_name == 'schedule' || inputs.once_per_day) && '--once-per-day'" in text
    inputs = wf.get("on", wf.get(True))["workflow_dispatch"]["inputs"]
    assert inputs["once_per_day"]["type"] == "boolean" and inputs["once_per_day"]["default"] is False
    assert wf["concurrency"]["cancel-in-progress"] is False          # overlapping slots queue, never run together
