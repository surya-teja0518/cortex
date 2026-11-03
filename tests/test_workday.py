from pathlib import Path

import pytest
from pydantic import ValidationError

from cortex.agents.packet import ContactInfo, PacketBuilder
from cortex.agents.workday import StandardAnswers, load_answers, mm_yyyy, render_sheet
from cortex.core.resume import TailoredResume, load_master

ROOT = Path(__file__).parent.parent
MASTER = load_master(ROOT / "master_resume.example.yaml")
CONTACT = ContactInfo(name="Alex Example", location="Hyderabad, India", email="alex@example.com",
                      phone="+91 00000 00000", linkedin="linkedin.com/in/alex-example")
TAILORED = TailoredResume.model_validate({
    "summary": "Data engineer with 4+ years on Azure and Oracle.",
    "skills": ["Python", "PySpark"],
    "roles": [{"employer": "Sample Consulting", "title": "Associate Data Engineer", "start": "2021-06",
               "end": "2024-06", "bullets": ["Migrated 120 tables to Snowflake."]}],
})


def sheet(answers=None, resume=TAILORED):
    return render_sheet(MASTER, resume, CONTACT, answers, company="Acme", title="Data Engineer")


def test_mm_yyyy():
    assert (mm_yyyy("2024-07"), mm_yyyy("present")) == ("07/2024", "")


def test_full_history_newest_first_with_tailored_bullets_where_available():
    s = sheet()
    assert s.index("Example Systems") < s.index("Sample Consulting")          # newest job first
    job2 = s[s.index("### Job 2"):]
    assert "• Migrated 120 tables to Snowflake." in job2                       # tailored bullet used
    assert "nightly load time by 35%" not in job2                               # ...instead of master's
    job1 = s[s.index("### Job 1"):s.index("### Job 2")]
    assert "Administered 6 production Oracle databases" in job1                 # untailored role: master bullets
    assert "• Client: Example Bank" in job1


def test_dates_and_current_role():
    s = sheet()
    job1 = s[s.index("### Job 1"):s.index("### Job 2")]
    assert "**From (MM/YYYY):** 07/2024" in job1 and "leave blank: current role" in job1
    assert "**I currently work here:** Yes" in job1
    assert "**To (MM/YYYY):** 06/2024" in s and "**I currently work here:** No" in s


def test_sections_education_certs_skills_links():
    s = sheet()
    for part in ("## 1. My Information", "**First name:** Alex", "**Last name:** Example", "## 3. Education",
                 "Example Institute of Technology", "05/2021", "DP-203", "## 5. Skills", "PySpark",
                 "linkedin.com/in/alex-example", "## 9. Attachments", "CORTEX never submits"):
        assert part in s, part


def test_missing_answers_never_guessed():
    s = sheet(answers=None)
    q = s[s.index("## 7."):s.index("## 8.")]
    assert "no answers.yaml" in q and q.count("_(answer manually)_") == 11
    assert "**GitHub:** _(answer manually)_" in s                              # absent contact field too


def test_answers_used_verbatim_and_extra_questions():
    a = StandardAnswers(notice_period="60 days", expected_ctc="Prefer to discuss",
                        extra={"Open to shifts?": "Yes, occasionally"})
    s = sheet(answers=a)
    assert "**Notice period:** 60 days" in s and "**Expected CTC:** Prefer to discuss" in s
    assert "**Open to shifts?:** Yes, occasionally" in s
    assert "**Current CTC:** _(answer manually)_" in s                         # still not guessed


def test_answers_file(tmp_path):
    assert load_answers(tmp_path / "none.yaml") is None
    (tmp_path / "a.yaml").write_text("notice_period: 30 days\nsalary_history: secret\n")
    with pytest.raises(ValidationError):                                        # unknown fields rejected
        load_answers(tmp_path / "a.yaml")
    assert load_answers(ROOT / "answers.example.yaml").notice_period == "60 days"
    assert "answers.yaml" in (ROOT / ".gitignore").read_text().splitlines()


def test_packet_writes_workday_sheet(tmp_path):
    res = PacketBuilder(MASTER, CONTACT, root=tmp_path / "p",
                        answers=StandardAnswers(notice_period="60 days")).build(
        TAILORED, user_id="u1", job_id=3, company="Acme", title="DE")
    text = res.workday_md.read_text()
    assert "**Notice period:** 60 days" in text and "workday.md" in res.job_md.read_text()
