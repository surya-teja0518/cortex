from pathlib import Path

import pytest

from cortex.agents.packet import PacketBuilder, PacketError
from cortex.agents.scorer import ScoringInput
from cortex.agents.track_tailor import TrackTailor
from cortex.agents.truth_guard import TruthGuard
from cortex.core.skillfit import SkillMatcher, classify_job, domain_scores
from tests.test_packet import CONTACT
from tests.test_track_tailor import MASTER

ROOT = Path(__file__).parent.parent


def job(title, description=None):
    return ScoringInput(source="t", external_id="1", title=title, company="Acme", description=description)


# ---------- job classification (the resume picker) ----------

@pytest.mark.parametrize("title,desc,want", [
    ("Oracle Database Administrator", "RMAN, Data Guard, RAC, AWR/ASH, OEM, PL/SQL", "database"),
    ("Data Engineer", "PySpark, Databricks, ADF, Snowflake, Airflow pipelines", "data"),
    ("ML Engineer", "LLM, RAG, LangChain, scikit-learn", "ai"),
    ("Database Engineer", "PostgreSQL: backups, replication, patching, high availability", "database"),
    ("Data Engineer", "ETL in Spark; some ML feature pipelines", "data"),            # title outweighs a mention
    ("AI Engineer", "LLM agents and RAG over our data lake", "ai"),
    ("Software Engineer", "We value great people.", None),                          # no signal
])
def test_classify_job(title, desc, want):
    assert classify_job(title, desc) == want


def test_title_counts_three_times():
    assert domain_scores("Data Engineer", "") == {"database": 0, "data": 3, "ai": 0}


def test_tie_keeps_the_jobs_own_track():
    sm = SkillMatcher(MASTER)
    assert classify_job("Engineer", "pipelines and backups") is None
    assert sm.best_track("Engineer", "pipelines and backups", "oracle_dba") == "oracle_dba"
    assert sm.best_track("Oracle DBA", "", "data_engineer") == "oracle_dba"
    assert sm.best_track("ML Engineer", "LLM RAG", "data_engineer") == "data_engineer"   # no ai track: fallback


def test_skill_fit_score():
    sm = SkillMatcher(MASTER)
    f = sm.fit("Python, SQL, Kafka, Snowflake, Haskell")
    assert f.score is not None and 0 < f.score <= 100 and "haskell" not in f.matched
    assert sm.fit("We value great people").score is None                         # too few terms: no score


# ---------- original mode ----------

def test_original_mode_classifies_keeps_order_and_marks_original():
    tt = TrackTailor(MASTER, original=True)
    r = tt.build(job("Oracle DBA", "RMAN, Data Guard, backups, patching"), "data")   # tracker said data
    assert r.based_on == "original:oracle_dba"
    master_track = MASTER.tracks["oracle_dba"]
    assert [x.bullets for x in r.roles] == [x.bullets for x in master_track.roles]     # his order, untouched
    assert TruthGuard(MASTER).check(r) == []
    assert TrackTailor(MASTER).build(job("Oracle DBA", "RMAN"), "data").based_on == "track:data_engineer"


def builder(tmp_path, files: dict[str, bytes]):
    d = tmp_path / "resumes"
    d.mkdir(parents=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    return PacketBuilder(MASTER, CONTACT, root=tmp_path / "packets", originals_dir=d)


def test_packet_copies_the_original_pdf_byte_for_byte(tmp_path):
    pdf = b"%PDF-1.7\nSurya's own DBA resume\n%%EOF"
    b = builder(tmp_path, {"oracle_dba.pdf": pdf})
    r = TrackTailor(MASTER, original=True).build(job("Oracle DBA", "RMAN backups patching"), "database")
    out = b.build(r, user_id="u1", job_id=7, company="Acme", title="Oracle DBA")
    assert out.resume_pdf.read_bytes() == pdf
    assert out.workday_md.exists()                                                  # workday sheet still built


def test_missing_or_bad_original_is_a_clear_error(tmp_path):
    r = TrackTailor(MASTER, original=True).build(job("Oracle DBA", "RMAN backups patching"), "database")
    with pytest.raises(PacketError, match="oracle_dba.pdf not found"):
        builder(tmp_path / "a", {}).build(r, user_id="u1", job_id=1, company="A", title="T")
    with pytest.raises(PacketError, match="not a PDF"):
        builder(tmp_path / "b", {"oracle_dba.pdf": b"<html>"}).build(r, user_id="u1", job_id=1, company="A", title="T")


def test_track_name_cannot_escape_resumes_dir(tmp_path):
    b = builder(tmp_path, {})
    with pytest.raises(PacketError, match="invalid track"):
        b._copy_original("../../etc/passwd", tmp_path / "out.pdf")


def test_packet_keeps_your_own_file_name(tmp_path):
    from cortex.agents.packet import packet_resume
    name = "Surya_teja_vajjhala_OracleDBA_Resume.pdf"
    d = tmp_path / "resumes"
    d.mkdir()
    (d / name).write_bytes(b"%PDF-1.7 mine")
    b = PacketBuilder(MASTER, CONTACT, root=tmp_path / "packets", originals_dir=d, originals={"oracle_dba": name})
    r = TrackTailor(MASTER, original=True).build(job("Oracle DBA", "RMAN backups patching"), "database")
    out = b.build(r, user_id="u1", job_id=3, company="Acme", title="DBA")
    assert out.resume_pdf.name == name and out.resume_pdf.read_bytes() == b"%PDF-1.7 mine"
    assert not (out.folder / "resume.pdf").exists()
    assert name in out.job_md.read_text()
    assert packet_resume(out.folder) == out.resume_pdf                          # workday-fill attaches this one


@pytest.mark.parametrize("bad", ["../x.pdf", "a/b.pdf", "x.exe", "/etc/passwd.pdf", ".hidden.pdf", ""])
def test_original_file_names_validated(tmp_path, bad):
    b = PacketBuilder(MASTER, CONTACT, root=tmp_path / "p", originals_dir=tmp_path, originals={"oracle_dba": bad})
    with pytest.raises(PacketError, match="invalid resume file name"):
        b._original_path("oracle_dba")
