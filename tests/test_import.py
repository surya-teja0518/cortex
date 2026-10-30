from pathlib import Path

import pytest
from sqlalchemy import select

from cortex.core.audit import AuditLog
from cortex.core.config import load_config
from cortex.db.schema import JobRow, get_session
from cortex.review import review_queue
from cortex.sources.synapse_import import import_files, read_tracker_csv

FIX = Path(__file__).parent / "fixtures" / "tracker_sample.csv"
CFG = load_config("config.yaml")


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


def job(ses, company, title=None, loc=None):
    q = select(JobRow).where(JobRow.company == company)
    if title:
        q = q.where(JobRow.title == title)
    if loc:
        q = q.where(JobRow.location.like(f"%{loc}%"))
    return ses.scalars(q).first()


def test_header_found_despite_blank_leading_row():
    rows = read_tracker_csv(FIX)
    assert rows[0]["company"] == "Milestone Technologies, Inc."
    assert "comp" in rows[0] and "stack_fit" in rows[0]


def test_import_counts_and_fields(ses, tmp_path):
    audit = AuditLog(str(tmp_path / "a.log"))
    s = import_files([FIX], ses, CFG, "u1", audit)
    assert (s.rows_read, s.inserted, s.skipped_blank) == (13, 12, 1)
    assert s.gated == 3                       # Lead title + min YOE 8 + Amazon III (min YOE 6)
    assert s.flagged == 1                     # injection phrase in notes
    assert s.duplicates == 1                  # Onit vs Onit India (same city)
    assert audit.verify()
    m = job(ses, "Milestone Technologies, Inc.")
    assert (m.min_yoe, m.max_yoe, m.track, m.synapse_score, m.synapse_priority) == (5, None, "data", 82, "P1")
    assert (m.score, m.band) == (72, "P2")    # 64+4+3+2-1; SYNAPSE says 82 (the gap is what `compare` surfaces)
    assert job(ses, "Acme").gate_reason and job(ses, "Globex").gate_reason
    assert job(ses, "Initech").sanitizer_flags
    assert job(ses, "Virtusa").track == "data" and job(ses, "Virtusa").external_id.startswith("h")


def test_idempotent_reimport_and_no_status_downgrade(ses):
    import_files([FIX], ses, CFG, "u1")
    n = len(ses.scalars(select(JobRow)).all())
    amazon = job(ses, "Amazon", "Data Engineer II, DSP Analytics")
    assert amazon.status == "applied"
    # re-import a copy where the same row regressed to To Apply
    import_files([FIX], ses, CFG, "u1")
    assert len(ses.scalars(select(JobRow)).all()) == n
    assert job(ses, "Amazon", "Data Engineer II, DSP Analytics").status == "applied"


def test_status_not_downgraded_by_later_batch(ses, tmp_path):
    import_files([FIX], ses, CFG, "u1")
    later = tmp_path / "later.csv"
    lines = FIX.read_text().splitlines()
    later.write_text("\n".join(l.replace(",Applied,", ",To Apply,") for l in lines))
    import_files([later], ses, CFG, "u1")
    assert job(ses, "Amazon", "Data Engineer II, DSP Analytics").status == "applied"
    assert job(ses, "Virtusa").status == "applied"


def test_dup_rules_in_db(ses):
    import_files([FIX], ses, CFG, "u1")
    onit, onit_in = job(ses, "Onit"), job(ses, "Onit India")
    assert onit.dup_of_id == onit_in.id and onit_in.dup_of_id is None
    ey = ses.scalars(select(JobRow).where(JobRow.company == "EY")).all()
    assert all(e.dup_of_id is None for e in ey)                  # Hyderabad vs Pune are distinct
    a3 = job(ses, "Amazon", "Data Engineer III, DSP Analytics")
    assert a3.dup_of_id is None


def test_user_isolation_and_review_queue(ses):
    import_files([FIX], ses, CFG, "u1")
    import_files([FIX], ses, CFG, "u2")
    assert len(review_queue(ses, "u3")) == 0
    q = review_queue(ses, "u1", min_synapse=0)
    companies = [r["company"] for r in q]
    assert "Acme" not in companies and "Globex" not in companies      # gated hidden by default
    assert "Onit" not in companies                                    # dup hidden
    assert "Amazon" not in companies                                  # III gated (YOE 6), II already applied
    assert all(r["synapse"] is None or r["synapse"] >= 0 for r in q)
    assert q[0]["synapse"] >= q[-1]["synapse"]                        # ordered by SYNAPSE score
    assert any(r["flagged"] for r in review_queue(ses, "u1"))         # Initech flagged, still visible
