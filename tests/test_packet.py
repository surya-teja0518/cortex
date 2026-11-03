import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfReader
from sqlalchemy import select

from cortex.agents.packet import (
    ContactInfo,
    PacketBuilder,
    PacketError,
    fmt_month,
    load_contact,
    safe_url,
    slug,
)
from cortex.agents.tailor import FakeTailorProvider, Tailor
from cortex.agents.truth_guard import TruthGuard
from cortex.core.audit import AuditLog
from cortex.core.resume import TailoredResume, load_master
from cortex.db.schema import JobRow, PacketRow, get_session
from cortex.graph import (
    Deps,
    build_graph,
    build_packet,
    open_checkpointer,
    start_tailoring,
    submit_decision,
)

ROOT = Path(__file__).parent.parent
MASTER = load_master(ROOT / "master_resume.example.yaml")
CONTACT = ContactInfo(name="Alex Example", location="Hyderabad", email="alex@example.com")
GOOD = {
    "summary": "Data engineer with 4+ years on Azure and Oracle.",
    "skills": ["Python", "PySpark", "Kafka"],
    "roles": [{"employer": "Example Systems", "client": "Example Bank", "title": "Data Engineer",
               "start": "2024-07", "end": "present",
               "bullets": ["Ran 6 production Oracle databases behind Kafka streaming flows."]}],
    "certifications": ["DP-203"],
}


def pdf_text(path: Path) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)


def builder(tmp_path):
    return PacketBuilder(MASTER, CONTACT, root=tmp_path / "packets")


# ---------- helpers ----------

def test_slug_and_month():
    assert slug("Crowe Capabilities Center!") == "crowe-capabilities-center"
    assert slug("../../etc/passwd") == "etc-passwd" and slug("") == "x"
    assert (fmt_month("2024-07"), fmt_month("present")) == ("Jul 2024", "Present")


@pytest.mark.parametrize("url,ok", [("https://jobs.example.com/1", True), ("http://x.com/a", True),
                                    ("javascript:alert(1)", False), ("file:///etc/passwd", False),
                                    ("ftp://x.com", False), ("", False), (None, False), ("https://", False)])
def test_safe_url(url, ok):
    assert (safe_url(url) is not None) == ok


def test_folder_cannot_escape_root(tmp_path):
    b = builder(tmp_path)
    f = b.folder_for("../../root", 7, "../../../etc", "a/../../b")
    assert (tmp_path / "packets").resolve() in f.parents and ".." not in f.parts


def test_contact_file(tmp_path):
    with pytest.raises(PacketError, match="contact.example.yaml"):
        load_contact(tmp_path / "missing.yaml")
    (tmp_path / "c.yaml").write_text("name: A\nssn: 123\n")
    with pytest.raises(ValidationError):
        load_contact(tmp_path / "c.yaml")
    assert load_contact(ROOT / "contact.example.yaml").name == "Alex Example"


def test_real_contact_file_is_gitignored():
    assert "contact.yaml" in (ROOT / ".gitignore").read_text().splitlines()


# ---------- build ----------

def test_build_writes_pdf_and_job_md(tmp_path):
    res = builder(tmp_path).build(TailoredResume.model_validate(GOOD), user_id="u1", job_id=7,
                                  company="Acme Corp", title="Data Engineer", location="Hyderabad",
                                  url="https://jobs.acme.com/7")
    assert res.resume_pdf.read_bytes()[:5] == b"%PDF-"
    text = pdf_text(res.resume_pdf)
    for s in ("Alex Example", "alex@example.com", "Data Engineer", "Example Systems", "Client: Example Bank",
              "Jul 2024", "Present", "Ran 6 production Oracle databases", "DP-203", "B.Tech"):
        assert s in text, s
    md = res.job_md.read_text()
    assert "https://jobs.acme.com/7" in md and "never submits" in md
    assert res.folder.name == "7-acme-corp-data-engineer"


def test_markup_in_llm_text_is_escaped_not_rendered(tmp_path):
    # Markup with no URL, tech term or number gets past every Truth Guard rule, so only escaping stops it.
    evil = dict(GOOD, roles=[dict(GOOD["roles"][0], bullets=[
        "Ran 6 production Oracle databases <a href='#jump'>click</a> <font color='red'>BIG</font>"])])
    res = builder(tmp_path).build(TailoredResume.model_validate(evil), user_id="u1", job_id=8, company="A", title="B")
    raw = res.resume_pdf.read_bytes()
    assert b"/URI" not in raw and b"/Link" not in raw            # no link annotation of any kind
    assert "<a href='#jump'>click</a>" in pdf_text(res.resume_pdf)   # shown as plain text, not a link


def test_real_urls_are_blocked_before_rendering(tmp_path):
    evil = dict(GOOD, roles=[dict(GOOD["roles"][0], bullets=["Ran 6 databases, see https://evil.example"])])
    with pytest.raises(PacketError, match="link"):
        builder(tmp_path).build(TailoredResume.model_validate(evil), user_id="u1", job_id=8, company="A", title="B")


def test_bad_apply_link_dropped(tmp_path):
    res = builder(tmp_path).build(TailoredResume.model_validate(GOOD), user_id="u1", job_id=9, company="A",
                                  title="B", url="javascript:alert(1)")
    md = res.job_md.read_text()
    assert "javascript" not in md and "no valid http(s) link" in md


def test_build_refuses_if_truth_guard_fails(tmp_path):
    bad = dict(GOOD, skills=["Kubernetes"])
    with pytest.raises(PacketError, match="Truth Guard"):
        builder(tmp_path).build(TailoredResume.model_validate(bad), user_id="u1", job_id=1, company="A", title="B")
    assert not (tmp_path / "packets").exists()


# ---------- graph integration ----------

class NoWait:
    def wait(self):
        pass


@pytest.fixture
def env(tmp_path):
    db = f"sqlite:///{tmp_path}/c.db"
    ses = get_session(db)
    job = JobRow(user_id="u1", source="t", external_id="1", company="Acme", title="Data Engineer",
                 url="https://jobs.acme.com/1", status="to_apply")
    ses.add(job)
    ses.commit()
    return {"db": db, "job": job.id, "ckpt": str(tmp_path / "ck.db"), "audit": str(tmp_path / "a.log"), "tmp": tmp_path}


def app_for(env, responses=None, with_packet=True):
    ses = get_session(env["db"])
    a = AuditLog(env["audit"])
    provider = FakeTailorProvider(responses or [])
    deps = Deps(session_factory=lambda: ses, audit=a, guard=TruthGuard(MASTER),
                tailor=Tailor(provider, MASTER, NoWait(), a) if responses is not None else None,
                packet=builder(env["tmp"]) if with_packet else None)
    return build_graph(deps, open_checkpointer(env["ckpt"])), ses, provider


def test_approve_builds_packet(env):
    app, ses, _ = app_for(env, [GOOD])
    start_tailoring(app, ses, "u1", env["job"])
    out = submit_decision(app, ses, "u1", env["job"], {"decision": "approve"})
    assert out["status"] == "approved" and Path(out["packet_path"], "resume.pdf").exists()
    assert ses.scalars(select(PacketRow)).one().path == out["packet_path"]
    assert "packet" in [json.loads(x)["actor"] for x in Path(env["audit"]).read_text().splitlines()]


def test_reject_builds_no_packet(env):
    app, ses, _ = app_for(env, [GOOD])
    start_tailoring(app, ses, "u1", env["job"])
    out = submit_decision(app, ses, "u1", env["job"], {"decision": "reject"})
    assert "packet_path" not in out and ses.scalars(select(PacketRow)).all() == []
    with pytest.raises(LookupError):
        build_packet(app, ses, "u1", env["job"])


def test_packet_failure_keeps_approval_and_resumes_later(env):
    app, ses, provider = app_for(env, [GOOD], with_packet=False)       # contact.yaml missing, say
    start_tailoring(app, ses, "u1", env["job"])
    out = submit_decision(app, ses, "u1", env["job"], {"decision": "approve"})
    assert out["status"] == "approved" and "not configured" in out["packet_error"]
    assert ses.get(JobRow, env["job"]).status == "approved"            # approval recorded regardless
    # later, in a new process, with the packet builder configured:
    app2, ses2, provider2 = app_for(env, [], with_packet=True)          # empty: any LLM call would fail
    st = build_packet(app2, ses2, "u1", env["job"])
    assert Path(st["packet_path"], "resume.pdf").exists() and provider2.prompts == []
    assert len(provider.prompts) == 1                                    # tailoring never re-ran
    assert build_packet(app2, ses2, "u1", env["job"])["packet_path"] == st["packet_path"]   # idempotent


def test_cover_note_written_and_escaped(tmp_path):
    cover = ["Dear hiring team <b>at</b> Acme,", "At Example Systems I ran 6 production Oracle databases."]
    res = builder(tmp_path).build(TailoredResume.model_validate(dict(GOOD, cover_note=cover)), user_id="u1",
                                  job_id=5, company="Acme", title="DE")
    assert res.cover_txt.read_text().startswith("Dear hiring team <b>at</b> Acme,")
    text = pdf_text(res.cover_pdf)
    assert "Alex Example" in text and "<b>at</b>" in text and "6 production Oracle databases" in text
    assert "cover_note.pdf" in res.job_md.read_text()


def test_no_cover_note_no_files(tmp_path):
    res = builder(tmp_path).build(TailoredResume.model_validate(GOOD), user_id="u1", job_id=6, company="A", title="B")
    assert res.cover_pdf is None and not (res.folder / "cover_note.pdf").exists()
    assert "cover_note" not in res.job_md.read_text()
