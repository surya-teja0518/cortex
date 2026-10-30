import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cortex.agents.truth_guard import TruthGuard, guarded_tailor, metric_tokens
from cortex.core.audit import AuditLog
from cortex.core.resume import MasterResume, TailoredResume, load_master

EXAMPLE = Path(__file__).parent.parent / "master_resume.example.yaml"
MASTER = load_master(EXAMPLE)
RAW = yaml.safe_load(EXAMPLE.read_text())


def good(**over) -> dict:
    d = {
        "summary": "Data engineer with 4+ years on Azure and Oracle.",
        "skills": ["Python", "PySpark", "Databricks", "Kafka", "Oracle"],
        "roles": [
            {"employer": "Example Systems", "client": "Example Bank", "title": "Data Engineer",
             "start": "2024-07", "end": "present",
             "bullets": ["Ran 6 production Oracle databases behind Apache Kafka streaming flows."]},
            {"employer": "Sample Consulting", "title": "Associate Data Engineer",
             "start": "2021-06", "end": "2024-06",
             "bullets": ["Moved 120 tables to Snowflake; nightly loads 35% faster with PySpark on Databricks."]},
        ],
        "projects": [{"name": "KPI Automation Platform", "bullets": ["Tracked 40 database KPIs in Flask."]}],
        "certifications": ["DP-203"],
    }
    d.update(over)
    return d


def check(d):
    return TruthGuard(MASTER).check(TailoredResume.model_validate(d))


def kinds(vs):
    return sorted((v.kind, v.claim) for v in vs)


def role(i, **kw):
    r = good()["roles"][i]
    r.update(kw)
    return r


# ---------- master schema ----------

def test_example_master_is_valid():
    assert MASTER.roles[0].employer == "Example Systems"


@pytest.mark.parametrize("field", ["email", "phone", "linkedin", "address"])
def test_contact_fields_forbidden(field):
    with pytest.raises(ValidationError):
        MasterResume.model_validate({**RAW, field: "x"})


@pytest.mark.parametrize("text", ["reach me at someone@example.com", "+91 98765 43210",
                                  "see linkedin.com/in/someone", "(555) 123-4567"])
def test_contact_details_in_any_string_rejected(text):
    with pytest.raises(ValidationError, match="contact details"):
        MasterResume.model_validate({**RAW, "summary": text})


def test_extra_field_in_nested_role_forbidden():
    bad = json.loads(json.dumps(RAW))
    bad["roles"][0]["salary"] = "secret"
    with pytest.raises(ValidationError):
        MasterResume.model_validate(bad)


def test_role_skill_must_be_declared():
    bad = json.loads(json.dumps(RAW))
    bad["roles"][0]["skills"].append("Kubernetes")
    with pytest.raises(ValidationError, match="Kubernetes"):
        MasterResume.model_validate(bad)


@pytest.mark.parametrize("start,end", [("2024-13", "present"), ("2024", "present"), ("2024-07", "2023-01")])
def test_bad_dates_rejected(start, end):
    bad = json.loads(json.dumps(RAW))
    bad["roles"][0].update(start=start, end=end)
    with pytest.raises(ValidationError):
        MasterResume.model_validate(bad)


def test_real_master_file_is_gitignored():
    gi = (Path(__file__).parent.parent / ".gitignore").read_text().splitlines()
    assert "master_resume.yaml" in gi


# ---------- metric extraction ----------

def test_metric_tokens():
    assert metric_tokens("Cut cost 35% for 120 tables, 3x faster, $1.5M saved, 10+ pipelines") == \
        {"35%", "120", "3x", "$1.5m", "10+"}
    assert metric_tokens("DP-203 on S3 and EC2") == set()   # identifiers are not metrics
    assert metric_tokens("8 million rows") == {"8million"}


# ---------- truth guard ----------

def test_faithful_resume_passes():
    assert check(good()) == []


def test_invented_skill_in_skills_section():
    assert kinds(check(good(skills=["Python", "Kubernetes"]))) == [("skill", "Kubernetes")]


def test_invented_skill_in_bullet():
    r = role(1, bullets=["Built PySpark jobs on Kubernetes."])
    assert kinds(check(good(roles=[good()["roles"][0], r]))) == [("skill", "Kubernetes")]


@pytest.mark.parametrize("bullet", ["Built PySpark/Kafka jobs.", "Kafka-based ingestion in PySpark.",
                                    "Streaming (Kafka) with PySpark."])
def test_skill_detected_next_to_punctuation(bullet):
    r = role(1, bullets=[bullet])
    assert kinds(check(good(roles=[good()["roles"][0], r]))) == [("skill", "Kafka")]


def test_terms_not_false_flagged():
    g = TruthGuard(MASTER)
    assert g._terms_in("Go-live in two weeks; PL/SQL tuning") == ["PL/SQL"]


def test_skill_moved_to_wrong_employer():
    # Kafka is real, but only at Example Systems; claiming it at Sample Consulting is false.
    r = role(1, bullets=["Built Kafka streaming for Sample Health."])
    assert kinds(check(good(roles=[good()["roles"][0], r]))) == [("skill", "Kafka")]


def test_unknown_employer():
    r = role(0, employer="Google")
    assert kinds(check(good(roles=[r]))) == [("employer", "Google")]


@pytest.mark.parametrize("start,end", [("2024-01", "present"), ("2024-07", "2025-12")])
def test_wrong_dates(start, end):
    vs = check(good(roles=[role(0, start=start, end=end)]))
    assert [v.kind for v in vs] == ["date"] and "2024-07..present" in vs[0].reason


def test_inflated_title():
    assert kinds(check(good(roles=[role(0, title="Senior Data Engineer")]))) == [("title", "Senior Data Engineer")]


def test_wrong_client():
    assert kinds(check(good(roles=[role(0, client="Other Bank")]))) == [("client", "Other Bank")]


def test_unknown_project():
    vs = check(good(projects=[{"name": "Fraud Detection ML", "bullets": []}]))
    assert kinds(vs) == [("project", "Fraud Detection ML")]


@pytest.mark.parametrize("bullet,bad", [
    ("Migrated 150 tables to Snowflake.", "150"),           # inflated
    ("Nightly loads 40% faster with PySpark.", "40%"),       # changed
    ("Migrated 120 tables, saving $2M.", "$2m"),            # invented
])
def test_metric_not_in_master(bullet, bad):
    vs = check(good(roles=[good()["roles"][0], role(1, bullets=[bullet])]))
    assert kinds(vs) == [("metric", bad)]


def test_metric_moved_to_wrong_employer():
    # "120" is real, but belongs to Sample Consulting, not Example Systems.
    vs = check(good(roles=[role(0, bullets=["Ran 120 Oracle databases."])]))
    assert kinds(vs) == [("metric", "120")]


def test_summary_metric_must_exist():
    assert kinds(check(good(summary="Data engineer with 7+ years."))) == [("metric", "7+")]


def test_unknown_certification():
    assert kinds(check(good(certifications=["DP-203", "AWS SAA"]))) == [("certification", "AWS SAA")]


def test_reports_every_failing_claim():
    vs = check(good(skills=["Rust"], certifications=["CKA"], roles=[role(0, title="Lead")]))
    assert kinds(vs) == [("certification", "CKA"), ("skill", "Rust"), ("title", "Lead")]


# ---------- retry loop ----------

class Tailor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.feedback = []

    def __call__(self, feedback):
        self.feedback.append(feedback)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def run(outputs, tmp_path):
    t = Tailor(outputs)
    audit = AuditLog(str(tmp_path / "a.log"))
    out = guarded_tailor(t, TruthGuard(MASTER), audit, user_id="u1", job_id=7)
    recs = [json.loads(x) for x in audit.path.read_text().splitlines()]
    return out, t, recs, audit


def test_passes_first_try(tmp_path):
    out, t, recs, _ = run([good()], tmp_path)
    assert (out.status, out.attempts) == ("passed", 1) and out.resume is not None
    assert t.feedback == [[]] and recs[0]["details"]["passed"] is True


def test_fixed_on_retry_gets_failure_reason(tmp_path):
    out, t, _, _ = run([good(skills=["Kubernetes"]), good()], tmp_path)
    assert (out.status, out.attempts) == ("passed", 2)
    assert "Kubernetes" in t.feedback[1][0] and "skill" in t.feedback[1][0]


def test_rejected_after_two_retries(tmp_path):
    bad = good(roles=[role(0, title="Principal Engineer")])
    out, t, recs, audit = run([bad, bad, bad, good()], tmp_path)
    assert (out.status, out.attempts, out.resume) == ("rejected", 3, None)
    assert len(t.feedback) == 3 and len(t.outputs) == 1          # 4th never requested
    assert kinds(out.violations) == [("title", "Principal Engineer")]
    assert recs[-1]["action"] == "rejected" and "title:Principal Engineer" in recs[-1]["details"]["failures"]
    assert audit.verify()


def test_schema_invalid_output_counts_as_failure(tmp_path):
    out, t, _, _ = run([{"summary": "x", "skills": [], "roles": [], "salary": 1}, good()], tmp_path)
    assert (out.status, out.attempts) == ("passed", 2)
    assert t.feedback[1][0].startswith("schema:")


def test_tailor_exception_counts_as_failure_and_no_message_logged(tmp_path):
    err = RuntimeError("api key sk-SUPERSECRET")
    out, _, _, audit = run([err, err, err], tmp_path)
    assert out.status == "rejected" and out.violations[0].kind == "tailor_error"
    assert "sk-SUPERSECRET" not in audit.path.read_text()
