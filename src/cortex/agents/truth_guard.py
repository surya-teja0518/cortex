"""Truth Guard: deterministic check of a tailored resume against master_resume.yaml.

Every claim must trace back to the master file:
  skills       -> skills section must name master skills; tech terms found in free text must be
                  skills of THAT role/project (no moving Kafka from one employer to another)
  employer     -> must be a master employer (and client, if given)
  dates        -> start/end must equal a master role for that employer exactly
  title        -> must equal that master role's title
  project      -> must be a master project (name or alias)
  metric       -> every number in free text must appear in the master text for the same scope
  certification-> must be a master certification (name or code)
Unsupported claims go back to the tailor with reasons, max 2 retries, then Reject (logged).

Known limits (deterministic, by design): paraphrased claims without numbers or known tech terms
are not semantically verified, and tech terms outside TECH_TERMS + master skills are not detected.
"""
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ValidationError

from cortex.core.audit import AuditLog
from cortex.core.resume import MasterResume, Project, Role, TailoredResume, norm

MAX_RETRIES = 2

# Common tech terms ("Go"/"R" left out: too ambiguous in prose). Any of these (or any master skill/alias) appearing in free text must be
# backed by the master skills for that scope. Extend as new false negatives show up.
TECH_TERMS = [
    "Python", "Java", "Scala", "SQL", "PL/SQL", "Golang", "Rust", "C++", "C#", "JavaScript", "TypeScript",
    "Spark", "PySpark", "Hadoop", "Hive", "Kafka", "Flink", "Airflow", "dbt", "Snowflake", "Databricks",
    "Azure Databricks", "Azure Data Factory", "ADF", "Synapse", "BigQuery", "Redshift", "Glue", "EMR",
    "AWS", "Azure", "GCP", "S3", "Lambda", "Kubernetes", "Docker", "OpenShift", "Terraform", "Jenkins",
    "Oracle", "PostgreSQL", "MySQL", "MongoDB", "Cassandra", "Redis", "Elasticsearch", "SQL Server",
    "Power BI", "Tableau", "Looker", "Flask", "Django", "FastAPI", "React", "Streamlit",
    "TensorFlow", "PyTorch", "scikit-learn", "LangChain", "LangGraph", "LLM", "RAG", "MLflow",
    "Delta Lake", "Iceberg", "GoldenGate", "Data Guard", "RAC", "RMAN",
]

_NUM = re.compile(
    r"(?<![A-Za-z]-)(?<![\w.])[$₹€£]?\d+(?:[.,]\d+)*"
    r"(?:\s?(?:%|x|\+|k|m|b|mn|bn|cr|million|billion|thousand|lakhs?|crores?|tb|gb|pb|hrs?|ms))?(?![\w%+])",
    re.IGNORECASE)

Kind = Literal["skill", "employer", "client", "title", "date", "project", "metric", "certification",
               "schema", "tailor_error"]


class Violation(BaseModel):
    kind: Kind
    claim: str
    where: str
    reason: str

    def feedback(self) -> str:
        return f"{self.kind}: '{self.claim}' in {self.where}: {self.reason}"


class GuardOutcome(BaseModel):
    status: Literal["passed", "rejected"]
    resume: TailoredResume | None
    attempts: int
    violations: list[Violation]   # from the last attempt (empty when passed)


def metric_tokens(text: str) -> set[str]:
    return {re.sub(r"[\s,]", "", m.group(0)).lower() for m in _NUM.finditer(text)}


def _term_regex(terms: set[str]) -> re.Pattern | None:
    if not terms:
        return None
    alts = sorted((re.escape(t) for t in terms), key=len, reverse=True)  # longest first
    return re.compile(r"(?<![\w+#.])(?:" + "|".join(alts) + r")(?![\w+#])", re.IGNORECASE)


class TruthGuard:
    def __init__(self, master: MasterResume, extra_terms: list[str] | None = None):
        self.m = master
        self._skill_canon: dict[str, str] = {}
        for s in master.skills:
            for n in (s.name, *s.aliases):
                self._skill_canon[norm(n)] = norm(s.name)
        vocab = {*TECH_TERMS, *(extra_terms or []), *(n for s in master.skills for n in (s.name, *s.aliases))}
        self._vocab_rx = _term_regex(vocab)
        self._all_tokens = metric_tokens(" ".join(self._texts_all()))

    # --- scopes ---
    def _canon(self, skills) -> set[str]:
        return {self._skill_canon.get(norm(s), norm(s)) for s in skills}

    def _projects_of(self, employer: str) -> list[Project]:
        return [p for p in self.m.projects if p.employer and norm(p.employer) == norm(employer)]

    def _texts_all(self) -> list[str]:
        out = [self.m.summary]
        for r in self.m.roles:
            out += [*r.bullets, *r.metrics]
        for p in self.m.projects:
            out += [p.description, *p.metrics]
        for e in self.m.education:
            out += [e.degree]
        return out

    def _role_scope(self, r: Role) -> tuple[set[str], set[str]]:
        projs = self._projects_of(r.employer)
        skills = self._canon([*r.skills, *(s for p in projs for s in p.skills)])
        texts = [*r.bullets, *r.metrics, *(t for p in projs for t in (p.description, *p.metrics))]
        return skills, metric_tokens(" ".join(texts))

    def _project_scope(self, p: Project) -> tuple[set[str], set[str]]:
        roles = [r for r in self.m.roles if p.employer and norm(r.employer) == norm(p.employer)]
        skills = self._canon([*p.skills, *(s for r in roles for s in r.skills)])
        texts = [p.description, *p.metrics, *(t for r in roles for t in (*r.bullets, *r.metrics))]
        return skills, metric_tokens(" ".join(texts))

    # --- free-text checks ---
    def _terms_in(self, text: str) -> list[str]:
        return [m.group(0) for m in self._vocab_rx.finditer(text)] if self._vocab_rx else []

    def _check_text(self, text: str, where: str, skills: set[str], tokens: set[str]) -> list[Violation]:
        v = []
        for term in self._terms_in(text):
            if self._skill_canon.get(norm(term), norm(term)) not in skills:
                v.append(Violation(kind="skill", claim=term, where=where,
                                   reason="not a master skill for this scope"))
        for tok in sorted(metric_tokens(text) - tokens):
            v.append(Violation(kind="metric", claim=tok, where=where,
                               reason="number not found in master resume for this scope"))
        return v

    # --- main check ---
    def check(self, t: TailoredResume) -> list[Violation]:
        v: list[Violation] = []
        all_skills = set(self._skill_canon.values())

        for s in t.skills:
            if norm(s) not in self._skill_canon:
                v.append(Violation(kind="skill", claim=s, where="skills", reason="not in master skills"))
        v += self._check_text(t.summary, "summary", all_skills, self._all_tokens)

        for i, tr in enumerate(t.roles):
            where = f"roles[{i}] {tr.employer}"
            cands = [r for r in self.m.roles if norm(r.employer) == norm(tr.employer)]
            if not cands:
                v.append(Violation(kind="employer", claim=tr.employer, where=where, reason="not a master employer"))
                continue
            match = next((r for r in cands if (r.start, r.end) == (tr.start, tr.end)), None)
            if match is None:
                have = ", ".join(f"{r.start}..{r.end}" for r in cands)
                v.append(Violation(kind="date", claim=f"{tr.start}..{tr.end}", where=where,
                                   reason=f"master dates for this employer: {have}"))
                continue
            if norm(tr.title) != norm(match.title):
                v.append(Violation(kind="title", claim=tr.title, where=where, reason=f"master title is '{match.title}'"))
            if tr.client and (match.client is None or norm(tr.client) != norm(match.client)):
                v.append(Violation(kind="client", claim=tr.client, where=where,
                                   reason=f"master client is '{match.client}'"))
            skills, tokens = self._role_scope(match)
            for j, b in enumerate(tr.bullets):
                v += self._check_text(b, f"{where} bullet {j}", skills, tokens)

        for i, tp in enumerate(t.projects):
            where = f"projects[{i}] {tp.name}"
            proj = next((p for p in self.m.projects
                         if norm(tp.name) in {norm(n) for n in (p.name, *p.aliases)}), None)
            if proj is None:
                v.append(Violation(kind="project", claim=tp.name, where=where, reason="not a master project"))
                continue
            skills, tokens = self._project_scope(proj)
            for j, b in enumerate(tp.bullets):
                v += self._check_text(b, f"{where} bullet {j}", skills, tokens)

        certs = {norm(x) for c in self.m.certifications for x in (c.name, c.code) if x}
        for c in t.certifications:
            if norm(c) not in certs:
                v.append(Violation(kind="certification", claim=c, where="certifications",
                                   reason="not a master certification"))
        return v


TailorFn = Callable[[list[str]], TailoredResume | dict]


def guarded_tailor(tailor: TailorFn, guard: TruthGuard, audit: AuditLog,
                   max_retries: int = MAX_RETRIES, **job_ref) -> GuardOutcome:
    """Call tailor(feedback) until Truth Guard passes: 1 attempt + max_retries, then reject.
    feedback is [] on the first call, then the previous attempt's violations as strings."""
    feedback: list[str] = []
    violations: list[Violation] = []
    for attempt in range(1, max_retries + 2):
        try:
            raw = tailor(feedback)
            resume = raw if isinstance(raw, TailoredResume) else TailoredResume.model_validate(raw)
        except ValidationError as e:
            resume = None
            violations = [Violation(kind="schema", claim=str(err["loc"]), where="tailor output",
                                    reason=err["msg"]) for err in e.errors()[:10]]
        except Exception as e:  # noqa: BLE001 — tailor (LLM) down; message may echo request details
            resume = None
            violations = [Violation(kind="tailor_error", claim=type(e).__name__, where="tailor", reason="tailor raised")]
        else:
            violations = guard.check(resume)
        audit.log("truth_guard", "check", attempt=attempt, passed=not violations,
                  violations=sorted({x.kind for x in violations}), n=len(violations), **job_ref)
        if not violations:
            return GuardOutcome(status="passed", resume=resume, attempts=attempt, violations=[])
        feedback = [x.feedback() for x in violations]

    audit.log("truth_guard", "rejected", attempts=max_retries + 1,
              failures=[f"{x.kind}:{x.claim}" for x in violations][:20], **job_ref)
    return GuardOutcome(status="rejected", resume=None, attempts=max_retries + 1, violations=violations)
