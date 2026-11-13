"""Skill fit: how much of what a job description asks for does Surya's resume cover? Runs locally, no LLM.

Reuses Truth Guard's tech-term detection (master skills + their aliases + a known tech vocabulary), so "ADF",
"Azure Data Factory" and the like count as the same skill everywhere in CORTEX.

- `fit(job_text)`: 0-100 = share of the job's recognised tech terms that appear anywhere in the master resume
  (all tracks). None when the job text names fewer than MIN_TERMS terms (title-only rows): no evidence, no score.
- `best_track(title, description, fallback)`: which curated resume to send, by classifying the JOB (DBA / DE /
  AI signals, title weighted 3x); ties and no-signal jobs keep `fallback` (the job's own track).
The resume never leaves the Mac here; only counts and term names are returned.
"""
import re
from dataclasses import dataclass, field

from cortex.agents.truth_guard import TruthGuard
from cortex.core.resume import MasterResume, norm

MIN_TERMS = 3


@dataclass
class Fit:
    score: int | None
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _track_text(t) -> str:
    parts = [t.summary]
    for g in t.skill_groups:
        parts += g.skills
    for r in t.roles:
        parts += r.bullets
    for p in t.projects:
        parts += p.bullets
    return "\n".join(parts)


class SkillMatcher:
    def __init__(self, master: MasterResume):
        self._g = TruthGuard(master)
        self.tracks = {name: self.terms(_track_text(t)) for name, t in (master.tracks or {}).items()}
        everything = [s.name for s in master.skills] + [b for r in master.roles for b in r.bullets]
        self.have = set().union(self.terms("\n".join(everything)), *self.tracks.values())

    def terms(self, text: str) -> set[str]:
        return {self._g._skill_canon.get(norm(t), norm(t)) for t in self._g._terms_in(text or "")}

    def fit(self, job_text: str) -> Fit:
        want = self.terms(job_text)
        if len(want) < MIN_TERMS:
            return Fit(None)
        hit = want & self.have
        return Fit(round(100 * len(hit) / len(want)), sorted(hit), sorted(want - hit))

    def best_track(self, title: str, description: str, fallback: str) -> str:
        """Which of the three curated resumes fits this job. Classifies the JOB, not the resumes: Surya's three
        resumes list nearly the same skills (all share the same employers), so resume-vs-JD overlap can't tell
        them apart (3 of 5 test jobs picked the wrong resume, 2026-11-13). Domain signals in the title count 3x;
        ties and jobs with no signal keep `fallback` (the track the job title / tracker already gave)."""
        domain = classify_job(title, description)
        name = TRACK_FOR_DOMAIN.get(domain) if domain else None
        return name if name in self.tracks else fallback


TRACK_FOR_DOMAIN = {"database": "oracle_dba", "data": "data_engineer", "ai": "ai_ml"}
_SIGNALS = {
    "database": [r"\bdba\b", r"database administrat", r"\brman\b", r"data ?guard", r"\brac\b", r"\basm\b",
                 r"goldengate", r"exadata", r"\bawr\b", r"\boem\b", r"backup", r"recovery", r"patch(ing|es)?\b",
                 r"upgrades?\b", r"tablespace", r"oracle database", r"database (engineer|reliability|admin)",
                 r"high availability", r"replication", r"\bdb2\b", r"sql server"],
    "data": [r"data engineer", r"pipelines?\b", r"\betl\b", r"\belt\b", r"\bspark\b", r"pyspark", r"databricks",
             r"airflow", r"data ?lake", r"lakehouse", r"data ?warehous", r"\bdbt\b", r"snowflake", r"data factory",
             r"\badf\b", r"ingestion", r"\bglue\b", r"redshift", r"bigquery", r"big data", r"analytics engineer"],
    "ai": [r"machine learning", r"\bml\b", r"\bai\b", r"\bllms?\b", r"\brag\b", r"deep learning", r"\bnlp\b",
           r"gen ?ai", r"generative", r"model (training|deployment|serving)", r"mlops", r"data scientist",
           r"computer vision", r"langchain|langgraph", r"pytorch|tensorflow", r"prompt", r"agentic|\bagents?\b"],
}
_RX = {d: [re.compile(p, re.IGNORECASE) for p in ps] for d, ps in _SIGNALS.items()}


def domain_scores(title: str, description: str) -> dict[str, int]:
    """3 x distinct signals in the title + distinct signals in the description, per domain."""
    t, d = title or "", description or ""
    return {dom: sum(3 * bool(r.search(t)) + bool(r.search(d)) for r in rxs) for dom, rxs in _RX.items()}


def classify_job(title: str, description: str) -> str | None:
    """'database' | 'data' | 'ai', or None when no domain clearly leads (tie or no signal)."""
    sc = domain_scores(title, description)
    top = max(sc.values())
    leaders = [d for d, v in sc.items() if v == top]
    return leaders[0] if top > 0 and len(leaders) == 1 else None
