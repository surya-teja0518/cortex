"""Master resume (source of truth for Truth Guard) and the tailored-resume contract.

The real `master_resume.yaml` is gitignored; `master_resume.example.yaml` documents the shape.
Contact details are deliberately NOT part of the schema: extra fields are forbidden, and any
string that looks like an email, phone number or profile URL is rejected.
"""
import re
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

YearMonth = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
EndDate = Annotated[str, StringConstraints(pattern=r"^(\d{4}-(0[1-9]|1[0-2])|present)$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_CONTACT = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("phone", re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}(?!\d)|\(\d{3}\)\s?\d{3}-\d{4}")),
    ("profile_url", re.compile(r"(linkedin\.com|github\.com)/", re.IGNORECASE)),
]


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _strings(v)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Skill(_Strict):
    name: Text
    aliases: list[Text] = Field(default_factory=list)


class Role(_Strict):
    employer: Text
    client: Text | None = None
    title: Text
    start: YearMonth
    end: EndDate
    bullets: list[Text] = Field(default_factory=list)
    metrics: list[Text] = Field(default_factory=list)   # copied exactly from the resume files
    skills: list[Text] = Field(default_factory=list)    # skills used in THIS role

    @model_validator(mode="after")
    def _ordered(self):
        if self.end != "present" and self.end < self.start:
            raise ValueError(f"{self.employer}: end {self.end} before start {self.start}")
        return self


class Project(_Strict):
    name: Text
    aliases: list[Text] = Field(default_factory=list)
    employer: Text | None = None                        # None = personal project
    description: Text
    metrics: list[Text] = Field(default_factory=list)
    skills: list[Text] = Field(default_factory=list)


class Certification(_Strict):
    name: Text
    code: Text | None = None
    issuer: Text


class Education(_Strict):
    institution: Text
    degree: Text
    start: YearMonth
    end: YearMonth


class MasterResume(_Strict):
    version: int = 1
    summary: Text
    skills: list[Skill]
    roles: list[Role] = Field(min_length=1)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _no_contact_details(cls, data):
        for s in _strings(data):
            for kind, rx in _CONTACT:
                if rx.search(s):
                    raise ValueError(f"master resume must not contain contact details ({kind} found)")
        return data

    @model_validator(mode="after")
    def _role_and_project_skills_declared(self):
        known = {norm(n) for s in self.skills for n in (s.name, *s.aliases)}
        for where, skills in [(r.employer, r.skills) for r in self.roles] + [(p.name, p.skills) for p in self.projects]:
            missing = [s for s in skills if norm(s) not in known]
            if missing:
                raise ValueError(f"{where}: skills {missing} not in top-level skills list")
        employers = {norm(r.employer) for r in self.roles}
        for p in self.projects:
            if p.employer and norm(p.employer) not in employers:
                raise ValueError(f"project {p.name}: employer {p.employer} not in roles")
        return self


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_master(path: str | Path = "master_resume.yaml") -> MasterResume:
    return MasterResume.model_validate(yaml.safe_load(Path(path).read_text()))


# ---- Tailor output contract (what Truth Guard checks) ----

class TailoredRole(_Strict):
    employer: Text
    client: Text | None = None
    title: Text
    start: YearMonth
    end: EndDate
    bullets: list[Text] = Field(default_factory=list)


class TailoredProject(_Strict):
    name: Text
    bullets: list[Text] = Field(default_factory=list)


class TailoredResume(_Strict):
    summary: Text
    skills: list[Text]
    roles: list[TailoredRole]
    projects: list[TailoredProject] = Field(default_factory=list)
    certifications: list[Text] = Field(default_factory=list)
