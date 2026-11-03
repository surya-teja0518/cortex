"""Workday copy sheet: every value a Workday application asks for, in Workday's page order, ready to paste.

Workday re-asks for the full work history on every company's tenant and usually mis-parses the resume.
This sheet removes the retyping; it does not touch Workday (no browser automation, nothing behind a login).

Sources, in order of trust:
- master_resume.yaml: the full work history, education, certifications (Workday wants every job)
- the approved tailored resume: title, bullets and skills for the roles it covers (already Truth-Guard checked)
- contact.yaml: name, email, phone, links
- answers.yaml: Surya's own standard answers (notice period, CTC, authorization...). Written by him, never by
  an LLM. A question with no answer in the file is listed as "answer manually", never guessed.
"""
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cortex.core.resume import MasterResume, TailoredResume, norm

Answer = str | None


class StandardAnswers(BaseModel):
    """answers.yaml: written by Surya, reused across applications. Gitignored."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    how_did_you_hear: Answer = None
    current_location: Answer = None
    willing_to_relocate: Answer = None
    preferred_locations: Answer = None
    notice_period: Answer = None
    earliest_start_date: Answer = None
    total_experience: Answer = None
    current_ctc: Answer = None
    expected_ctc: Answer = None
    work_authorization: Answer = None
    requires_sponsorship: Answer = None
    gender: Answer = None
    veteran_status: Answer = None
    disability_status: Answer = None
    extra: dict[str, str] = Field(default_factory=dict, max_length=50)   # any other recurring question -> answer


QUESTIONS = [  # (field, how Workday usually phrases it)
    ("how_did_you_hear", "How did you hear about us?"),
    ("current_location", "Current location"),
    ("willing_to_relocate", "Are you willing to relocate?"),
    ("preferred_locations", "Preferred work locations"),
    ("notice_period", "Notice period"),
    ("earliest_start_date", "Earliest start date"),
    ("total_experience", "Total years of experience"),
    ("current_ctc", "Current CTC"),
    ("expected_ctc", "Expected CTC"),
    ("work_authorization", "Are you legally authorized to work in the job's country?"),
    ("requires_sponsorship", "Will you require visa sponsorship?"),
]
DISCLOSURES = [("gender", "Gender"), ("veteran_status", "Veteran status"), ("disability_status", "Disability status")]


def load_answers(path: str | Path = "answers.yaml") -> StandardAnswers | None:
    """None if the file doesn't exist (the sheet then lists every question as 'answer manually')."""
    p = Path(path)
    return StandardAnswers.model_validate(yaml.safe_load(p.read_text()) or {}) if p.exists() else None


def mm_yyyy(ym: str) -> str:
    return "" if ym == "present" else f"{ym[5:7]}/{ym[:4]}"


def _field(label: str, value: str | None) -> str:
    v = " ".join((value or "").split())
    return f"- **{label}:** {v}" if v else f"- **{label}:** _(answer manually)_"


def render_sheet(master: MasterResume, resume: TailoredResume, contact, answers: StandardAnswers | None,
                 *, company: str, title: str) -> str:
    first, _, last = contact.name.partition(" ")
    out = [f"# Workday copy sheet: {' '.join(company.split())} | {' '.join(title.split())}", "",
           ("Paste each value into the matching Workday field, top to bottom. Review every page yourself before "
            "submitting. CORTEX never submits."), "",
           "## 1. My Information", _field("First name", first), _field("Last name", last),
           _field("Email", contact.email), _field("Phone", contact.phone), _field("City / location", contact.location),
           _field("How did you hear about us?", answers.how_did_you_hear if answers else None), "",
           "## 2. My Experience: Work Experience (newest first; add one entry per job)"]

    tailored = {(norm(r.employer), r.start, r.end): r for r in resume.roles}
    for i, role in enumerate(sorted(master.roles, key=lambda r: r.start, reverse=True), start=1):
        t = tailored.get((norm(role.employer), role.start, role.end))
        bullets = list(t.bullets) if t and t.bullets else role.bullets[:4]
        if role.client:
            bullets = [f"Client: {role.client}", *bullets]
        out += ["", f"### Job {i}", _field("Job Title", t.title if t else role.title), _field("Company", role.employer),
                _field("Location", contact.location),
                _field("I currently work here", "Yes" if role.end == "present" else "No"),
                _field("From (MM/YYYY)", mm_yyyy(role.start)),
                f"- **To (MM/YYYY):** {mm_yyyy(role.end)}" if role.end != "present" else "- **To:** (leave blank: current role)",
                "- **Role Description:**", "", "```", *[f"• {b}" for b in bullets], "```"]

    out += ["", "## 3. Education"]
    for ed in master.education:
        out += [_field("School or University", ed.institution), _field("Degree / Field of study", ed.degree),
                _field("From (MM/YYYY)", mm_yyyy(ed.start)), _field("To (MM/YYYY)", mm_yyyy(ed.end)), ""]

    if master.certifications:
        out += ["## 4. Certifications"]
        for c in master.certifications:
            out.append(_field("Certification", f"{c.name}" + (f" ({c.code})" if c.code else "") + f", {c.issuer}"))
        out.append("")

    out += ["## 5. Skills (type each into the Skills box)", "", "```", *resume.skills, "```", "",
            "## 6. Websites", _field("LinkedIn", contact.linkedin), _field("GitHub", contact.github), "",
            "## 7. Application Questions" + ("" if answers else " (no answers.yaml: copy answers.example.yaml and fill it in)")]
    out += [_field(q, getattr(answers, f) if answers else None) for f, q in QUESTIONS]
    for q, a in (answers.extra.items() if answers else []):
        out.append(_field(q, a))
    out += ["", "## 8. Voluntary Disclosures"]
    out += [_field(q, getattr(answers, f) if answers else None) for f, q in DISCLOSURES]
    out += ["", "## 9. Attachments", "- Resume: `resume.pdf` (this folder)",
            "- Cover letter: `cover_note.pdf`" if resume.cover_note else "- Cover letter: (none generated)", ""]
    return "\n".join(out)
