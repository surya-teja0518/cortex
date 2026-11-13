"""Workday fill helper: types Surya's application data into the Workday form he has open. NEVER submits.

Approved by Surya on 2026-11-13 (AGENTS.md, WORKDAY FILL HELPER), with these rules enforced in code:
- He opens the browser session and logs in himself. CORTEX never sees, types or stores a password, and the
  browser profile is throwaway (no saved cookies).
- It only TYPES into empty text fields (and attaches resume.pdf to the resume upload input). It never clicks:
  no Next, no Save, no Submit, no checkboxes, no dropdowns. He reviews every page and clicks every button.
- A field he edits or clears after CORTEX filled it is never re-filled.
- It fills only on Workday hosts (*.myworkdayjobs.com / *.myworkday.com / *.workday.com), so a redirect to some
  other site can't harvest his details.
- Values come from the same sources as workday.md: contact.yaml, answers.yaml, master_resume.yaml and the
  APPROVED tailored resume. Values are never logged; the audit log gets counts only.

Split in two so the decisions are testable without a browser:
  plan_fills(fields, plan, fmap, done)  pure: which value goes into which field
  run(page, ...)                        the loop that asks the page for fields and types the planned values
"""
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import yaml

from cortex.agents.workday import StandardAnswers
from cortex.core.resume import MasterResume, TailoredResume, norm

WORKDAY_HOST_SUFFIXES = (".myworkdayjobs.com", ".myworkday.com", ".workday.com")


def is_workday(url: str) -> bool:
    u = urlparse(url or "")
    host = (u.hostname or "").lower()
    return u.scheme == "https" and any(host.endswith(s) for s in WORKDAY_HOST_SUFFIXES)


# ---------- what to type ----------

@dataclass
class Entry:
    fields: dict[str, str]                       # key -> value (title, company, ...)
    start: tuple[str, str] | None = None         # (MM, YYYY)
    end: tuple[str, str] | None = None


@dataclass
class FillPlan:
    simple: dict[str, str]
    jobs: list[Entry]
    education: list[Entry]
    resume_pdf: Path | None = None


def _ym(ym: str) -> tuple[str, str] | None:
    return None if ym == "present" else (ym[5:7], ym[:4])


def build_plan(master: MasterResume, resume: TailoredResume, contact, answers: StandardAnswers | None,
               resume_pdf: Path | None = None) -> FillPlan:
    first, _, last = contact.name.partition(" ")
    simple = {"first_name": first, "last_name": last, "email": contact.email or "", "phone": contact.phone or "",
              "city": contact.location or "", "linkedin": contact.linkedin or ""}
    tailored = {(norm(r.employer), r.start, r.end): r for r in resume.roles}
    jobs = []
    for role in sorted(master.roles, key=lambda r: r.start, reverse=True):       # Workday wants every job
        t = tailored.get((norm(role.employer), role.start, role.end))
        bullets = list(t.bullets) if t and t.bullets else role.bullets[:4]
        if role.client:
            bullets = [f"Client: {role.client}", *bullets]
        jobs.append(Entry({"title": t.title if t else role.title, "company": role.employer,
                           "location": contact.location or "", "description": "\n".join(f"• {b}" for b in bullets)},
                          _ym(role.start), _ym(role.end)))
    edu = [Entry({"school": e.institution}, _ym(e.start), _ym(e.end)) for e in master.education]
    return FillPlan({k: v for k, v in simple.items() if v}, jobs, edu,
                    resume_pdf if resume_pdf and resume_pdf.exists() else None)


# ---------- where to type it ----------

@dataclass(frozen=True)
class FieldInfo:
    """One input on the page, as the browser adapter reports it. No values: only whether it is empty."""
    index: int                      # adapter's handle
    aid: str                        # the element's own data-automation-id ('' if none)
    kind: str                       # text | textarea | file | password | other
    empty: bool
    ancestors: tuple[str, ...] = ()  # data-automation-ids of enclosing elements, nearest first
    label: str = ""                 # visible label text (for --inspect only)


def load_fmap(path: str | Path = "workday_fields.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def _container(f: FieldInfo, prefix: str) -> str | None:
    return next((a for a in f.ancestors if a.startswith(prefix)), None)


def _date_part(f: FieldInfo, sect: dict, month_ids, year_ids) -> tuple[str, str] | None:
    """('start'|'end', 'month'|'year') if f is a month/year input inside a known date section."""
    part = "month" if f.aid in month_ids else ("year" if f.aid in year_ids else None)
    if part is None:
        return None
    for which, ids in sect.items():
        if any(a in ids for a in f.ancestors):
            return which, part
    return None


def plan_fills(fields: list[FieldInfo], plan: FillPlan, fmap: dict, done: set) -> list[tuple[int, str, tuple]]:
    """[(field index, value, key)] for EMPTY text fields CORTEX hasn't filled before. `done` holds the keys already
    filled this session, so a field he clears stays cleared. Passwords and unknown fields are never touched."""
    out = []

    def take(f: FieldInfo, key: tuple, value: str):
        if value and f.empty and key not in done and f.kind in ("text", "textarea"):
            out.append((f.index, value, key))

    for f in fields:
        if f.kind == "password":
            continue
        for k, ids in (fmap.get("simple") or {}).items():
            if f.aid in ids:
                take(f, ("simple", k), plan.simple.get(k, ""))
        if f.kind == "file" and f.aid in (fmap.get("resume_upload") or []) and plan.resume_pdf \
                and ("resume",) not in done:
            out.append((f.index, str(plan.resume_pdf), ("resume",)))
        for section, entries in (("work_experience", plan.jobs), ("education", plan.education)):
            spec = fmap.get(section) or {}
            prefix = spec.get("container_prefix")
            cont = _container(f, prefix) if prefix else None
            if cont is None:
                continue
            order = sorted({c for c in (_container(x, prefix) for x in fields) if c}, key=_block_order)
            i = order.index(cont)
            if i >= len(entries):
                continue
            e = entries[i]
            for k, ids in (spec.get("fields") or {}).items():
                if f.aid in ids:
                    take(f, (section, i, k), e.fields.get(k, ""))
            dp = _date_part(f, spec.get("dates") or {}, spec.get("month_ids") or [], spec.get("year_ids") or [])
            if dp:
                which, part = dp
                ym = e.start if which == "start" else e.end
                if ym:
                    take(f, (section, i, which, part), ym[0] if part == "month" else ym[1])
    return out


def _block_order(cid: str):
    """'workExperience-12' sorts after 'workExperience-3'."""
    tail = cid.rsplit("-", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, cid)


# ---------- the loop ----------

class Page(Protocol):
    def url(self) -> str: ...
    def fields(self) -> list[FieldInfo]: ...
    def type_into(self, index: int, value: str) -> None: ...
    def attach(self, index: int, path: str) -> None: ...
    def is_open(self) -> bool: ...


@dataclass
class FillStats:
    filled: int = 0
    skipped_non_workday: int = 0
    keys: list[tuple] = field(default_factory=list)


def run(page: Page, plan: FillPlan, fmap: dict, *, poll: float = 1.5, sleep: Callable[[float], None] = time.sleep,
        max_rounds: int | None = None, on_fill: Callable[[tuple], None] | None = None) -> FillStats:
    """Fill whatever matching empty fields appear, page after page, until he closes the browser."""
    st, done, rounds = FillStats(), set(), 0
    while page.is_open() and (max_rounds is None or rounds < max_rounds):
        rounds += 1
        if not is_workday(page.url()):
            st.skipped_non_workday += 1
        else:
            for idx, value, key in plan_fills(page.fields(), plan, fmap, done):
                if key == ("resume",):
                    page.attach(idx, value)
                else:
                    page.type_into(idx, value)
                done.add(key)
                st.filled += 1
                st.keys.append(key)
                if on_fill:
                    on_fill(key)
        sleep(poll)
    return st


def inspect(page: Page) -> list[str]:
    """Field ids + kinds + labels on the current page, for fixing workday_fields.yaml. Never values."""
    return [f"{f.kind:<8} aid={f.aid or '-':<40} in={'/'.join(f.ancestors[:3]) or '-':<60} label={f.label[:50]}"
            for f in page.fields()]
