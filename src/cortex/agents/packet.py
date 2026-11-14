"""Packet agent: turn an APPROVED tailored resume into files Surya can submit by hand.

    packets/<user>/<job_id>-<company>-<title>/
        resume.pdf        tailored resume (contact header from contact.yaml, education/certs from the master)
        cover_note.pdf    cover note (if the tailor wrote one), same header; cover_note.txt for paste-in forms
        workday.md        every Workday field in page order, ready to paste (see agents/workday.py)
        job.md            company, role, apply link, what to do next

It never submits anything. Safety:
- Runs only after the human approval gate (graph node after `record`, approved branch only).
- Re-runs Truth Guard against the current master before rendering; a changed master refuses the build.
- All text is XML-escaped before it reaches ReportLab (its Paragraph parses markup like <a href>), so
  LLM output influenced by untrusted job text can't inject links or formatting into the PDF.
- Folder names come from untrusted company/title text: slugged to [a-z0-9-] and checked to stay under
  the packets root. Apply links are kept only if they are http(s).
- Contact details live in a gitignored contact.yaml, never in the master resume Truth Guard reads.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cortex.agents.truth_guard import TruthGuard
from cortex.agents.workday import StandardAnswers, render_sheet
from cortex.core.resume import MasterResume, TailoredResume

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class PacketError(RuntimeError):
    pass


class ContactInfo(BaseModel):
    """Header of the resume PDF. Kept out of master_resume.yaml on purpose."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=80)
    location: str | None = Field(default=None, max_length=80)
    email: str | None = Field(default=None, max_length=120)
    phone: str | None = Field(default=None, max_length=40)
    linkedin: str | None = Field(default=None, max_length=200)
    github: str | None = Field(default=None, max_length=200)


def load_contact(path: str | Path = "contact.yaml") -> ContactInfo:
    p = Path(path)
    if not p.exists():
        raise PacketError(f"{p} not found: copy contact.example.yaml to {p} and fill it in")
    return ContactInfo.model_validate(yaml.safe_load(p.read_text()))


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:max_len].rstrip("-") or "x"


def safe_url(url: str | None) -> str | None:
    if not url:
        return None
    u = urlparse(url.strip())
    return url.strip() if u.scheme in ("http", "https") and u.netloc else None


def fmt_month(ym: str) -> str:
    if ym == "present":
        return "Present"
    y, m = ym.split("-")
    return f"{MONTHS[int(m) - 1]} {y}"


def _e(text: str) -> str:
    """Escape for ReportLab Paragraph markup (it parses <b>, <a href>, &entities;)."""
    return escape(text or "")


@dataclass
class PacketResult:
    folder: Path
    resume_pdf: Path
    job_md: Path
    cover_pdf: Path | None = None
    cover_txt: Path | None = None
    workday_md: Path | None = None


class PacketBuilder:
    def __init__(self, master: MasterResume, contact: ContactInfo, root: str | Path = "packets",
                 answers: StandardAnswers | None = None, originals_dir: str | Path = "resumes",
                 originals: dict[str, str] | None = None):
        """originals: track -> his own file name in originals_dir (config tailor.originals); default <track>.pdf.
        The packet keeps that exact file name: recruiters see the attachment's name."""
        self.originals_dir = Path(originals_dir)
        self.originals = dict(originals or {})
        self.master = master
        self.contact = contact
        self.answers = answers
        self.root = Path(root)
        self.guard = TruthGuard(master)

    def folder_for(self, user_id: str, job_id: int, company: str, title: str) -> Path:
        root = self.root.resolve()
        folder = (root / slug(user_id, 32) / f"{int(job_id)}-{slug(company, 30)}-{slug(title, 30)}").resolve()
        if root not in folder.parents:
            raise PacketError("packet path escaped the packets root")
        return folder

    def build(self, resume: TailoredResume, *, user_id: str, job_id: int, company: str, title: str,
              location: str = "", url: str | None = None) -> PacketResult:
        violations = self.guard.check(resume)
        if violations:
            raise PacketError("tailored resume no longer passes Truth Guard against the current master: "
                              + "; ".join(v.feedback() for v in violations[:5]))
        folder = self.folder_for(user_id, job_id, company, title)
        folder.mkdir(parents=True, exist_ok=True)
        pdf = folder / "resume.pdf"
        if (resume.based_on or "").startswith("original:"):
            src = self._original_path(resume.based_on.split(":", 1)[1])
            pdf = folder / src.name
            self._copy_original(resume.based_on.split(":", 1)[1], pdf)
        else:
            self._render_pdf(resume, pdf, company=company, title=title)
        cover_pdf = cover_txt = None
        if resume.cover_note:
            cover_txt = folder / "cover_note.txt"
            cover_txt.write_text("\n\n".join(resume.cover_note) + "\n")
            cover_pdf = folder / "cover_note.pdf"
            self._render_cover(resume.cover_note, cover_pdf, company=company, title=title)
        wd = folder / "workday.md"
        wd.write_text(render_sheet(self.master, resume, self.contact, self.answers, company=company, title=title))
        md = folder / "job.md"
        md.write_text(self._job_md(company, title, location, url, has_cover=bool(resume.cover_note), resume_name=pdf.name))
        return PacketResult(folder, pdf, md, cover_pdf, cover_txt, wd)

    def _original_path(self, track: str) -> Path:
        """resumes/<file> for this track. Track and file name are checked so nothing can point outside resumes/."""
        if not re.fullmatch(r"[a-z0-9_]{1,40}", track):
            raise PacketError(f"invalid track name {track!r}")
        name = self.originals.get(track, f"{track}.pdf")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._()-]{0,120}\.pdf", name, re.IGNORECASE):
            raise PacketError(f"invalid resume file name for {track}: {name!r} (plain file name ending in .pdf)")
        return self.originals_dir / name

    def _copy_original(self, track: str, dest: Path) -> None:
        """Surya's own resume file, byte for byte."""
        src = self._original_path(track)
        if not src.is_file():
            raise PacketError(f"{src} not found: put your {track.replace('_', ' ')} resume there, or map the track to "
                              f"your file name under tailor.originals in config.local.yaml")
        data = src.read_bytes()
        if not data.startswith(b"%PDF"):
            raise PacketError(f"{src} is not a PDF")
        dest.write_bytes(data)

    # ---- rendering ----
    def _job_md(self, company: str, title: str, location: str, url: str | None, has_cover: bool = False,
                resume_name: str = "resume.pdf") -> str:
        link = safe_url(url)
        clean = " ".join(f"{company} | {title}".split())
        return (f"# {clean}\n\n"
                + (f"- Location: {' '.join(location.split())}\n" if location else "")
                + (f"- Apply: {link}\n" if link else "- Apply: (no valid http(s) link on record; find it manually)\n")
                + f"- Resume: {resume_name} (approved in the CORTEX dashboard)\n"
                + ("- Cover note: cover_note.pdf (cover_note.txt for paste-in forms)\n" if has_cover else "")
                + "- Workday: workday.md has every field in page order, ready to paste\n"
                + "\n"
                  f"CORTEX never submits applications. Open the link, attach {resume_name}, and apply yourself.\n")

    def _render_pdf(self, r: TailoredResume, path: Path, *, company: str, title: str) -> None:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        base = ParagraphStyle("base", fontName="Helvetica", fontSize=9.5, leading=12.5)
        name = ParagraphStyle("name", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21,
                              alignment=TA_CENTER)
        contact = ParagraphStyle("contact", parent=base, fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
        head = ParagraphStyle("head", parent=base, fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=7)
        role = ParagraphStyle("role", parent=base, fontName="Helvetica-Bold", spaceBefore=4)
        bullet = ParagraphStyle("bullet", parent=base, leftIndent=10, bulletIndent=0)
        dates = ParagraphStyle("dates", parent=base, alignment=TA_RIGHT, textColor=colors.HexColor("#555555"))
        width = A4[0] - 30 * mm

        def heading_row(left_markup: str, start: str, end: str, style):
            """Title/employer on the left, dates right-aligned, so long employer names never push dates
            onto their own line."""
            t = Table([[Paragraph(left_markup, style), Paragraph(_e(f"{fmt_month(start)} – {fmt_month(end)}"), dates)]],
                      colWidths=[width - 34 * mm, 34 * mm])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 3),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
            return t

        def section(label):
            return [Paragraph(_e(label.upper()), head), HRFlowable(width="100%", thickness=0.6, color=colors.grey,
                                                                    spaceBefore=1, spaceAfter=3)]

        c = self.contact
        parts = [x for x in (c.location, c.email, c.phone, c.linkedin, c.github) if x]
        story = [Paragraph(_e(c.name), name), Paragraph(_e("  |  ".join(parts)), contact), Spacer(1, 4)]
        story += section("Summary") + [Paragraph(_e(r.summary), base)]
        if r.skill_groups:   # grouped, as on Surya's own resumes
            story += section("Technical Skills") + [
                Paragraph(f"<b>{_e(g.label)}:</b> {_e(', '.join(g.skills))}", base) for g in r.skill_groups]
        elif r.skills:
            story += section("Skills") + [Paragraph(_e(", ".join(r.skills)), base)]
        if r.roles:
            story += section("Experience")
            for role_ in r.roles:
                who = role_.employer + (f" (Client: {role_.client})" if role_.client else "")
                story.append(heading_row(f"{_e(role_.title)}, {_e(who)}", role_.start, role_.end, role))
                story += [Paragraph(_e(b), bullet, bulletText="•") for b in role_.bullets]
        if r.projects:
            story += section("Projects")
            for p in r.projects:
                story.append(Paragraph(_e(p.name), role))
                story += [Paragraph(_e(b), bullet, bulletText="•") for b in p.bullets]
        certs = [x for x in self.master.certifications
                 if {x.name.lower(), (x.code or "").lower()} & {cc.lower() for cc in r.certifications}]
        if certs:
            story += section("Certifications")
            story += [Paragraph(_e(f"{x.name} ({x.code})" if x.code else x.name), bullet, bulletText="•") for x in certs]
        if self.master.education:
            story += section("Education")
            for ed in self.master.education:
                story.append(heading_row(f"{_e(ed.degree)}, {_e(ed.institution)}", ed.start, ed.end, base))

        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                                topMargin=12 * mm, bottomMargin=12 * mm, title=f"{c.name} - {company} {title}"[:120],
                                author=c.name, creator="CORTEX")
        doc.build(story)

    def _render_cover(self, paragraphs: list[str], path: Path, *, company: str, title: str) -> None:
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        base = ParagraphStyle("base", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=8)
        name = ParagraphStyle("name", parent=base, fontName="Helvetica-Bold", fontSize=15, alignment=TA_CENTER)
        contact = ParagraphStyle("contact", parent=base, fontSize=9, alignment=TA_CENTER)
        c = self.contact
        parts = [x for x in (c.location, c.email, c.phone, c.linkedin) if x]
        story = [Paragraph(_e(c.name), name), Paragraph(_e("  |  ".join(parts)), contact), Spacer(1, 10)]
        story += [Paragraph(_e(p), base) for p in paragraphs]
        story += [Spacer(1, 6), Paragraph(_e(c.name), base)]
        SimpleDocTemplate(str(path), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=18 * mm,
                          bottomMargin=18 * mm, title=f"{c.name} - cover note - {company} {title}"[:120],
                          author=c.name, creator="CORTEX").build(story)


def packet_resume(folder: str | Path) -> Path | None:
    """The resume PDF in a packet folder: resume.pdf (rendered) or his own original (any other PDF that isn't the
    cover note)."""
    f = Path(folder)
    if (f / "resume.pdf").is_file():
        return f / "resume.pdf"
    pdfs = sorted(p for p in f.glob("*.pdf") if p.name != "cover_note.pdf")
    return pdfs[0] if len(pdfs) == 1 else None
