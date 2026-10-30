"""Import SYNAPSE tracker CSV exports (frozen master + dated batch sheets) into CORTEX.

Tracker text is untrusted (it originates from scraped postings): every field is
sanitized + length-capped, injection patterns are flagged for manual review, and
re-imports are idempotent (upsert on user_id+source+external_id). A re-import never
downgrades a more advanced status (e.g. Applied is not reset to To Apply).
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from cortex.core import parsing as P
from cortex.core.audit import AuditLog
from cortex.core.dedupe import find_dups
from cortex.core.gates import run_gates
from cortex.core.models import Job
from cortex.core.sanitize import sanitize_jd
from cortex.core.scoring import score_job
from cortex.db.schema import JobRow

REQUIRED = ("company", "role")
_ALIASES = {"company": "company", "role": "role", "track": "track", "match": "match",
            "priority": "priority", "location": "location", "workmodel": "work_model",
            "expreq": "exp_raw", "posted": "posted", "resumeversion": "resume_version",
            "source": "source", "joburl": "url", "dateadded": "date_added",
            "applieddate": "applied_date", "status": "status", "oa": "oa",
            "interview": "interview", "hr": "hr", "offer": "offer",
            "followupdate": "followup_date", "notes": "notes", "stackfit": "stack_fit"}


@dataclass
class ImportSummary:
    rows_read: int = 0
    inserted: int = 0
    updated: int = 0
    skipped_blank: int = 0
    gated: int = 0
    flagged: int = 0
    duplicates: int = 0


def _canon_header(h: str) -> str | None:
    k = "".join(ch for ch in (h or "").lower() if ch.isalnum())
    if k in ("comp", "compestdisclosed", "compensation"):
        return "comp"
    return _ALIASES.get(k)


def read_tracker_csv(path: str | Path) -> list[dict]:
    """Read a Google-Sheets CSV export. Header row is located by content, not position."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for i, r in enumerate(rows[:6]):
        mapped = [_canon_header(c) for c in r]
        if all(req in mapped for req in REQUIRED):
            header = mapped
            body = rows[i + 1:]
            break
    else:
        raise ValueError(f"{path}: no header row with Company/Role found in first 6 rows")
    return [{h: (c if h else None) for h, c in zip(header, r) if h} for r in body]


def _txt(v, cap: int = 256) -> str:
    clean, _ = sanitize_jd(v or "")
    return clean.strip()[:cap]


def row_to_values(raw: dict, user_id: str, batch: str, cfg: dict) -> tuple[dict, dict] | None:
    company, title = _txt(raw.get("company")), _txt(raw.get("role"))
    if not company or not title:
        return None
    url = _txt(raw.get("url"), 2000)
    source = P.normalize_source(raw.get("source"))
    min_yoe, max_yoe = P.parse_exp(raw.get("exp_raw"))
    location = _txt(raw.get("location"))
    track = P.normalize_track(raw.get("track"))
    stack = P.stack_category(raw.get("stack_fit"))
    notes = _txt(raw.get("notes"), 2000)

    _, flags = sanitize_jd(" ".join([company, title, raw.get("notes") or ""]))
    job = Job(source=source, external_id=P.extract_external_id(url), title=title,
              company=company, location=location, url=url, min_yoe=min_yoe, max_yoe=max_yoe)
    gate = run_gates(job, cfg)
    res = score_job(track=track, stack=stack, location=location, min_yoe=min_yoe,
                    applicants=None, cfg=cfg)
    match = str(raw.get("match") or "").strip()
    status_raw = _txt(raw.get("status"), 64)
    vals = dict(
        user_id=user_id, source=source, external_id=job.external_id, batch=batch[:64],
        company=company, title=title, url=url, location=location,
        work_model=_txt(raw.get("work_model"), 64) or None, comp=_txt(raw.get("comp"), 128) or None,
        posted=P.parse_date(raw.get("posted")), date_added=P.parse_date(raw.get("date_added")),
        exp_raw=_txt(raw.get("exp_raw"), 128) or None, min_yoe=min_yoe, max_yoe=max_yoe,
        stack_fit=_txt(raw.get("stack_fit"), 128) or None,
        resume_version=_txt(raw.get("resume_version"), 64) or None,
        track_raw=_txt(raw.get("track"), 32) or None,
        synapse_score=int(match) if match.isdigit() else None,
        synapse_priority=_txt(raw.get("priority"), 4) or None,
        track=track, score=res.score, band=res.band, score_reasons=json.dumps(res.reasons),
        gate_reason=gate, sanitizer_flags=json.dumps(flags) if flags else None,
        status=P.normalize_status(status_raw), status_raw=status_raw or None,
        applied_date=P.parse_date(raw.get("applied_date")),
        oa=_txt(raw.get("oa"), 64) or None, interview=_txt(raw.get("interview"), 64) or None,
        hr=_txt(raw.get("hr"), 64) or None, offer=_txt(raw.get("offer"), 64) or None,
        followup_date=_txt(raw.get("followup_date"), 32) or None, notes=notes or None,
    )
    return vals, {"gated": gate is not None, "flagged": bool(flags)}


def import_files(paths: list[str | Path], session, cfg: dict, user_id: str,
                 audit: AuditLog | None = None) -> ImportSummary:
    """Import in the order given (pass the master first, then batches oldest -> newest)."""
    sm = ImportSummary()
    for path in paths:
        batch = Path(path).stem
        for raw in read_tracker_csv(path):
            sm.rows_read += 1
            out = row_to_values(raw, user_id, batch, cfg)
            if out is None:
                sm.skipped_blank += 1
                continue
            vals, info = out
            sm.gated += info["gated"]
            sm.flagged += info["flagged"]
            existing = session.scalar(select(JobRow).where(
                JobRow.user_id == user_id, JobRow.source == vals["source"],
                JobRow.external_id == vals["external_id"]))
            if existing is None:
                session.add(JobRow(**vals))
                sm.inserted += 1
            else:
                if P.STATUS_RANK.get(vals["status"], 0) < P.STATUS_RANK.get(existing.status, 0):
                    for k in ("status", "status_raw", "applied_date"):
                        vals.pop(k)          # never downgrade a more advanced status
                for k, v in vals.items():
                    setattr(existing, k, v)
                sm.updated += 1
    session.flush()
    rows = session.scalars(select(JobRow).where(JobRow.user_id == user_id).order_by(JobRow.id)).all()
    for r in rows:
        r.dup_of_id = None
    mapping = find_dups(rows)
    by_id = {r.id: r for r in rows}
    for dup_id, canon_id in mapping.items():
        by_id[dup_id].dup_of_id = canon_id
    sm.duplicates = len(mapping)
    session.commit()
    if audit:
        audit.log("importer", "synapse_import", files=[Path(p).name for p in paths], **sm.__dict__)
    return sm
