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


def parse_rows(rows: list[list[str]], where: str = "sheet") -> tuple[int, list[str | None], list[dict]]:
    """Rows from a CSV export or the Sheets API -> (header row index, canonical header, row dicts).
    The header row is located by content, not position."""
    for i, r in enumerate(rows[:6]):
        mapped = [_canon_header(c) for c in r]
        if all(req in mapped for req in REQUIRED):
            body = rows[i + 1:]
            return i, mapped, [{h: (c if h else None) for h, c in zip(mapped, r) if h} for r in body]
    raise ValueError(f"{where}: no header row with Company/Role found in first 6 rows")


def read_tracker_csv(path: str | Path) -> list[dict]:
    """Read a Google-Sheets CSV export."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return parse_rows(list(csv.reader(f)), str(path))[2]


def _txt(v, cap: int = 256) -> str:
    clean, _ = sanitize_jd(v or "")
    return clean.strip()[:cap]


_OA_HINTS = ("applied", "not_accepting", "rejected", "no_response")


def oa_status_hint(oa) -> str | None:
    """In the real consolidated sheet, statuses were sometimes typed into the OA column instead of Status
    (12x "Applied", 28x "No longer accepting applications", 1x "No longer under consideration" on 2026-11-08).
    Only these exact recognised statuses count; free-text OA notes ("No DBT hands-on") are ignored."""
    s = P.normalize_status(_txt(oa, 64)) if _txt(oa, 64) else None
    return s if s in _OA_HINTS else None


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
    status = P.normalize_status(status_raw)
    hint = oa_status_hint(raw.get("oa"))
    if hint and P.STATUS_RANK.get(hint, 0) > P.STATUS_RANK.get(status, 0):
        status, status_raw = hint, f"{status_raw} (OA: {_txt(raw.get('oa'), 40)})"[:64]
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
        status=status, status_raw=status_raw or None,
        applied_date=P.parse_date(raw.get("applied_date")),
        oa=_txt(raw.get("oa"), 64) or None, interview=_txt(raw.get("interview"), 64) or None,
        hr=_txt(raw.get("hr"), 64) or None, offer=_txt(raw.get("offer"), 64) or None,
        followup_date=_txt(raw.get("followup_date"), 32) or None, notes=notes or None,
    )
    return vals, {"gated": gate is not None, "flagged": bool(flags)}


def import_rows(raws: list[dict], batch: str, session, cfg: dict, user_id: str, sm: ImportSummary) -> None:
    """Upsert tracker rows (from a CSV file or the live sheet). Never downgrades a more advanced status."""
    for raw in raws:
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


def import_files(paths: list[str | Path], session, cfg: dict, user_id: str,
                 audit: AuditLog | None = None) -> ImportSummary:
    """Import in the order given (pass the master first, then batches oldest -> newest)."""
    sm = ImportSummary()
    for path in paths:
        import_rows(read_tracker_csv(path), Path(path).stem, session, cfg, user_id, sm)
    session.flush()
    sm.duplicates = refresh_dups(session, user_id)
    session.commit()
    if audit:
        audit.log("importer", "synapse_import", files=[Path(p).name for p in paths], **sm.__dict__)
    return sm


def refresh_dups(session, user_id: str) -> int:
    """Recompute dup_of_id across ALL of a user's jobs (every source); the oldest row stays canonical."""
    rows = session.scalars(select(JobRow).where(JobRow.user_id == user_id).order_by(JobRow.id)).all()
    for r in rows:
        r.dup_of_id = None
    mapping = find_dups(rows)
    by_id = {r.id: r for r in rows}
    for dup_id, canon_id in mapping.items():
        by_id[dup_id].dup_of_id = canon_id
    return len(mapping)
