"""Two-way sync with Surya's Google Sheet tracker ("SYNAPSE Job Pipeline Tracker"). Approved 2026-11-13.

READ  (sheet -> CORTEX): every row goes through the same importer as a CSV export (status mapping, OA hints,
      never-downgrade), so whatever SYNAPSE added and whatever status he typed comes in.
WRITE (CORTEX -> sheet): ONLY the Status cell, ONLY when CORTEX knows a status further along than the sheet
      (e.g. CORTEX marked a job closed, or he ran `cortex applied`). Values are the sheet's own labels, which the
      importer maps back to the same status, so a round trip is stable. Never touches any other column, never
      writes CORTEX-internal states (pending_approval / approved), never downgrades.

Auth: a Google service account (free). Its JSON key file is a secret: gitignored, path from config, never
logged; the sheet is shared with the service account's email only, so it can reach nothing else in his Drive.
The spreadsheet id and tab name are validated before going into a URL. The access token goes in the
Authorization header only. Audit log: counts only.
Needs `pip install -e ".[sheets]"` (google-auth for the service-account token).
"""
import re
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from cortex.core import parsing as P
from cortex.core.audit import AuditLog
from cortex.db.schema import JobRow
from cortex.sources.synapse_import import (
    ImportSummary,
    import_rows,
    parse_rows,
    refresh_dups,
)

API = "https://sheets.googleapis.com/v4/spreadsheets"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_ID = re.compile(r"^[A-Za-z0-9_-]{20,100}$")
_TAB = re.compile(r"^[^'\[\]*?:/\\]{1,100}$")

# CORTEX status -> the label written to the sheet (the importer maps each label straight back).
SHEET_LABEL = {"applied": "Applied", "interview": "Interview", "offer": "Offer", "rejected": "Rejected",
               "no_response": "No response", "not_accepting": "Not accepting applications", "closed": "Closed"}


class SheetsError(Exception):
    pass


def col_letter(i: int) -> str:
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def service_account_token(key_file: str) -> str:
    """Short-lived OAuth token for the service account. The key file never leaves this function."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as e:
        raise SheetsError('google-auth missing: pip install -e ".[sheets]"') from e
    try:
        creds = service_account.Credentials.from_service_account_file(key_file, scopes=SCOPES)
        creds.refresh(Request())
    except FileNotFoundError:
        raise SheetsError(f"service account key not found at {key_file}") from None
    except Exception as e:  # noqa: BLE001 — auth library errors can echo request details: type only
        raise SheetsError(f"Google auth failed: {type(e).__name__}") from None
    return creds.token


def _why_denied(r: httpx.Response) -> str:
    """Google's own reason for a 401/403, mapped to the fix. Only the status/reason codes are used, never the
    message text (it can echo request details)."""
    try:
        err = r.json().get("error", {})
    except ValueError:
        err = {}
    codes = {str(err.get("status", ""))} | {str(d.get("reason", "")) for d in err.get("details", []) if isinstance(d, dict)}
    if codes & {"SERVICE_DISABLED", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"} or "accessNotConfigured" in str(err):
        return ("Google Sheets API is not enabled in the service account's project: Cloud Console -> APIs & Services "
                "-> Enable APIs and services -> Google Sheets API -> Enable (then wait a minute)")
    if r.status_code == 401:
        return "the service account key was rejected: create a new JSON key and replace google-service-account.json"
    return "share the sheet with the service account's email (client_email in the key file) as Editor"


class SheetsClient:
    def __init__(self, spreadsheet_id: str, tab: str, token: str, client: httpx.Client | None = None):
        if not _ID.match(spreadsheet_id or ""):
            raise SheetsError("invalid spreadsheet id (copy it from the sheet URL: /d/<id>/edit)")
        if not _TAB.match(tab or ""):
            raise SheetsError("invalid tab name")
        self.sid, self.tab, self._token = spreadsheet_id, tab, token
        self._c = client or httpx.Client(timeout=30)

    def __repr__(self) -> str:
        return f"SheetsClient(sheet={self.sid[:6]}..., tab={self.tab!r}, token=***)"

    def _a1(self, cells: str) -> str:
        return f"'{self.tab}'!{cells}"

    def _req(self, method: str, url: str, **kw):
        try:
            r = self._c.request(method, url, headers={"Authorization": f"Bearer {self._token}"},
                                follow_redirects=False, **kw)
        except httpx.HTTPError as e:
            raise SheetsError(type(e).__name__) from None
        if r.status_code in (401, 403):
            raise SheetsError(f"HTTP {r.status_code}: {_why_denied(r)}")
        if r.status_code == 404:
            raise SheetsError("HTTP 404: spreadsheet or tab not found")
        if r.status_code != 200:
            raise SheetsError(f"HTTP {r.status_code}")
        return r.json()

    def get_rows(self) -> list[list[str]]:
        data = self._req("GET", f"{API}/{self.sid}/values/{quote(self._a1('A1:Z'), safe='')}",
                         params={"majorDimension": "ROWS"})
        rows = data.get("values") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise SheetsError("unexpected values response")
        return [[str(c) for c in r] for r in rows]

    def write_cells(self, updates: list[tuple[str, str]]) -> int:
        """[(A1 cell like 'P12', value)] in one batchUpdate. RAW: a value is never parsed as a formula."""
        if not updates:
            return 0
        body = {"valueInputOption": "RAW",
                "data": [{"range": self._a1(cell), "values": [[value]]} for cell, value in updates]}
        res = self._req("POST", f"{API}/{self.sid}/values:batchUpdate", json=body)
        return int(res.get("totalUpdatedCells", 0)) if isinstance(res, dict) else 0


@dataclass
class SyncResult:
    read: ImportSummary = field(default_factory=ImportSummary)
    to_write: list[tuple[str, str, str, str]] = field(default_factory=list)   # (cell, company, title, new label)
    written: int = 0


def plan_writes(header: list[str | None], header_row: int, raws: list[dict], session, user_id: str) \
        -> list[tuple[str, str, str, str]]:
    """Status cells where CORTEX knows more than the sheet."""
    if "status" not in header:
        raise SheetsError("sheet has no Status column")
    col = col_letter(header.index("status"))
    out = []
    for i, raw in enumerate(raws):
        url = (raw.get("url") or "").strip()
        if not url:
            continue
        job = session.query(JobRow).filter_by(user_id=user_id, source=P.normalize_source(raw.get("source")),
                                              external_id=P.extract_external_id(url)).one_or_none()
        if job is None or job.status not in SHEET_LABEL:
            continue
        sheet_status = P.normalize_status(raw.get("status") or "")
        if P.STATUS_RANK.get(job.status, 0) > P.STATUS_RANK.get(sheet_status, 0):
            row_no = header_row + 2 + i                         # 1-based sheet row of this data row
            out.append((f"{col}{row_no}", raw.get("company") or "", raw.get("role") or "", SHEET_LABEL[job.status]))
    return out


def sync(session, cfg: dict, user_id: str, client: SheetsClient, *, dry_run: bool = False,
         audit: AuditLog | None = None) -> SyncResult:
    sc = cfg.get("sheets") or {}
    rows = client.get_rows()
    header_row, header, raws = parse_rows(rows, "Google Sheet")
    res = SyncResult()
    import_rows(raws, f"sheet:{client.tab}"[:64], session, cfg, user_id, res.read)
    session.flush()
    res.read.duplicates = refresh_dups(session, user_id)
    res.to_write = plan_writes(header, header_row, raws, session, user_id) if sc.get("write_back", True) else []
    if dry_run:
        session.rollback()
    else:
        session.commit()
        res.written = client.write_cells([(cell, label) for cell, _, _, label in res.to_write])
    if audit:
        audit.log("sheets_sync", "sync", user_id=user_id, rows=res.read.rows_read, inserted=res.read.inserted,
                  updated=res.read.updated, status_writes=len(res.to_write), written=res.written, dry_run=dry_run)
    return res


def client_from_config(cfg: dict) -> SheetsClient:
    sc = cfg.get("sheets") or {}
    if not sc.get("enabled"):
        raise SheetsError("sheets sync is off: set sheets.enabled + spreadsheet_id in config.local.yaml")
    return SheetsClient(sc.get("spreadsheet_id", ""), sc.get("tab", "Untitled"),
                        service_account_token(sc.get("key_file", "google-service-account.json")))
