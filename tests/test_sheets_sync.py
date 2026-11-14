import json
from datetime import date

import httpx
import pytest
from sqlalchemy import select

from cortex.core import parsing as P
from cortex.core.audit import AuditLog
from cortex.core.config import load_config
from cortex.db.schema import JobRow, get_session
from cortex.sources.sheets_sync import (
    SHEET_LABEL,
    SheetsClient,
    SheetsError,
    col_letter,
    sync,
)

SID = "1BDSJOX4iLLmnFvi_LunQ3OUxhB_McL5CRNBFH-94yMU"
TOKEN = "ya29.SECRET-TOKEN"
HEADER = ["Company", "Role", "Track", "Match", "Priority", "Location", "Work Model", "Exp Req", "Posted",
          "Comp (est/disclosed)", "Resume Version", "Source", "Job URL", "Date Added", "Applied Date", "Status",
          "OA", "Interview", "HR", "Offer", "Follow-up Date", "Notes", "Stack Fit"]


def row(company, role, jid, status="To Apply", oa=""):
    r = [company, role, "Data", "80", "P1", "Hyderabad", "Hybrid", "3+ yrs", "2026-11-01", "", "DE", "LinkedIn",
         f"https://in.linkedin.com/jobs/view/{jid}", "2026-11-02", "", status, oa, "", "", "", "", "", ""]
    return r


SHEET = [HEADER, row("Acme", "Data Engineer", 4460000001), row("Beta", "Data Engineer", 4460000002, "Applied"),
         row("Gamma", "Oracle DBA", 4460000003)]


class Fake:
    def __init__(self, rows):
        self.rows, self.calls, self.writes = rows, [], []

    def handler(self, req: httpx.Request):
        self.calls.append(req)
        if req.method == "GET":
            return httpx.Response(200, json={"range": "x", "majorDimension": "ROWS", "values": self.rows})
        body = json.loads(req.content)
        self.writes.append(body)
        return httpx.Response(200, json={"totalUpdatedCells": len(body["data"])})

    def client(self, tab="Untitled"):
        return SheetsClient(SID, tab, TOKEN, client=httpx.Client(transport=httpx.MockTransport(self.handler)))


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def job(ses, jid):
    return ses.scalar(select(JobRow).where(JobRow.external_id == str(jid)))


def test_labels_round_trip_through_the_importer():
    for status, label in SHEET_LABEL.items():
        assert P.normalize_status(label) == status


def test_col_letter():
    assert [col_letter(i) for i in (0, 15, 25, 26, 27)] == ["A", "P", "Z", "AA", "AB"]


def test_read_imports_rows_and_dry_run_changes_nothing(ses, cfg, tmp_path):
    f = Fake(SHEET)
    r = sync(ses, cfg, "u1", f.client(), dry_run=True)
    assert r.read.rows_read == 3 and r.read.inserted == 3 and f.writes == []
    assert ses.query(JobRow).count() == 0
    a = AuditLog(str(tmp_path / "a.log"))
    r = sync(ses, cfg, "u1", f.client(), audit=a)
    assert ses.query(JobRow).count() == 3 and job(ses, 4460000002).status == "applied"
    assert r.to_write == [] and f.writes == []                                 # CORTEX knows nothing newer yet
    assert TOKEN not in a.path.read_text() and "Acme" not in a.path.read_text()


def test_writes_only_status_cells_cortex_knows_better(ses, cfg):
    f = Fake(SHEET)
    sync(ses, cfg, "u1", f.client())
    job(ses, 4460000001).status = "applied"                                    # he ran `cortex applied`
    job(ses, 4460000003).status = "not_accepting"                              # check-open closed it
    ses.commit()
    r = sync(ses, cfg, "u1", f.client())
    assert [(c, lab) for c, _, _, lab in r.to_write] == [("P2", "Applied"), ("P4", "Not accepting applications")]
    data = f.writes[-1]["data"]
    assert f.writes[-1]["valueInputOption"] == "RAW"
    assert [d["range"] for d in data] == ["'Untitled'!P2", "'Untitled'!P4"]   # Status column only
    assert r.written == 2


def test_never_downgrades_and_never_writes_internal_states(ses, cfg):
    f = Fake(SHEET)
    sync(ses, cfg, "u1", f.client())
    j = job(ses, 4460000002)
    j.status = "to_apply"                                                      # sheet says Applied: sheet wins
    job(ses, 4460000001).status = "approved"                                   # CORTEX-internal: never written
    ses.commit()
    r = sync(ses, cfg, "u1", f.client())
    assert r.to_write == [] and job(ses, 4460000002).status == "applied"


def test_sheet_status_upgrades_cortex(ses, cfg):
    f = Fake(SHEET)
    sync(ses, cfg, "u1", f.client())
    f.rows = [HEADER, row("Acme", "Data Engineer", 4460000001, "Interview")] + SHEET[2:]
    sync(ses, cfg, "u1", f.client())
    assert job(ses, 4460000001).status == "interview"


def test_token_in_header_tab_quoted_and_errors_clean(ses, cfg):
    f = Fake(SHEET)
    f.client("My Tab").get_rows()
    req = f.calls[0]
    assert req.headers["Authorization"] == f"Bearer {TOKEN}" and TOKEN not in str(req.url)
    assert "%27My%20Tab%27%21A1%3AZ" in str(req.url)
    assert TOKEN not in repr(f.client())
    bad = SheetsClient(SID, "Untitled", TOKEN, client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(403, json={"error": {"message": f"denied {TOKEN}"}}))))
    with pytest.raises(SheetsError, match="share the sheet") as e:
        bad.get_rows()
    assert TOKEN not in str(e.value)


@pytest.mark.parametrize("sid,tab", [("../../etc", "Untitled"), ("short", "Untitled"), (SID, "a'b"), (SID, "x!y:z/")])
def test_ids_validated(sid, tab):
    with pytest.raises(SheetsError):
        SheetsClient(sid, tab, TOKEN)


def test_disabled_by_default(cfg):
    from cortex.sources.sheets_sync import client_from_config
    with pytest.raises(SheetsError, match="off"):
        client_from_config(cfg)


def test_daily_sync_failure_is_reported(ses):
    from cortex.daily import run_daily

    def boom(s):
        raise SheetsError("HTTP 403: share the sheet with the service account's email (Editor)")
    seen = {}
    res = run_daily(ses, {}, "u1", date(2026, 11, 14), sync=boom, pull=None, fetch=None, score=None,
                    digest=lambda s, n: seen.setdefault("n", list(n)) and "sent")
    assert res.failed_steps == ["sync"] and "share the sheet" in seen["n"][0]
