from datetime import date

import httpx
import pytest

from cortex.core.audit import AuditLog
from cortex.db.schema import JobRow, get_session
from cortex.sources.linkedin_status import Throttled, check_open, classify

TODAY = date(2026, 11, 13)
CFG = {"linkedin_check": {"enabled": True, "max_per_run": 60, "delay_seconds": 6, "max_age_days": 30}}
OPEN_HTML = '<section class="top-card-layout"><div class="show-more-less-html">Build pipelines</div></section>'
CLOSED_HTML = '<section class="top-card-layout"><figcaption>No longer accepting applications</figcaption></section>'


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


def add(ses, ext, status="to_apply", posted=date(2026, 11, 10), source="linkedin", **kw):
    j = JobRow(user_id="u1", source=source, external_id=ext, company=f"C{ext}", title="Data Engineer", url="u",
               status=status, posted=posted, **kw)
    ses.add(j)
    ses.commit()
    return j


def client(pages: dict, calls: list | None = None):
    def handler(req):
        if calls is not None:
            calls.append(req)
        resp = pages.get(req.url.path.rsplit("/", 1)[-1], (404, ""))
        return httpx.Response(resp[0], text=resp[1], headers=resp[2] if len(resp) > 2 else None)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_classify():
    assert classify(200, OPEN_HTML) == "open" and classify(200, CLOSED_HTML) == "closed"
    assert classify(404, "") == classify(410, "") == "closed"
    assert classify(200, "<html>something else</html>") == "unknown" and classify(500, "") == "unknown"
    for s in (429, 999, 403, 302, 301):
        with pytest.raises(Throttled):
            classify(s, "")


def test_disabled_by_default(ses):
    with pytest.raises(PermissionError, match="disabled"):
        check_open(ses, {}, "u1", today=TODAY, client=client({}))


def test_marks_closed_keeps_open_and_unclear(ses, tmp_path):
    o, c, u = add(ses, "1000001"), add(ses, "1000002"), add(ses, "1000003")
    waits, calls = [], []
    a = AuditLog(str(tmp_path / "a.log"))
    r = check_open(ses, CFG, "u1", today=TODAY, client=client({"1000001": (200, OPEN_HTML), "1000002": (200, CLOSED_HTML),
                                                              "1000003": (200, "<p>?</p>")}, calls),
                   sleep=waits.append, audit=a)
    assert (r.checked, r.open, r.closed, r.unknown) == (3, 1, 1, 1)
    assert (o.status, c.status, u.status) == ("to_apply", "not_accepting", "to_apply")
    assert c.status_raw == "auto: LinkedIn says closed 2026-11-13"
    assert len(waits) == 2 and all(w >= 6 for w in waits)                       # polite delay between requests
    req = calls[0]
    assert "cookie" not in {k.lower() for k in req.headers} and req.url.host == "www.linkedin.com"
    assert "Build pipelines" not in a.path.read_text()


def test_only_recent_linkedin_to_apply_and_capped(ses):
    ok = add(ses, "2000001")
    add(ses, "2000002", posted=date(2026, 9, 1))                                  # too old
    add(ses, "2000003", status="applied")                                         # he acted on it
    add(ses, "2000004", source="naukri")
    add(ses, "2000005", posted=None)                                              # no date: skipped
    for n in range(6, 10):
        add(ses, f"200000{n}")
    calls = []
    cfg = {"linkedin_check": {**CFG["linkedin_check"], "max_per_run": 3}}
    r = check_open(ses, cfg, "u1", today=TODAY, client=client({}, calls), sleep=lambda s: None, limit=50)
    assert r.candidates == 3 and len(calls) == 3                                  # limit can't exceed config cap
    assert all(c.url.path.rsplit("/", 1)[-1] not in ("2000002", "2000003", "2000004", "2000005") for c in calls)
    assert ok.id


def test_stops_at_first_pushback(ses):
    for n in range(1, 5):
        add(ses, f"300000{n}")
    calls = []
    r = check_open(ses, CFG, "u1", today=TODAY, sleep=lambda s: None,
                   client=client({"3000004": (200, CLOSED_HTML), "3000003": (999, "")}, calls))
    assert r.stopped and "999" in r.stopped and len(calls) == 2 and r.closed == 1
    assert ses.query(JobRow).filter_by(status="not_accepting").count() == 1


def test_redirect_to_authwall_stops_and_is_not_followed(ses):
    add(ses, "4000001")
    calls = []
    r = check_open(ses, CFG, "u1", today=TODAY, sleep=lambda s: None,
                   client=client({"4000001": (302, "", {"location": "https://www.linkedin.com/authwall"})}, calls))
    assert r.stopped and len(calls) == 1


def test_dry_run_changes_nothing(ses):
    j = add(ses, "5000001")
    r = check_open(ses, CFG, "u1", today=TODAY, sleep=lambda s: None,
                   client=client({"5000001": (200, CLOSED_HTML)}), dry_run=True)
    ses.refresh(j)
    assert r.closed == 1 and j.status == "to_apply"


def test_rotates_through_queue_and_dry_run_records_nothing(ses):
    from cortex.db.schema import JobCheckRow
    for n in range(1, 6):
        add(ses, f"600000{n}")
    cfg = {"linkedin_check": {**CFG["linkedin_check"], "max_per_run": 2}}
    pages = {f"600000{n}": (200, OPEN_HTML) for n in range(1, 6)}
    check_open(ses, cfg, "u1", today=TODAY, sleep=lambda s: None, client=client(pages), dry_run=True)
    assert ses.query(JobCheckRow).count() == 0
    seen = []
    for _ in range(3):
        calls = []
        check_open(ses, cfg, "u1", today=TODAY, sleep=lambda s: None, client=client(pages, calls))
        seen += [c.url.path.rsplit("/", 1)[-1] for c in calls]
    assert sorted(seen) == [f"600000{n}" for n in range(1, 6)]               # 2 + 2 + 1, no repeats
    from datetime import timedelta
    calls = []
    check_open(ses, cfg, "u1", today=TODAY + timedelta(days=3), sleep=lambda s: None, client=client(pages, calls))
    assert len(calls) == 2                                                    # 3 days later: due again


def test_highest_llm_score_checked_first(ses):
    import json

    from cortex.db.schema import LLMScoreCacheRow
    low, high, unscored = add(ses, "7000001"), add(ses, "7000002", posted=date(2026, 10, 20)), add(ses, "7000003")
    for j, sc in ((low, 60), (high, 92)):
        ses.add(LLMScoreCacheRow(user_id="u1", job_id=j.id, provider="p", prompt_sha256="s",
                                 result_json=json.dumps({"score": sc, "band": "P2", "reasons": [], "source": "llm"})))
    ses.commit()
    calls = []
    check_open(ses, CFG, "u1", today=TODAY, sleep=lambda s: None, client=client({}, calls))
    assert [c.url.path.rsplit("/", 1)[-1] for c in calls] == ["7000002", "7000001", "7000003"]
    assert unscored.id
