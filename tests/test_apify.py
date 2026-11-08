"""Fixtures are trimmed copies of REAL rows from SYNAPSE's Apify datasets (2026-11-08), not guessed shapes."""
from datetime import UTC, date, datetime

import httpx
import pytest
from sqlalchemy import select

from cortex.core.audit import AuditLog
from cortex.core.config import load_config
from cortex.db.schema import JobRow, get_session
from cortex.sources.apify import (
    ApifyClient,
    ApifyError,
    parse_applicants,
    parse_linkedin,
    parse_naukri,
    pull_all,
)

NOW = datetime(2026, 11, 8, 12, tzinfo=UTC)
TODAY = date(2026, 11, 8)
TOKEN = "apify_api_SECRETTOKEN123"
_REAL_CLIENT = httpx.Client

LI = {"id": "4449747860", "url": "https://in.linkedin.com/jobs/view/lead-ai-engineer-at-qualys-4449747860",
      "title": "Lead AI Engineer", "location": "Pune Division, Maharashtra, India",
      "postedDate": "2026-10-31T00:00:00.000Z", "companyName": "Qualys", "companyUrl": "https://www.linkedin.com/company/qualys",
      "recruiterName": "", "experienceLevel": "Not Applicable", "contractType": "Full-time",
      "workType": "Computer and Network Security", "salary": "", "applyType": "EXTERNAL", "postedTimeAgo": "6 days ago",
      "applicationsCount": "Over 200 applicants",
      "description": "Design, build, and scale enterprise-grade AI/GenAI solutions. 7–12+ years of experience in Software Engineering.",
      "descriptionHtml": "<strong>Job</strong>", "applyUrl": ""}
LI_GOOD = {**LI, "id": "4450000001", "title": "Data Engineer", "companyName": "Acme",
           "url": "https://in.linkedin.com/jobs/view/data-engineer-at-acme-4450000001", "location": "Hyderabad, Telangana, India",
           "applicationsCount": "27 applicants", "description": "3+ years of experience with Azure Data Factory and PySpark."}

NK = {"jobId": "031126009902", "title": "Azure Data Engineer",
      "staticUrl": "https://www.naukri.com/job-listings-azure-data-engineer-tata-consultancy-services-hyderabad-pune-bengaluru-12-to-18-years-031126009902",
      "description": "Experience building scalable ETL / ELT pipelines. Experience : 12+ years.",
      "shortDescription": "12+ years experience with expertise in Azure Synapse, ADF",
      "companyDetail": {"name": "Tata Consultancy Services", "details": "<p><strong>TCS</strong></p>"},
      "salaryDetail": {"minimumSalary": 0, "maximumSalary": 0, "currency": "INR", "label": "Not Disclosed", "hideSalary": True},
      "minimumExperience": 12, "maximumExperience": 18, "experienceText": "12-18 Yrs",
      "wfhType": "0", "wfhLabel": "Work from office",
      "locations": [{"localities": [], "label": "Hyderabad"}, {"label": "Pune"}, {"label": "Bengaluru"}],
      "applyCount": 4, "createdDate": "2026-11-03 10:11:18",
      "url": "https://www.naukri.com/job-listings-azure-data-engineer-tata-consultancy-services-hyderabad-pune-bengaluru-12-to-18-years-031126009902",
      "basicInfo": {"jobId": "031126009902", "title": "Azure Data Engineer", "companyName": "Tata Consultancy Services"}}
NK_GOOD = {**NK, "jobId": "031126000001", "title": "Data Engineer", "minimumExperience": 3, "maximumExperience": 6,
           "companyDetail": {"name": "Beta Corp"}, "url": "https://www.naukri.com/job-listings-data-engineer-031126000001",
           "salaryDetail": {"label": "12-18 Lacs PA", "hideSalary": False}}

RUNS = {"data": {"items": [
    {"id": "r1", "status": "SUCCEEDED", "startedAt": "2026-11-06T19:53:33.521Z", "defaultDatasetId": "0Bn6SlCEcSSE1dQJ1"},
    {"id": "r0", "status": "SUCCEEDED", "startedAt": "2026-10-01T10:00:00.000Z", "defaultDatasetId": "OLDoldOLD123"},
    {"id": "rx", "startedAt": "garbage", "defaultDatasetId": "x"}, "junk"]}}
NK_RUNS = {"data": {"items": [
    {"id": "n1", "status": "SUCCEEDED", "startedAt": "2026-11-03T06:25:01.976Z", "defaultDatasetId": "oTuJhqeBdxpQhZxUj"}]}}


def api(routes: dict, calls: list | None = None):
    def handler(req: httpx.Request):
        if calls is not None:
            calls.append(req)
        for key, resp in routes.items():
            if req.url.path.endswith(key):
                return resp() if callable(resp) else httpx.Response(200, json=resp)
        return httpx.Response(404)
    return ApifyClient(TOKEN, client=_REAL_CLIENT(transport=httpx.MockTransport(handler)))


def ok_api(calls=None):
    return api({"/acts/valig~linkedin-jobs-scraper/runs": RUNS, "/datasets/0Bn6SlCEcSSE1dQJ1/items": [LI, LI_GOOD, LI_GOOD, {}],
                "/acts/memo23~naukri-scraper/runs": NK_RUNS, "/datasets/oTuJhqeBdxpQhZxUj/items": [NK, NK_GOOD, {"title": "gulf"}]},
               calls)


@pytest.fixture
def ses(tmp_path):
    return get_session(f"sqlite:///{tmp_path}/t.db")


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def run(ses, cfg, client, **kw):
    return pull_all(ses, cfg, "u1", client=client, today=TODAY, now=NOW, **kw)


# ---------- parsers ----------

def test_parse_applicants():
    assert parse_applicants("Over 200 applicants") == 201 and parse_applicants("57 applicants") == 57
    assert parse_applicants("Be among the first 25 applicants") == 25 and parse_applicants("1,234 applicants") == 1234
    assert parse_applicants("") is None and parse_applicants(None) is None


def test_parse_real_linkedin_row():
    p, company = parse_linkedin(LI)
    assert (p.external_id, p.title, company, p.posted, p.applicants) == ("4449747860", "Lead AI Engineer", "Qualys",
                                                                        date(2026, 10, 31), 201)
    assert p.comp is None and parse_linkedin({"id": "abc", "title": "x"}) is None and parse_linkedin("x") is None


NK_NULL_COMPANY = {"jobId": "031126008878", "title": "Snowflake Developer - SA", "companyDetail": None,
                   "staticCompanyName": "https://www.naukri.com/pwc-jobs-careers-676499", "consultant": False,
                   "locations": [{"label": "Hyderabad"}, {"label": "Chennai"}], "minimumExperience": 4,
                   "maximumExperience": 8, "applyCount": 200, "createdDate": "2026-11-03 09:19:42",
                   "url": "https://www.naukri.com/job-listings-snowflake-developer-sa-pwc-hyderabad-chennai-bengaluru-4-to-8-years-031126008878",
                   "wfhLabel": "Work from office"}


def test_naukri_company_from_slug_when_company_detail_null():
    from cortex.sources.apify import company_from_slug
    p, company = parse_naukri(NK_NULL_COMPANY)                                      # real row shape: companyDetail null
    assert (company, p.exp_raw, p.applicants) == ("Pwc", "4-8 yrs", 200)
    assert company_from_slug("https://www.naukri.com/lpl-financials-jobs-careers-5507842") == "Lpl Financials"
    assert company_from_slug("https://evil.example/pwc-jobs-careers-1") == "" and company_from_slug(None) == ""


def test_parse_real_naukri_row():
    p, company = parse_naukri(NK)
    assert (p.external_id, company, p.exp_raw, p.applicants, p.posted) == ("031126009902", "Tata Consultancy Services",
                                                                            "12-18 yrs", 4, date(2026, 11, 3))
    assert p.location == "Hyderabad; Pune; Bengaluru" and p.work_model == "Work from office" and p.comp is None
    assert parse_naukri(NK_GOOD)[0].comp == "12-18 Lacs PA"
    assert parse_naukri({"title": "Gulf row, no jobId"}) is None


# ---------- client ----------

def test_token_only_in_header_never_in_url_repr_or_errors():
    calls = []
    c = ok_api(calls)
    c.recent_datasets("valig~linkedin-jobs-scraper", NOW.replace(day=1))
    req = calls[0]
    assert req.headers["Authorization"] == f"Bearer {TOKEN}" and TOKEN not in str(req.url)
    assert TOKEN not in repr(c)
    bad = api({"/runs": lambda: httpx.Response(401, json={"error": {"message": f"bad token {TOKEN}"}})})
    with pytest.raises(ApifyError) as e:
        bad.recent_datasets("valig~linkedin-jobs-scraper", NOW)
    assert TOKEN not in str(e.value) and "401" in str(e.value)


def test_ids_validated_before_urls():
    c = ok_api()
    for bad in ("../x", "valig/linkedin", "a~b/../c", "valig~x?y=1"):
        with pytest.raises(ApifyError, match="invalid actor"):
            c.recent_datasets(bad, NOW)
    with pytest.raises(ApifyError, match="invalid dataset"):
        c.items("../../users/me")


def test_only_recent_succeeded_runs():
    assert ok_api().recent_datasets("valig~linkedin-jobs-scraper", datetime(2026, 10, 31, tzinfo=UTC)) == ["0Bn6SlCEcSSE1dQJ1"]


# ---------- pull ----------

def test_pull_inserts_filters_and_scores_with_applicants(ses, cfg, tmp_path):
    a = AuditLog(str(tmp_path / "a.log"))
    res = {r.source: r for r in run(ses, cfg, ok_api(), audit=a)}
    li, nk = res["linkedin"], res["naukri"]
    assert (li.datasets, li.items, li.kept, li.inserted) == (1, 4, 1, 1)      # Lead filtered, dup row seen once, {} skipped
    assert (nk.items, nk.kept, nk.inserted) == (3, 2, 2)
    rows = {r.external_id: r for r in ses.scalars(select(JobRow))}
    acme = rows["4450000001"]
    assert (acme.source, acme.company, acme.min_yoe, acme.status, acme.batch) == ("linkedin", "Acme", 3, "to_apply",
                                                                                 "apify-2026-11-08")
    assert "applicants<30" in acme.score_reasons
    tcs = rows["031126009902"]
    assert (tcs.min_yoe, tcs.max_yoe) == (12, 18) and tcs.gate_reason                 # YOE gate from structured fields
    assert rows["031126000001"].comp == "12-18 Lacs PA"
    assert TOKEN not in a.path.read_text() and "PySpark" not in a.path.read_text()


def test_sheet_row_only_gets_missing_description(ses, cfg):
    sheet = JobRow(user_id="u1", source="linkedin", external_id="4450000001", batch="Batch011", company="Acme",
                   title="Data Engineer", location="Hyderabad", url="u", status="applied", track="data", score=72,
                   band="P2", exp_raw="~3-6 yrs", min_yoe=3, notes="mine")
    ses.add(sheet)
    ses.commit()
    res = {r.source: r for r in run(ses, cfg, ok_api())}
    assert (res["linkedin"].inserted, res["linkedin"].enriched) == (0, 1)
    ses.refresh(sheet)
    assert (sheet.status, sheet.score, sheet.band, sheet.exp_raw, sheet.notes) == ("applied", 72, "P2", "~3-6 yrs", "mine")
    assert "Azure Data Factory" in sheet.description
    assert ses.query(JobRow).count() == 3


def test_repull_updates_own_rows_idempotently(ses, cfg):
    run(ses, cfg, ok_api())
    j = ses.scalar(select(JobRow).where(JobRow.external_id == "4450000001"))
    j.status = "applied"
    ses.commit()
    res = {r.source: r for r in run(ses, cfg, ok_api())}
    assert sum(r.inserted for r in res.values()) == 0 and res["linkedin"].updated == 1
    ses.refresh(j)
    assert j.status == "applied" and ses.query(JobRow).count() == 3


def test_errors_isolated_per_actor(ses, cfg):
    client = api({"/acts/valig~linkedin-jobs-scraper/runs": lambda: httpx.Response(500),
                  "/acts/memo23~naukri-scraper/runs": NK_RUNS, "/datasets/oTuJhqeBdxpQhZxUj/items": [NK_GOOD]})
    cfg["sources"]["apify"]["actors"].append({"actor": "x~y", "source": "indeed"})
    res = {r.actor: r for r in run(ses, cfg, client)}
    assert res["valig~linkedin-jobs-scraper"].error == "HTTP 500" and res["memo23~naukri-scraper"].inserted == 1
    assert "unknown source" in res["x~y"].error


def test_dry_run_and_missing_token(ses, cfg, monkeypatch):
    res = run(ses, cfg, ok_api(), dry_run=True)
    assert sum(r.inserted for r in res) == 3 and ses.query(JobRow).count() == 0
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    with pytest.raises(ApifyError, match="APIFY_TOKEN missing"):
        pull_all(ses, cfg, "u1", today=TODAY, now=NOW)


def test_cross_source_dedupe_with_ats_row(ses, cfg):
    ats = JobRow(user_id="u1", source="greenhouse", external_id="acme:9", batch="ats-2026-11-07", company="Acme",
                 title="Data Engineer", location="Hyderabad, India", url="u")
    ses.add(ats)
    ses.commit()
    run(ses, cfg, ok_api())
    li = ses.scalar(select(JobRow).where(JobRow.external_id == "4450000001"))
    assert li.dup_of_id == ats.id


def test_daily_pull_failure_is_reported(ses):
    from cortex.daily import run_daily

    def boom(s):
        raise ApifyError("HTTP 401: APIFY_TOKEN rejected")
    seen = {}
    res = run_daily(ses, {}, "u1", TODAY, pull=boom, fetch=None, score=None,
                    digest=lambda s, n: seen.setdefault("n", list(n)) and "sent")
    assert res.failed_steps == ["pull"] and seen["n"] == ["apify FAILED: ApifyError (HTTP 401: APIFY_TOKEN rejected)"]
