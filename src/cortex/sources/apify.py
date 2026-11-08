"""SYNAPSE/Apify adapter: read the datasets SYNAPSE's own Apify runs already produced. Replaces the
"export the Google Sheet to CSV" step.

- READ ONLY. It never starts an actor run (runs cost money; SYNAPSE's schedule starts them) and never
  scrapes anything itself: CORTEX reads its owner's Apify account through the official API with his token.
  This is the "SYNAPSE/Apify adapter" source AGENTS.md allows; nothing touches LinkedIn directly.
- Actors (config sources.apify.actors): valig~linkedin-jobs-scraper and memo23~naukri-scraper. Field names
  were checked against real dataset rows from those runs (2026-11-08), not guessed.
- Rows are keyed exactly like the SYNAPSE sheet importer keys them (source linkedin|naukri, external_id =
  the job's numeric id), so a job already imported from the sheet merges instead of duplicating. Such rows
  only get a missing description; their track / YOE / score stay as the sheet set them (eval baseline).
- Same safety as the ATS source: sanitize + injection flags, filters, gates, deterministic score (now with
  the applicant count SYNAPSE used), cross-source dedupe, audit = counts only. The token goes in the
  Authorization header, never in a URL, a log line or an error message.
"""
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import httpx

from cortex.core.audit import AuditLog
from cortex.core.ratelimit import NoLimit, RateLimiter
from cortex.core.secrets import get_secret
from cortex.sources.ats import (
    Posting,
    _clean,
    html_to_text,
    posting_values,
    relevant,
    upsert,
)
from cortex.sources.synapse_import import refresh_dups

API = "https://api.apify.com/v2"
_ACTOR = re.compile(r"^[A-Za-z0-9_.-]{1,64}~[A-Za-z0-9_.-]{1,64}$")
_DATASET = re.compile(r"^[A-Za-z0-9]{6,32}$")
MAX_ITEMS = 1000
TIMEOUT = 60.0


class ApifyError(Exception):
    pass


# ---------- parsers (real field names from the two actors' datasets) ----------

def _s(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _int(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def parse_applicants(text) -> int | None:
    """'Over 200 applicants' -> 201 (more than 200); '57 applicants' -> 57; 'Be among the first 25' -> 25."""
    t = _s(text).lower()
    m = re.search(r"(\d[\d,]*)", t)
    if not m:
        return None
    n = int(m.group(1).replace(",", ""))
    return n + 1 if t.startswith("over") else n


def _date(v) -> date | None:
    """ISO ('2026-10-31T00:00:00.000Z', LinkedIn) or 'YYYY-MM-DD HH:MM:SS' (Naukri). Only the date is kept."""
    s = _s(v)
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").date()  # noqa: DTZ007 (date only)
    except ValueError:
        return None


def parse_linkedin(item) -> tuple[Posting, str] | None:
    """valig~linkedin-jobs-scraper row -> (Posting, company)."""
    if not isinstance(item, dict) or not _s(item.get("id")).isdigit() or not _s(item.get("title")):
        return None
    desc = _s(item.get("description")) or html_to_text(_s(item.get("descriptionHtml")))
    return (Posting(_s(item["id"]), _s(item["title"]), _s(item.get("location")), _s(item.get("url")), desc,
                    _date(item.get("postedDate")), applicants=parse_applicants(item.get("applicationsCount")),
                    comp=_s(item.get("salary")) or None), _s(item.get("companyName")))


_NK_COMPANY = re.compile(r"^https://www\.naukri\.com/([a-z0-9-]{1,100}?)-jobs-careers-\d+$")


def company_from_slug(url) -> str:
    """'https://www.naukri.com/lpl-financials-jobs-careers-5507842' -> 'Lpl Financials'. Many real rows have
    companyDetail=null and no basicInfo (checked 2026-11-08: 6 of the first 9); this is their only company field."""
    m = _NK_COMPANY.match(_s(url))
    return " ".join(w.capitalize() for w in m.group(1).split("-")) if m else ""


def parse_naukri(item) -> tuple[Posting, str] | None:
    """memo23~naukri-scraper row (Naukri shape; NaukriGulf rows have no jobId and are skipped)."""
    if not isinstance(item, dict) or not _s(item.get("jobId")).isdigit() or not _s(item.get("title")):
        return None
    cd = item.get("companyDetail") if isinstance(item.get("companyDetail"), dict) else {}
    bi = item.get("basicInfo") if isinstance(item.get("basicInfo"), dict) else {}
    company = _s(cd.get("name")) or _s(bi.get("companyName")) or company_from_slug(item.get("staticCompanyName"))
    locs = [_s(x.get("label")) for x in item.get("locations") or [] if isinstance(x, dict)]
    lo, hi = _int(item.get("minimumExperience")), _int(item.get("maximumExperience"))
    exp = f"{lo}-{hi} yrs" if lo is not None and hi is not None else (f"{lo}+ yrs" if lo is not None else None)
    sal = item.get("salaryDetail") if isinstance(item.get("salaryDetail"), dict) else {}
    comp = None if sal.get("hideSalary") is True else (_s(sal.get("label")) or None)
    desc = "\n".join(x for x in (_s(item.get("shortDescription")), html_to_text(_s(item.get("description")))) if x)
    url = _s(item.get("url")) or _s(item.get("staticUrl"))
    return (Posting(_s(item["jobId"]), _s(item["title"]), "; ".join(x for x in locs if x), url, desc,
                    _date(item.get("createdDate")), _s(item.get("wfhLabel")) or None, exp,
                    _int(item.get("applyCount")), comp), company)


PARSERS = {"linkedin": parse_linkedin, "naukri": parse_naukri}


# ---------- API ----------

class ApifyClient:
    def __init__(self, token: str, client: httpx.Client | None = None, limiter=None):
        self._token = token
        self._client = client or httpx.Client(timeout=TIMEOUT)
        self._limiter = limiter or NoLimit()

    def __repr__(self) -> str:
        return "ApifyClient(token=***)"

    def _get(self, path: str, params: dict):
        self._limiter.wait()
        try:
            r = self._client.get(f"{API}{path}", params=params, follow_redirects=False,
                                 headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"})
        except httpx.TimeoutException:
            raise ApifyError("timeout") from None
        except httpx.HTTPError as e:
            raise ApifyError(type(e).__name__) from None
        if r.status_code == 401:
            raise ApifyError("HTTP 401: APIFY_TOKEN rejected")
        if r.status_code != 200:
            raise ApifyError(f"HTTP {r.status_code}")
        try:
            return r.json()
        except ValueError:
            raise ApifyError("not JSON") from None

    def recent_datasets(self, actor: str, since: datetime, limit: int = 10) -> list[str]:
        """Default datasets of this account's SUCCEEDED runs of `actor` started after `since`, newest first."""
        if not _ACTOR.match(actor):
            raise ApifyError(f"invalid actor id {actor!r} (use username~actor-name)")
        data = self._get(f"/acts/{actor}/runs", {"status": "SUCCEEDED", "desc": "true", "limit": limit})
        items = (data.get("data") or {}).get("items") if isinstance(data, dict) else None
        out = []
        for run in items or []:
            if not isinstance(run, dict):
                continue
            started = _s(run.get("startedAt"))
            ds = _s(run.get("defaultDatasetId"))
            try:
                when = datetime.fromisoformat(started)
            except ValueError:
                continue
            if when >= since and _DATASET.match(ds):
                out.append(ds)
        return out

    def items(self, dataset: str, limit: int = MAX_ITEMS) -> list:
        if not _DATASET.match(dataset):
            raise ApifyError("invalid dataset id")
        data = self._get(f"/datasets/{dataset}/items", {"clean": "true", "format": "json", "limit": limit})
        if not isinstance(data, list):
            raise ApifyError("unexpected dataset response")
        return data


# ---------- import ----------

@dataclass
class ActorResult:
    actor: str
    source: str
    datasets: int = 0
    items: int = 0
    kept: int = 0
    inserted: int = 0
    updated: int = 0
    enriched: int = 0        # sheet rows that got their missing description
    gated: int = 0
    flagged: int = 0
    error: str | None = None
    new_jobs: list[tuple[str, str, str, str]] = field(default_factory=list)   # company, title, location, band


def pull_all(session, cfg: dict, user_id: str, *, client: ApifyClient | None = None, audit: AuditLog | None = None,
             dry_run: bool = False, today: date | None = None, now: datetime | None = None) -> list[ActorResult]:
    acfg = (cfg.get("sources") or {}).get("apify") or {}
    actors = acfg.get("actors") or []
    filters = acfg.get("filters") or ((cfg.get("sources") or {}).get("ats") or {}).get("filters") or {}
    since = (now or datetime.now(UTC)) - timedelta(days=int(acfg.get("since_days", 8)))
    today = today or date.today()  # noqa: DTZ011 (local date)
    if client is None:
        token = get_secret("APIFY_TOKEN", required=False)
        if not token:
            raise ApifyError("APIFY_TOKEN missing in .env")
        client = ApifyClient(token, limiter=RateLimiter(cfg.get("rate_limits", {}).get("apify", 10)))
    results = []
    for a in actors:
        actor, source = str(a.get("actor", "")), str(a.get("source", "")).lower()
        res = ActorResult(actor, source)
        results.append(res)
        if source not in PARSERS:
            res.error = f"unknown source {source!r} (linkedin or naukri)"
            continue
        try:
            datasets = client.recent_datasets(actor, since, int(acfg.get("max_runs", 10)))
            res.datasets = len(datasets)
            seen = set()
            for ds in datasets:
                for item in client.items(ds, int(acfg.get("max_items", MAX_ITEMS))):
                    res.items += 1
                    parsed = PARSERS[source](item)
                    if parsed is None:
                        continue
                    p, company = parsed
                    if p.external_id in seen or not company or not relevant(p, filters):
                        continue
                    seen.add(p.external_id)
                    res.kept += 1
                    vals = posting_values(p, source, p.external_id, _clean(company, 256), user_id, cfg, today,
                                          f"apify-{today.isoformat()}")
                    res.gated += vals["gate_reason"] is not None
                    res.flagged += vals["sanitizer_flags"] is not None
                    what = upsert(session, vals, own_prefix="apify-", dry_run=dry_run)
                    if what == "inserted":
                        res.inserted += 1
                        res.new_jobs.append((vals["company"], vals["title"], vals["location"], vals["band"]))
                    elif what in ("updated", "enriched"):
                        setattr(res, what, getattr(res, what) + 1)
        except ApifyError as e:
            res.error = str(e)
        if audit:
            audit.log("apify_source", "pull", user_id=user_id, apify_actor=actor, datasets=res.datasets, items=res.items,
                      kept=res.kept, inserted=res.inserted, error=res.error, dry_run=dry_run)
    if dry_run:
        session.rollback()
    else:
        session.flush()
        refresh_dups(session, user_id)
        session.commit()
    return results
