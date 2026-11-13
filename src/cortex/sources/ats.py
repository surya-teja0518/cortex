"""Public ATS job boards (Greenhouse, Lever, Ashby): the companies' own postings, via their public JSON APIs.

Why: SYNAPSE's LinkedIn/Naukri feed gets thousands of applicants per posting. These boards are where
product companies post first. All three APIs are public, unauthenticated and read-only: no login, no
scraping, no key, $0. Nothing here applies or writes anywhere except CORTEX's own DB.

Safety:
- Board slugs come from config.yaml and are validated (slug regex) before going into a URL.
- Responses: status 200 only, no redirects, size-capped, must be JSON; parsers ignore unknown keys and skip
  malformed postings instead of failing the board. One failing board never stops the others.
- Every text field is untrusted: HTML stripped, sanitized, length-capped, injection-flagged (flagged jobs
  go to manual review like any other source). The description is stored, never executed or followed.
- Only title/location-relevant postings are kept (config filters) so the DB doesn't fill with noise.
- Re-fetching is idempotent (user_id + source + external_id) and never touches status, so an applied
  job stays applied. Then the usual gates, deterministic score and cross-source dedupe run.
- Audit log: per-board counts and HTTP status only, never posting text.

Response shapes follow each vendor's public docs; parsers are defensive because those shapes were not
verified live from the build environment (run `cortex fetch --dry-run` to check against the real APIs).
"""
import html
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from html.parser import HTMLParser

import httpx
from sqlalchemy import select

from cortex.core import parsing as P
from cortex.core.audit import AuditLog
from cortex.core.gates import run_gates
from cortex.core.models import Job
from cortex.core.ratelimit import NoLimit, RateLimiter
from cortex.core.sanitize import sanitize_jd
from cortex.core.scoring import score_job
from cortex.db.schema import JobRow
from cortex.sources.synapse_import import refresh_dups

ATS = ("greenhouse", "lever", "ashby")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
MAX_BYTES = 15_000_000
TIMEOUT = 30.0


def board_url(ats: str, board: str) -> str:
    if ats not in ATS:
        raise ValueError(f"unknown ATS {ats!r} (use one of {', '.join(ATS)})")
    if not _SLUG.match(board or ""):
        raise ValueError(f"invalid board slug {board!r}")
    return {"greenhouse": f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true",
            "lever": f"https://api.lever.co/v0/postings/{board}?mode=json",
            "ashby": f"https://api.ashbyhq.com/posting-api/job-board/{board}"}[ats]


# ---------- text helpers ----------

class _Text(HTMLParser):
    _BLOCK = frozenset({"p", "br", "li", "div", "h1", "h2", "h3", "h4", "tr", "ul", "ol"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []

    def handle_starttag(self, tag, attrs):     # attributes (links, scripts' src...) are dropped
        if tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data)


def html_to_text(s: str) -> str:
    """Greenhouse sends HTML-escaped HTML; Ashby/Lever send HTML too. Keep text only."""
    p = _Text()
    p.feed(html.unescape(s or ""))
    p.close()
    text = "".join(p.out)
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t\xa0]+", " ", text)).strip()


def _s(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _date(v) -> date | None:
    try:
        if isinstance(v, (int, float)) and not isinstance(v, bool):   # Lever: epoch ms
            return datetime.fromtimestamp(v / 1000, UTC).date()
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v).date()   # 3.11+ accepts "Z"
    except (ValueError, OverflowError, OSError):
        pass
    return None


@dataclass
class Posting:
    external_id: str
    title: str
    location: str
    url: str
    description: str = ""
    posted: date | None = None
    work_model: str | None = None
    exp_raw: str | None = None        # source's own experience text; else parsed from the description
    applicants: int | None = None     # scoring signal when the source has it (LinkedIn, Naukri)
    comp: str | None = None


# ---------- parsers (one per ATS; unknown keys ignored, malformed postings skipped) ----------

def parse_greenhouse(data) -> list[Posting]:
    out = []
    for j in (data.get("jobs") if isinstance(data, dict) else None) or []:
        if not isinstance(j, dict) or j.get("id") is None or not _s(j.get("title")):
            continue
        loc = j.get("location")
        out.append(Posting(str(j["id"]), _s(j["title"]), _s(loc.get("name")) if isinstance(loc, dict) else "",
                           _s(j.get("absolute_url")), html_to_text(_s(j.get("content"))),
                           _date(j.get("first_published") or j.get("updated_at"))))
    return out


def parse_lever(data) -> list[Posting]:
    out = []
    for j in data if isinstance(data, list) else []:
        if not isinstance(j, dict) or not _s(j.get("id")) or not _s(j.get("text")):
            continue
        cat = j.get("categories") if isinstance(j.get("categories"), dict) else {}
        locs = [x for x in (cat.get("allLocations") or []) if isinstance(x, str)] or [_s(cat.get("location"))]
        desc = _s(j.get("descriptionPlain")) or html_to_text(_s(j.get("description")))
        lists = " ".join(html_to_text(_s(x.get("content"))) for x in j.get("lists") or [] if isinstance(x, dict))
        out.append(Posting(_s(j["id"]), _s(j["text"]), "; ".join(x for x in locs if x), _s(j.get("hostedUrl")),
                           (desc + "\n" + lists).strip(), _date(j.get("createdAt")),
                           _s(j.get("workplaceType")) or None))
    return out


def parse_ashby(data) -> list[Posting]:
    out = []
    for j in (data.get("jobs") if isinstance(data, dict) else None) or []:
        if not isinstance(j, dict) or not _s(j.get("id")) or not _s(j.get("title")) or j.get("isListed") is False:
            continue
        locs = [_s(j.get("location"))] + [_s(x.get("location")) for x in j.get("secondaryLocations") or []
                                          if isinstance(x, dict)]
        wm = _s(j.get("workplaceType")) or ("Remote" if j.get("isRemote") is True else "")
        out.append(Posting(_s(j["id"]), _s(j["title"]), "; ".join(x for x in locs if x), _s(j.get("jobUrl")),
                           _s(j.get("descriptionPlain")) or html_to_text(_s(j.get("descriptionHtml"))),
                           _date(j.get("publishedAt")), wm or None))
    return out


PARSERS = {"greenhouse": parse_greenhouse, "lever": parse_lever, "ashby": parse_ashby}


# ---------- relevance filters + light enrichment (deterministic, no LLM) ----------

def _any_word(text: str, words) -> bool:
    t = text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(w.lower())}(?![a-z0-9])", t) for w in words)


def relevant(p: Posting, filters: dict) -> bool:
    if not _any_word(p.title, filters.get("title_include", [])):
        return False
    if _any_word(p.title, filters.get("title_exclude", [])):
        return False
    return _any_word(p.location, filters.get("locations", []))


_DB = ("dba", "database", "oracle", "postgres", "postgresql", "mysql", "sql server")
_AI = ("ml", "machine learning", "ai", "llm", "genai", "applied scientist", "mlops")


def infer_track(title: str) -> str:
    if _any_word(title, _DB):
        return "database"
    if _any_word(title, _AI):
        return "ai"
    return "data"


def infer_stack(desc: str) -> str:
    """Only the signals the scoring config knows; anything else is 'verify' (unknown, not guessed)."""
    if _any_word(desc, ("azure",)):
        return "azure"
    if _any_word(desc, ("snowflake", "databricks")):
        return "snowflake_databricks"
    if _any_word(desc, ("oracle",)) and _any_word(desc, ("dba", "database administrator", "database administration")):
        return "dba"
    return "verify"


_YOE = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*(\d{1,2})\s*\+?\s*)?(?:years?|yrs?)"
                  r"(?:\s+of)?[^.\n]{0,40}?\bexperience", re.IGNORECASE)


def infer_exp(desc: str) -> str | None:
    """First 'N(-M)(+) years ... experience' phrase, as text for parse_exp. None when the JD states none."""
    m = _YOE.search(desc or "")
    if not m:
        return None
    lo, hi = int(m.group(1)), m.group(2)
    if lo > 20:
        return None
    return f"{lo}-{hi} yrs" if hi else f"{lo}+ yrs"


# ---------- fetch + upsert ----------

class FetchError(Exception):
    pass


def fetch_board(client: httpx.Client, ats: str, board: str) -> list[Posting]:
    url = board_url(ats, board)
    try:
        r = client.get(url, timeout=TIMEOUT, follow_redirects=False, headers={"Accept": "application/json"})
    except httpx.TimeoutException:
        raise FetchError("timeout") from None
    except httpx.HTTPError as e:
        raise FetchError(type(e).__name__) from None
    if r.status_code != 200:
        raise FetchError(f"HTTP {r.status_code}" + (" (board slug not found?)" if r.status_code == 404 else ""))
    if len(r.content) > MAX_BYTES:
        raise FetchError("response too large")
    try:
        data = r.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise FetchError("not JSON") from None
    return PARSERS[ats](data)


@dataclass
class BoardResult:
    ats: str
    board: str
    company: str
    fetched: int = 0
    kept: int = 0
    inserted: int = 0
    updated: int = 0
    gated: int = 0
    flagged: int = 0
    error: str | None = None
    closed: int = 0          # to_apply jobs from this board that are no longer on it
    new_jobs: list[tuple[str, str, str]] = field(default_factory=list)   # (title, location, band) for display


def _clean(v: str, cap: int) -> str:
    return sanitize_jd(v)[0].strip()[:cap]


def posting_values(p: Posting, source: str, external_id: str, company: str, user_id: str, cfg: dict,
                   today: date, batch: str) -> dict:
    title, loc = _clean(p.title, 256), _clean(p.location, 256)
    desc, flags = sanitize_jd(p.description)
    _, more = sanitize_jd(f"{title} {loc} {company}")
    flags = sorted(set(flags) | set(more))
    url = p.url if p.url.startswith(("https://", "http://")) else ""
    exp_raw = _clean(p.exp_raw or "", 128) or infer_exp(desc)
    min_yoe, max_yoe = P.parse_exp(exp_raw)
    track = infer_track(title)
    job = Job(source=source, external_id=external_id[:128], title=title, company=company,
              location=loc, url=url or "-", min_yoe=min_yoe, max_yoe=max_yoe)
    gate = run_gates(job, cfg)
    res = score_job(track=track, stack=infer_stack(desc), location=loc, min_yoe=min_yoe, applicants=p.applicants,
                    cfg=cfg)
    return {"user_id": user_id, "source": source, "external_id": job.external_id, "batch": batch[:64],
            "company": company, "title": title, "url": url[:2000], "location": loc,
            "work_model": _clean(p.work_model or "", 64) or None, "comp": _clean(p.comp or "", 128) or None,
            "posted": p.posted, "date_added": today, "exp_raw": exp_raw, "min_yoe": min_yoe, "max_yoe": max_yoe,
            "track": track, "score": res.score, "band": res.band, "score_reasons": json.dumps(res.reasons),
            "gate_reason": gate, "sanitizer_flags": json.dumps(flags) if flags else None,
            "description": desc.strip()}


_UPDATABLE = ("title", "url", "location", "work_model", "comp", "posted", "exp_raw", "min_yoe", "max_yoe", "track",
              "score", "band", "score_reasons", "gate_reason", "sanitizer_flags", "description")


def upsert(session, vals: dict, *, own_prefix: str, dry_run: bool) -> str:
    """Insert, or update a row this source created ('updated'). A row another importer created (e.g. the same
    LinkedIn job from the SYNAPSE sheet) only gets a missing description ('enriched'): its track / YOE / score
    are SYNAPSE's judgement and the eval baseline. Never status / applied_date / notes: those are Surya's."""
    existing = session.scalar(select(JobRow).where(JobRow.user_id == vals["user_id"], JobRow.source == vals["source"],
                                                   JobRow.external_id == vals["external_id"]))
    if existing is None:
        if not dry_run:
            session.add(JobRow(**vals))
        return "inserted"
    if (existing.batch or "").startswith(own_prefix):
        if not dry_run:
            for k in _UPDATABLE:
                setattr(existing, k, vals[k])
        return "updated"
    if not existing.description and vals["description"]:
        if not dry_run:
            existing.description = vals["description"]
            if vals["sanitizer_flags"] and not existing.sanitizer_flags:
                existing.sanitizer_flags = vals["sanitizer_flags"]   # a flagged JD still means manual review
        return "enriched"
    return "unchanged"


def close_missing(session, user_id: str, ats: str, board: str, live_ids: set[str], today: date, *,
                  dry_run: bool = False) -> int:
    """A job that is gone from the company's own board is closed: an exact signal, unlike any guess.
    Only to_apply rows this source created; anything Surya acted on (applied, approved...) is never touched."""
    rows = session.scalars(select(JobRow).where(
        JobRow.user_id == user_id, JobRow.source == ats, JobRow.status == "to_apply",
        JobRow.external_id.startswith(f"{board}:"), JobRow.batch.startswith("ats-")))
    n = 0
    for r in rows:
        if r.external_id.split(":", 1)[1] not in live_ids:
            n += 1
            if not dry_run:
                r.status, r.status_raw = "closed", f"auto: gone from {ats} board {today.isoformat()}"
    return n


def fetch_all(session, cfg: dict, user_id: str, *, client: httpx.Client | None = None,
              limiter: RateLimiter | NoLimit | None = None, audit: AuditLog | None = None,
              only: str | None = None, dry_run: bool = False, today: date | None = None) -> list[BoardResult]:
    """Fetch every configured board (or only 'ats:board'), keep relevant postings, upsert, re-run dedupe."""
    scfg = (cfg.get("sources") or {}).get("ats") or {}
    boards = scfg.get("boards") or []
    filters = scfg.get("filters") or {}
    today = today or date.today()  # noqa: DTZ011 (local date)
    limiter = limiter or RateLimiter(cfg.get("rate_limits", {}).get("ats", 30))
    own = client is None
    client = client or httpx.Client()
    results = []
    try:
        for b in boards:
            ats, board = str(b.get("ats", "")).lower(), str(b.get("board", "")).lower()
            if only and only.lower() != f"{ats}:{board}":
                continue
            res = BoardResult(ats, board, _clean(str(b.get("company") or board), 256))
            results.append(res)
            try:
                limiter.wait()
                postings = fetch_board(client, ats, board)
            except (FetchError, ValueError) as e:
                res.error = str(e)
                if audit:
                    audit.log("ats_source", "fetch", user_id=user_id, ats=ats, board=board, error=res.error)
                continue
            res.fetched = len(postings)
            seen = set()
            for p in postings:
                if p.external_id in seen or not relevant(p, filters):
                    continue
                seen.add(p.external_id)
                res.kept += 1
                vals = posting_values(p, ats, f"{board}:{p.external_id}", res.company, user_id, cfg, today,
                                      f"ats-{today.isoformat()}")
                res.gated += vals["gate_reason"] is not None
                res.flagged += vals["sanitizer_flags"] is not None
                what = upsert(session, vals, own_prefix="ats-", dry_run=dry_run)
                if what == "inserted":
                    res.inserted += 1
                    res.new_jobs.append((vals["title"], vals["location"], vals["band"]))
                else:
                    res.updated += 1
            if postings:   # an empty board is more likely a glitch than every job closing at once: close nothing
                res.closed = close_missing(session, user_id, ats, board, {p.external_id for p in postings}, today,
                                           dry_run=dry_run)
            if audit:
                audit.log("ats_source", "fetch", user_id=user_id, ats=ats, board=board, fetched=res.fetched,
                          kept=res.kept, inserted=res.inserted, updated=res.updated, closed=res.closed,
                          dry_run=dry_run)
    finally:
        if own:
            client.close()
    if dry_run:
        session.rollback()
    else:
        session.flush()
        refresh_dups(session, user_id)
        session.commit()
    return results
