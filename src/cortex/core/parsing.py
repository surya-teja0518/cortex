"""Normalizers for messy tracker/scraper fields. Pure functions, heavily tested."""
import hashlib
import re
from datetime import date

CITIES = ["hyderabad", "bengaluru", "bangalore", "pune", "mumbai", "chennai",
          "gurgaon", "gurugram", "noida", "delhi", "kolkata", "ahmedabad"]
_CITY_ALIAS = {"bangalore": "bengaluru", "gurugram": "gurgaon"}
_COMPANY_NOISE = {"inc", "llc", "ltd", "limited", "pvt", "private", "india", "technologies",
                  "technology", "solutions", "software", "global", "services", "corporation",
                  "corp", "group", "co"}


def parse_exp(raw) -> tuple[int | None, int | None]:
    """'~3-6 yrs'->(3,6); '6+ yrs (stretch)'->(6,None); 'Associate (~2-5 yrs)'->(2,5);
    '5'->(5,None); '4.5'->(4,None); 'Verify'/'Mid-Senior'/''->(None,None)."""
    s = str(raw or "").strip().lower()
    if not s or s == "verify":
        return None, None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)", s)
    if m:
        return int(float(m.group(1))), int(float(m.group(2)))
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return int(float(m.group(1))), None
    return None, None


def extract_external_id(url: str) -> str:
    """Trailing numeric job id (LinkedIn/Naukri); stable hash fallback for odd/truncated URLs."""
    u = (url or "").strip()
    path = u.split("?")[0].split("#")[0].rstrip("/")
    m = re.search(r"(\d{6,})$", path)
    if m:
        return m.group(1)
    return "h" + hashlib.sha1(u.encode()).hexdigest()[:16]


def city_of(location: str) -> str:
    low = (location or "").lower()
    for c in CITIES:
        if c in low:
            return _CITY_ALIAS.get(c, c)
    return re.sub(r"[^a-z]", "", low.split(",")[0]) or "unknown"


def normalize_company(name: str) -> str:
    s = re.sub(r"\(.*?\)", " ", (name or "").lower())
    toks = [t for t in re.findall(r"[a-z0-9]+", s) if t not in _COMPANY_NOISE]
    return " ".join(toks) or (name or "").lower().strip()


def normalize_track(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r.startswith("database"):
        return "database"
    if r == "ai":
        return "ai"
    return "data"      # 'Data' and hybrid 'Data+AI' score on the data base


def normalize_source(raw: str) -> str:
    r = (raw or "").strip().lower()
    return r if r in {"linkedin", "naukri"} else (r or "unknown")


def stack_category(text: str) -> str:
    t = (text or "").lower()
    if "azure" in t:
        return "azure"
    if "aws" in t or "gcp" in t or "stretch" in t:
        return "aws_gcp_native"
    if "strong-adjacent" in t or "snowflake" in t or "databricks" in t:
        return "snowflake_databricks"
    if "dba" in t:
        return "dba"
    if "ai" in t or "ml" in t:
        return "ai_ml"
    return "verify"


_STATUS = {"to apply": "to_apply", "applied": "applied", "closed": "closed",
           "not accepting": "not_accepting", "not accepting applications": "not_accepting",
           "experience doesn't match": "exp_mismatch", "interview": "interview",
           "offer": "offer", "rejected": "rejected"}
STATUS_RANK = {"to_apply": 0, "closed": 1, "not_accepting": 1, "exp_mismatch": 1,
               "applied": 2, "interview": 3, "rejected": 3, "offer": 4}


def normalize_status(raw: str) -> str:
    r = (raw or "").strip().lower()
    return _STATUS.get(r) or (re.sub(r"[^a-z0-9]+", "_", r).strip("_") or "to_apply")


def parse_date(s) -> date | None:
    try:
        return date.fromisoformat(str(s).strip()[:10])
    except (ValueError, TypeError):
        return None
