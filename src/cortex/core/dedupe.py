"""Fuzzy duplicate detection. Flags (never deletes): same company + same city + near-identical title.

Seniority/level tokens must match exactly so 'Data Engineer II' != 'Data Engineer III'
and 'Senior Data Engineer' != 'Data Engineer'. Different cities are never duplicates
(e.g. the same EY role posted for Hyderabad and Pune).
"""
import re
from collections import defaultdict
from difflib import SequenceMatcher

from .parsing import city_of, normalize_company

LEVEL_TOKENS = {"i", "ii", "iii", "iv", "v", "1", "2", "3", "4", "5", "l1", "l2", "l3",
                "senior", "sr", "junior", "jr", "associate", "intern", "principal", "staff", "lead"}


def _tokens(title: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (title or "").lower())


def is_fuzzy_dup(title_a: str, title_b: str, threshold: float = 0.9) -> bool:
    ta, tb = _tokens(title_a), _tokens(title_b)
    if {t for t in ta if t in LEVEL_TOKENS} != {t for t in tb if t in LEVEL_TOKENS}:
        return False
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio() >= threshold


def find_dups(rows, threshold: float = 0.9) -> dict[int, int]:
    """rows: objects with .id .company .title .location. Returns {dup_id: canonical_id}."""
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: r.id):
        groups[(normalize_company(r.company), city_of(r.location))].append(r)
    out: dict[int, int] = {}
    for items in groups.values():
        canon: list = []
        for r in items:
            hit = next((c for c in canon if is_fuzzy_dup(c.title, r.title, threshold)), None)
            if hit:
                out[r.id] = hit.id
            else:
                canon.append(r)
    return out
