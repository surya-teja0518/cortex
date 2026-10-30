"""Deterministic scorer (SYNAPSE rules). No LLM, fully explainable, config-driven."""
from .models import ScoreResult


def band_for(score: int, bands: dict) -> str:
    if score >= bands["P0"]:
        return "P0"
    if score >= bands["P1"]:
        return "P1"
    if score >= bands["P2"]:
        return "P2"
    return "P3"


def score_job(*, track: str, stack: str, location: str, min_yoe: int | None,
              applicants: int | None, cfg: dict) -> ScoreResult:
    s = cfg["scoring"]
    score = s["base"][track]
    reasons = [f"base {track} {score}"]

    def add(label: str, delta: int):
        nonlocal score
        if delta:
            score += delta
            reasons.append(f"{label} {delta:+d}")

    add(f"stack:{stack}", s["stack"].get(stack, s["stack"]["verify"]))
    if "hyderabad" in (location or "").lower():
        add("hyderabad", s["hyderabad"])
    if applicants is not None:
        a = s["applicants"]
        if applicants < a["low_below"]:
            add("applicants<%d" % a["low_below"], a["low_bonus"])
        elif applicants > a["high_above"]:
            add("applicants>%d" % a["high_above"], a["high_penalty"])
    add("exp stated" if min_yoe is not None else "exp unstated",
        s["exp_stated"] if min_yoe is not None else s["exp_unstated"])
    if min_yoe is not None:
        m = s["min_yoe"]
        if min_yoe <= 3:
            add("min_yoe<=3", m["le3"])
        elif min_yoe == 4:
            add("min_yoe=4", m["eq4"])
        elif min_yoe == 5:
            add("min_yoe=5", m["eq5"])
    score = max(0, min(100, score))
    return ScoreResult(track=track, score=score, band=band_for(score, s["bands"]), reasons=reasons)
