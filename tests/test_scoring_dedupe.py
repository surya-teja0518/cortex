from types import SimpleNamespace as R

from cortex.core.config import load_config
from cortex.core.dedupe import find_dups, is_fuzzy_dup
from cortex.core.gates import run_gates
from cortex.core.models import Job
from cortex.core.scoring import score_job

CFG = load_config("config.yaml")


def S(**kw):
    base = dict(track="data", stack="verify", location="Pune", min_yoe=None, applicants=None, cfg=CFG)
    return score_job(**{**base, **kw})


def test_score_rules_exact():
    # base 64 + azure 7 + hyd 3 + exp stated 2 + min_yoe<=3 2 = 78 -> P1
    r = S(stack="azure", location="Hyderabad", min_yoe=3)
    assert (r.score, r.band) == (78, "P1")
    # base 61 + dba 5 + exp unstated -3 = 63 -> P3
    r = S(track="database", stack="dba")
    assert (r.score, r.band) == (63, "P3")
    # base 64 -6 aws + 2 exp - 1 (min=5) = 59
    assert S(stack="aws_gcp_native", min_yoe=5).score == 59
    # min=4 -> +1 ; applicants<30 +2, >200 -2
    assert S(min_yoe=4, applicants=10).score == 64 - 2 + 2 + 1 + 2
    assert S(min_yoe=4, applicants=500).score == 64 - 2 + 2 + 1 - 2


def test_band_edges():
    assert S(stack="azure", location="Hyderabad", min_yoe=3, applicants=5).band == "P1"   # 80
    r = S(track="database", stack="azure", location="Hyderabad", min_yoe=3, applicants=5)  # 61+7+3+2+2+2=77
    assert (r.score, r.band) == (77, "P2")


def J(title, yoe=None):
    return Job(source="t", external_id="1", title=title, company="X", url="u", min_yoe=yoe)


def test_gates_word_boundary():
    assert run_gates(J("Lead Data Engineer"), CFG)
    assert run_gates(J("Engineering Manager, Data"), CFG)
    assert run_gates(J("Head of Data"), CFG)
    assert run_gates(J("VP Data Platform"), CFG)
    assert run_gates(J("Leadership Analyst"), CFG) is None          # not a whole-word match
    assert run_gates(J("Data Engineer"), CFG) is None
    assert run_gates(J("Data Engineer", yoe=8), CFG)
    assert run_gates(J("Data Engineer", yoe=5), CFG) is None        # 5 is allowed (> 5 rejects)
    assert run_gates(J("Data Engineer", yoe=None), CFG) is None     # unstated -> verify, not rejected


def test_fuzzy_dedupe_rules():
    assert is_fuzzy_dup("Software Engineer - Data", "Software Engineer - Data")
    assert not is_fuzzy_dup("Data Engineer II", "Data Engineer III")
    assert not is_fuzzy_dup("Senior Data Engineer", "Data Engineer")
    rows = [R(id=1, company="Onit India", title="Software Engineer - Data", location="Pune Division, Maharashtra"),
            R(id=2, company="Onit", title="Software Engineer - Data", location="Pune Division, Maharashtra"),
            R(id=3, company="EY", title="Consultant", location="Hyderabad, Telangana"),
            R(id=4, company="EY", title="Consultant", location="Pune City, Maharashtra")]
    assert find_dups(rows) == {2: 1}      # same role, different city is NOT a dup
