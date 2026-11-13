from pathlib import Path

from cortex.agents.workday_fill import (
    FieldInfo,
    build_plan,
    inspect,
    is_workday,
    load_fmap,
    plan_fills,
    run,
)
from tests.test_workday import CONTACT, MASTER, TAILORED

ROOT = Path(__file__).parent.parent
FMAP = load_fmap(ROOT / "workday_fields.yaml")
WD = "https://acme.wd5.myworkdayjobs.com/en-US/careers/job/Hyderabad/Data-Engineer_R123/apply"


def F(i, aid, kind="text", empty=True, anc=(), label=""):
    return FieldInfo(i, aid, kind, empty, tuple(anc), label)


class FakePage:
    """Pages are a list of field lists; 'next' moves on each round (like him clicking Next)."""
    def __init__(self, pages, url=WD):
        self.pages, self.round, self._url = pages, 0, url
        self.typed, self.attached = [], []

    def url(self):
        return self._url

    def fields(self):
        return self.pages[min(self.round, len(self.pages) - 1)]

    def type_into(self, i, v):
        self.typed.append((i, v))

    def attach(self, i, p):
        self.attached.append((i, p))

    def is_open(self):
        return self.round < len(self.pages)


def sleeper(page):
    def s(_):
        page.round += 1
    return s


def plan(tmp_path=None):
    pdf = None
    if tmp_path:
        pdf = tmp_path / "resume.pdf"
        pdf.write_bytes(b"%PDF-1.4")
    return build_plan(MASTER, TAILORED, CONTACT, None, pdf)


def test_is_workday_only_https_workday_hosts():
    assert is_workday(WD) and is_workday("https://x.myworkday.com/a") and is_workday("https://wd3.workday.com/x")
    for bad in ("http://acme.wd5.myworkdayjobs.com/x", "https://myworkdayjobs.com.evil.example/x",
                "https://evil-myworkdayjobs.com/x", "https://linkedin.com/jobs/1", "", "javascript:alert(1)"):
        assert not is_workday(bad)


def test_plan_has_every_job_newest_first_with_tailored_bullets():
    p = plan()
    assert p.simple["first_name"] == "Alex" and p.simple["last_name"] == "Example"
    assert [j.fields["company"] for j in p.jobs] == [r.employer for r in sorted(MASTER.roles, key=lambda r: r.start,
                                                                                 reverse=True)]
    tailored = next(j for j in p.jobs if j.fields["title"] == "Associate Data Engineer")
    assert "Migrated 120 tables to Snowflake." in tailored.fields["description"] and tailored.start == ("06", "2021")


def test_fills_only_empty_known_text_fields_never_passwords():
    fields = [F(0, "legalNameSection_firstName"), F(1, "legalNameSection_lastName", empty=False),
              F(2, "password", kind="password"), F(3, "email", kind="password"), F(4, "unknownThing"),
              F(5, "email"), F(6, "phone-number", kind="other")]
    got = {i: v for i, v, _ in plan_fills(fields, plan(), FMAP, set())}
    assert got == {0: "Alex", 5: "alex@example.com"}       # filled last name, password-kind and unknown untouched


def test_work_experience_blocks_and_dates_in_order():
    p = plan()
    fields = []
    for n, cid in enumerate(["workExperience-3", "workExperience-12"]):    # numeric order, not string order
        base = 10 * n
        fields += [F(base + 1, "jobTitle", anc=["formField-jobTitle", cid]),
                   F(base + 2, "company", anc=[cid]),
                   F(base + 3, "description", kind="textarea", anc=[cid]),
                   F(base + 4, "dateSectionMonth-input", anc=["dateSection", "formField-startDate", cid]),
                   F(base + 5, "dateSectionYear-input", anc=["dateSection", "formField-startDate", cid])]
    got = {i: v for i, v, _ in plan_fills(fields, p, FMAP, set())}
    j0, j1 = p.jobs[0], p.jobs[1]
    assert got[1] == j0.fields["title"] and got[2] == j0.fields["company"]
    assert (got[4], got[5]) == j0.start and (got[14], got[15]) == j1.start
    assert got[12] == j1.fields["company"]


def test_current_job_end_date_left_blank():
    p = plan()
    current = next(i for i, j in enumerate(p.jobs) if j.end is None)
    cid = f"workExperience-{current + 1}"
    fields = [F(1, "dateSectionMonth-input", anc=["formField-endDate", cid])]
    blocks = [F(100 + k, "company", anc=[f"workExperience-{k + 1}"]) for k in range(len(p.jobs))]
    assert all(i != 1 for i, _, _ in plan_fills(fields + blocks, p, FMAP, set()))


def test_run_never_refills_a_field_he_cleared_and_stops_off_workday(tmp_path):
    page1 = [F(0, "legalNameSection_firstName"), F(1, "file-upload-input-ref", kind="file")]
    page = FakePage([page1, page1, page1])                      # he cleared first name after round 1
    st = run(page, plan(tmp_path), FMAP, sleep=sleeper(page))
    assert page.typed == [(0, "Alex")] and len(page.attached) == 1 and st.filled == 2
    other = FakePage([page1], url="https://evil.example/apply")
    st2 = run(other, plan(), FMAP, sleep=sleeper(other))
    assert other.typed == [] and st2.skipped_non_workday == 1


def test_inspect_prints_ids_never_values():
    page = FakePage([[F(0, "legalNameSection_firstName", label="First Name", empty=False)]])
    out = "\n".join(inspect(page))
    assert "legalNameSection_firstName" in out and "First Name" in out and "Alex" not in out
