import pytest

from cortex.core.parsing import (
    city_of,
    extract_external_id,
    normalize_company,
    normalize_status,
    normalize_track,
    parse_exp,
    stack_category,
)


@pytest.mark.parametrize("raw,exp", [
    ("~3-6 yrs", (3, 6)), ("6+ yrs (stretch)", (6, None)), ("Associate (~2-5 yrs)", (2, 5)),
    ("5", (5, None)), ("4.5", (4, None)), ("3-5", (3, 5)), ("2-14 yrs", (2, 14)), ("0", (0, None)),
    ("Verify", (None, None)), ("Mid-Senior", (None, None)), ("", (None, None)), (None, (None, None)),
    ("2+ yrs", (2, None)), ("3-5 yrs (disclosed)", (3, 5)),
])
def test_parse_exp(raw, exp):
    assert parse_exp(raw) == exp


def test_external_id():
    assert extract_external_id("https://in.linkedin.com/jobs/view/4460313691") == "4460313691"
    assert extract_external_id("https://in.linkedin.com/jobs/view/senior-de-at-x-4474060690?trk=1") == "4474060690"
    assert extract_external_id("https://www.naukri.com/job-listings-x-270826924443") == "270826924443"
    a = extract_external_id("https://www.naukri.com/job-listings-truncated-2-t...")
    assert a.startswith("h") and a == extract_external_id("https://www.naukri.com/job-listings-truncated-2-t...")


def test_city_and_company():
    assert city_of("Greater Hyderabad Area") == "hyderabad"
    assert city_of("Bangalore, Karnataka") == "bengaluru"
    assert city_of("Hyderabad / Bengaluru") == "hyderabad"
    assert normalize_company("Onit India") == normalize_company("Onit")
    assert normalize_company("Milestone Technologies, Inc.") == "milestone"
    assert normalize_company("Amazon Web Services (AWS)") == "amazon web"


def test_stack_status_track():
    assert stack_category("Azure-bullseye") == "azure"
    assert stack_category("Strong-adjacent") == "snowflake_databricks"
    assert stack_category("Stack-stretch (AWS/GCP-native)") == "aws_gcp_native"
    assert stack_category("AWS-stretch") == "aws_gcp_native"
    assert stack_category("DBA-crossover") == "dba"
    assert stack_category("AI/ML") == "ai_ml"
    assert stack_category("Verify-stack") == "verify"
    assert normalize_status("To Apply") == "to_apply"
    assert normalize_status("Experience Doesn't match") == "exp_mismatch"
    assert normalize_track("Data+AI") == "data"
    assert normalize_track("Database") == "database"
