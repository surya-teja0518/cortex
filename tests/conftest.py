import pytest


@pytest.fixture(autouse=True)
def _ignore_personal_config(monkeypatch):
    """Tests use the shared config.yaml only, never the developer's config.local.yaml."""
    monkeypatch.setenv("CORTEX_NO_LOCAL_CONFIG", "1")
