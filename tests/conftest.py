import os
import uuid

import pytest


@pytest.fixture(autouse=True)
def _ignore_personal_config(monkeypatch):
    """Tests use the shared config.yaml only, never the developer's config.local.yaml."""
    monkeypatch.setenv("CORTEX_NO_LOCAL_CONFIG", "1")
    monkeypatch.delenv("CORTEX_DB_URL", raising=False)   # .env may point at the real Neon database


@pytest.fixture(autouse=True)
def _postgres_mode(monkeypatch):
    """CORTEX_TEST_PG_URL=postgresql://... runs every test against real Postgres instead of SQLite: each
    sqlite URL a test asks for becomes a fresh, empty Postgres schema (throwaway test server only)."""
    pg = os.getenv("CORTEX_TEST_PG_URL")
    if not pg:
        yield
        return
    import sqlalchemy
    from sqlalchemy.pool import NullPool

    from cortex.db import schema
    real = sqlalchemy.create_engine
    admin = real(schema.db_url(pg), poolclass=NullPool)

    def per_test(url, **kw):
        if not str(url).startswith("sqlite"):
            return real(url, **kw)
        name = "t_" + uuid.uuid4().hex[:16]
        with admin.begin() as c:
            c.exec_driver_sql(f'CREATE SCHEMA "{name}"')
        return real(schema.db_url(pg), poolclass=NullPool, connect_args={"options": f"-csearch_path={name}"})
    monkeypatch.setattr(schema, "create_engine", per_test)
    monkeypatch.setattr(schema, "_ENGINES", {})

    from cortex import graph
    ckpt_schemas: dict[str, str] = {}

    def pg_saver(path):           # same checkpoint file in one test = same schema (a restarted 'process')
        if path not in ckpt_schemas:
            ckpt_schemas[path] = "c_" + uuid.uuid4().hex[:16]
            with admin.begin() as c:
                c.exec_driver_sql(f'CREATE SCHEMA "{ckpt_schemas[path]}"')
        return graph._postgres_saver(pg, search_path=ckpt_schemas[path])
    monkeypatch.setattr(graph, "_sqlite_saver", pg_saver)
    yield
    admin.dispose()
