"""Config: config.yaml (shared, in git) + optional config.local.yaml (yours, gitignored) merged on top.

Put personal settings in config.local.yaml (e.g. linkedin_check.enabled, ATS boards, schedule time) so
`git pull` never collides with your edits. Dicts merge key by key; any other value (lists included) replaces
the shared one. Set CORTEX_NO_LOCAL_CONFIG=1 to ignore the local file (the test suite does).
"""
import os
from pathlib import Path

import yaml

LOCAL_NAME = "config.local.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_config(path: str | None = None) -> dict:
    p = Path(path or os.getenv("CORTEX_CONFIG", "config.yaml"))
    cfg = yaml.safe_load(p.read_text()) or {}
    local = p.with_name(LOCAL_NAME)
    if not os.getenv("CORTEX_NO_LOCAL_CONFIG") and local.exists() and local.resolve() != p.resolve():
        over = yaml.safe_load(local.read_text()) or {}
        if not isinstance(over, dict):
            raise ValueError(f"{local} must be a YAML mapping")
        cfg = deep_merge(cfg, over)
    return cfg
