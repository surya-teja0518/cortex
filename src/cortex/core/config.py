import os
from pathlib import Path

import yaml


def load_config(path: str | None = None) -> dict:
    p = Path(path or os.getenv("CORTEX_CONFIG", "config.yaml"))
    return yaml.safe_load(p.read_text())
