"""Vercel entry point for the hosted dashboard (Phase 3). All settings come from Vercel environment variables;
the app refuses to start if any security setting is missing (cortex.web.settings)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cortex.web.app import create_app
from cortex.web.settings import load_settings

app = create_app(load_settings())
