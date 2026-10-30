"""Secrets access. Env/.env only; never log values."""
import os
from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str, required: bool = True) -> str | None:
    val = os.getenv(name)
    if required and not val:
        raise RuntimeError(f"Missing required secret: {name}")
    return val
