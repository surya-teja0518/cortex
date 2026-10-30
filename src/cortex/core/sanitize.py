"""Treat all job text as untrusted. Strip control chars, flag injection, wrap in delimiters."""
import re

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard .{0,40}instructions",
    r"you are now",
    r"system prompt",
    r"reveal .{0,30}(prompt|secret|key)",
    r"<\s*/?\s*(system|assistant)\s*>",
]
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_LEN = 20_000


def sanitize_jd(text: str) -> tuple[str, list[str]]:
    """Return (clean_text, flags). Flagged jobs go to manual review, never auto-processed."""
    text = _CTRL.sub("", text or "")[:MAX_LEN]
    flags = [p for p in INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]
    return text, flags


_DELIM = re.compile(r"<\s*/?\s*untrusted_job_text\s*>", re.IGNORECASE)


def wrap_untrusted(text: str) -> str:
    # Strip any open/close delimiter variant so job text cannot fake a boundary.
    safe = _DELIM.sub("", text)
    return f"<untrusted_job_text>\n{safe}\n</untrusted_job_text>"
