"""Dashboard settings, all from the environment (Vercel env vars or .env). Refuses to start when anything
security-relevant is missing or weak: no silent fallback to an open dashboard."""
import os
from dataclasses import dataclass
from urllib.parse import urlsplit


class SettingsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    db_url: str
    user_id: str
    allowed_email: str          # the ONLY Google account that may sign in
    client_id: str
    client_secret: str
    session_secret: str
    public_url: str             # https://<deployment> (or http://127.0.0.1:8000 for local use)
    session_days: int = 7

    @property
    def origin(self) -> str:
        u = urlsplit(self.public_url)
        return f"{u.scheme}://{u.netloc}"

    @property
    def secure(self) -> bool:
        return self.public_url.startswith("https://")

    @property
    def redirect_uri(self) -> str:
        return self.origin + "/auth/callback"


def _local(url: str) -> bool:
    return urlsplit(url).hostname in ("127.0.0.1", "localhost")


def load_settings(env=os.environ) -> Settings:
    from cortex.core import secrets  # noqa: F401  (loads .env for local runs)
    need = {k: (env.get(k) or "").strip() for k in (
        "CORTEX_DB_URL", "CORTEX_ALLOWED_EMAIL", "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET",
        "CORTEX_SESSION_SECRET", "CORTEX_PUBLIC_URL")}
    missing = [k for k, v in need.items() if not v]
    if missing:
        raise SettingsError("dashboard not configured, missing: " + ", ".join(missing))
    if len(need["CORTEX_SESSION_SECRET"]) < 32:
        raise SettingsError("CORTEX_SESSION_SECRET must be at least 32 characters (use a random value)")
    url = need["CORTEX_PUBLIC_URL"].rstrip("/")
    if not (url.startswith("https://") or (url.startswith("http://") and _local(url))):
        raise SettingsError("CORTEX_PUBLIC_URL must be https:// (plain http only for 127.0.0.1/localhost)")
    if "@" not in need["CORTEX_ALLOWED_EMAIL"]:
        raise SettingsError("CORTEX_ALLOWED_EMAIL must be an email address")
    return Settings(db_url=need["CORTEX_DB_URL"], user_id=(env.get("CORTEX_USER_ID") or "surya").strip(),
                    allowed_email=need["CORTEX_ALLOWED_EMAIL"].lower(), client_id=need["GOOGLE_OAUTH_CLIENT_ID"],
                    client_secret=need["GOOGLE_OAUTH_CLIENT_SECRET"], session_secret=need["CORTEX_SESSION_SECRET"],
                    public_url=url)
