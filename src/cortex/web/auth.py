"""Google sign-in (OpenID Connect, authorization code + PKCE), for exactly ONE allowed address.

- state + nonce + PKCE verifier travel in a signed, 10-minute cookie; the callback checks all three.
- The ID token comes straight from Google's token endpoint over TLS (OIDC Core 3.1.3.7: signature check may be
  skipped for tokens received this way); we still check iss, aud, exp, nonce, email_verified and the address.
- Session = a signed cookie (itsdangerous, HMAC-SHA1 over the server secret; NOT encrypted, so it holds only the
  email, issue time and a CSRF token). HttpOnly, Secure (https), SameSite=Lax, 7 days.
"""
import base64
import hashlib
import hmac
import json
import secrets
import time
from urllib.parse import urlencode

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from cortex.web.settings import Settings

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
SESSION_COOKIE = "cortex_session"
FLOW_COOKIE = "cortex_oauth"
FLOW_SECONDS = 600


class AuthError(Exception):
    pass


def _ser(s: Settings, salt: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(s.session_secret, salt=salt)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def start(s: Settings) -> tuple[str, str]:
    """(Google URL to redirect to, signed flow cookie value)."""
    state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    challenge = _b64(hashlib.sha256(verifier.encode()).digest())
    q = {"client_id": s.client_id, "redirect_uri": s.redirect_uri, "response_type": "code",
         "scope": "openid email", "state": state, "nonce": nonce, "code_challenge": challenge,
         "code_challenge_method": "S256", "prompt": "select_account", "login_hint": s.allowed_email}
    return f"{AUTH_URL}?{urlencode(q)}", _ser(s, "flow").dumps({"state": state, "nonce": nonce, "v": verifier})


def _claims(id_token: str) -> dict:
    try:
        payload = id_token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError) as e:
        raise AuthError("malformed id token") from e


def finish(s: Settings, flow_cookie: str | None, state: str, code: str, *,
           client: httpx.Client | None = None, now: float | None = None) -> str:
    """Verify the callback and return the signed-in (allowed) email. Raises AuthError otherwise."""
    if not flow_cookie:
        raise AuthError("sign-in expired: start again")
    try:
        flow = _ser(s, "flow").loads(flow_cookie, max_age=FLOW_SECONDS)
    except (BadSignature, SignatureExpired) as e:
        raise AuthError("sign-in expired: start again") from e
    if not state or not hmac.compare_digest(str(flow.get("state", "")), state):
        raise AuthError("state mismatch")
    if not code or len(code) > 2048:
        raise AuthError("missing code")
    c = client or httpx.Client(timeout=15)
    try:
        r = c.post(TOKEN_URL, data={"code": code, "client_id": s.client_id, "client_secret": s.client_secret,
                                    "redirect_uri": s.redirect_uri, "grant_type": "authorization_code",
                                    "code_verifier": flow["v"]}, follow_redirects=False)
    except httpx.HTTPError as e:
        raise AuthError(f"token exchange failed ({type(e).__name__})") from None
    if r.status_code != 200:
        raise AuthError(f"token exchange failed (HTTP {r.status_code})")
    claims = _claims(str((r.json() or {}).get("id_token", "")))
    now = now or time.time()
    if claims.get("iss") not in ISSUERS or claims.get("aud") != s.client_id:
        raise AuthError("token not issued for this app")
    if float(claims.get("exp", 0)) < now:
        raise AuthError("token expired")
    if not hmac.compare_digest(str(claims.get("nonce", "")), str(flow.get("nonce", ""))):
        raise AuthError("nonce mismatch")
    email = str(claims.get("email", "")).lower()
    if claims.get("email_verified") is not True or not hmac.compare_digest(email, s.allowed_email):
        raise AuthError("this Google account is not allowed")
    return email


def new_session(s: Settings, email: str) -> str:
    return _ser(s, "session").dumps({"email": email, "csrf": secrets.token_urlsafe(32), "iat": int(time.time())})


def read_session(s: Settings, cookie: str | None) -> dict | None:
    if not cookie:
        return None
    try:
        data = _ser(s, "session").loads(cookie, max_age=s.session_days * 86400)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict) or not hmac.compare_digest(str(data.get("email", "")), s.allowed_email):
        return None          # signed for someone else (allowed address changed): not valid any more
    return data
