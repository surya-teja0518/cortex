"""Per-agent LLM router: an ordered fallback chain plus a data-sensitivity rule enforced in code.

    scorer: [groq, gemini, ollama_small]   public job text -> fastest first, fall back on failure
    tailor: [ollama]                       Surya's resume -> local only

- personal_data=True routes refuse, at construction, any provider not marked personal_data.
  A config mistake fails loudly instead of sending the resume to a cloud API.
- On ProviderError (network, 429, 5xx, bad model, context overflow) the next provider is tried.
  Bad *output* is not the router's business: agents validate and retry.
- Each attempt is audit-logged (provider, outcome, error type, seconds). Never the prompt.
- `served_by` records which provider produced the last answer, for eval and debugging.
The router exposes both agent interfaces (score_job for the scorer, complete for the tailor).
"""
import time

from cortex.core.audit import AuditLog
from cortex.llm.providers import ProviderError, build_provider, inline_refs


class AllProvidersFailed(ProviderError):
    pass


class PrivacyViolation(ValueError):
    pass


class Router:
    def __init__(self, route: str, providers: list, audit: AuditLog, *, personal_data: bool = False,
                 schema: dict | None = None):
        if not providers:
            raise ValueError(f"route {route!r} has no providers")
        if personal_data:
            leaky = [p.name for p in providers if not getattr(p, "personal_data", False)]
            if leaky:
                raise PrivacyViolation(f"route {route!r} carries personal data; not allowed on: {leaky}")
        self.route = route
        self.providers = providers
        self.audit = audit
        self.personal_data = personal_data
        self.schema = inline_refs(schema) if schema else None
        self.name = "+".join(p.name for p in providers)   # stable id for eval caching
        self.served_by: str | None = None
        self.last_error: str | None = None   # provider's own message (already key-redacted), for diagnostics

    def complete(self, prompt: str) -> str:
        errors = []
        for p in self.providers:
            t0 = time.monotonic()
            try:
                out = p.complete(prompt, schema=self.schema)
            except ProviderError as e:
                errors.append(type(e).__name__)
                self.last_error = str(e)[:300]
                self.audit.log("llm_router", "call", route=self.route, provider=p.name, outcome="error",
                               error=type(e).__name__, seconds=round(time.monotonic() - t0, 2))
                continue
            self.served_by = p.name
            self.audit.log("llm_router", "call", route=self.route, provider=p.name, outcome="ok",
                           seconds=round(time.monotonic() - t0, 2))
            return out
        self.served_by = None
        raise AllProvidersFailed(f"route {self.route!r}: all providers failed ({', '.join(errors)})")

    score_job = complete   # scorer interface


def build_route(route: str, cfg: dict, audit: AuditLog, *, schema: dict | None = None,
                only: str | None = None, client=None) -> Router:
    """Build a route from config.yaml `llm`. `only` pins one provider (e.g. for a fair per-provider eval)."""
    llm = cfg["llm"]
    names = [only] if only else llm["routes"][route]
    unknown = [n for n in names if n not in llm["providers"]]
    if unknown:
        raise ValueError(f"unknown provider(s) {unknown}; configured: {sorted(llm['providers'])}")
    providers = [build_provider(n, llm["providers"][n], client=client) for n in names]
    return Router(route, providers, audit, personal_data=route in llm.get("personal_data_routes", []),
                  schema=schema)
