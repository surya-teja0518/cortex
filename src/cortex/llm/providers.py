"""LLM providers: Groq, Gemini (free cloud tiers) and Ollama (local). One interface: complete(prompt) -> str.

Rules every provider follows:
- The API key is read once via core.secrets and never appears in repr, errors or logs
  (Gemini gets it in a header, not the URL, so it can't leak through a logged URL).
- Every call waits on the provider's own rate limiter (free tiers have per-provider limits).
- Failures raise ProviderError subclasses with the status code and a redacted, truncated reason.
  The router decides whether to fall back; agents decide what to do with bad output.
- `personal_data` says whether prompts containing Surya's resume may be sent here. Only local
  providers set it; the router enforces it.
"""
import copy

import httpx

from cortex.core.ratelimit import RateLimiter
from cortex.core.secrets import get_secret


class ProviderError(Exception):
    """Provider unusable for this call (network, 5xx, bad request, bad model). Router may fall back."""


class RateLimitedError(ProviderError):
    """429 / quota exhausted."""


class ContextOverflowError(ProviderError):
    """Prompt does not fit the model's configured context; refusing instead of silently truncating."""


def inline_refs(schema: dict) -> dict:
    """Resolve Pydantic's $defs/$ref into one self-contained schema (grammar-based decoders need this)."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


class _HTTPProvider:
    name = "base"
    personal_data = False
    timeout = 60.0

    def __init__(self, model: str, rpm: int, client: httpx.Client | None = None):
        self.model = model
        self.limiter = RateLimiter(rpm)
        self._client = client or httpx.Client(timeout=self.timeout)
        self._secret: str | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _redact(self, text: str) -> str:
        return text.replace(self._secret, "***") if self._secret else text

    def _post(self, url: str, json: dict, headers: dict) -> dict:
        self.limiter.wait()
        try:
            r = self._client.post(url, json=json, headers=headers)
        except httpx.TimeoutException:
            raise ProviderError(f"{self.name}: timeout") from None
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: {type(e).__name__}") from None
        if r.status_code == 429:
            raise RateLimitedError(f"{self.name}: 429 rate limited")
        if r.status_code >= 400:
            reason = " ".join(self._redact(r.text).split())[:200]   # one line, key redacted, truncated
            raise ProviderError(f"{self.name}: HTTP {r.status_code}: {reason}")
        try:
            return r.json()
        except ValueError:
            raise ProviderError(f"{self.name}: non-JSON response") from None


class GroqProvider(_HTTPProvider):
    """OpenAI-compatible chat API. JSON mode on; the prompt must mention JSON (ours do)."""
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str, rpm: int = 25, max_tokens: int = 2048, client: httpx.Client | None = None):
        super().__init__(model, rpm, client)
        self._secret = get_secret("GROQ_API_KEY")
        self.max_tokens = max_tokens

    def complete(self, prompt: str, schema: dict | None = None) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": self.max_tokens, "response_format": {"type": "json_object"}}
        data = self._post(self.URL, body, {"Authorization": f"Bearer {self._secret}"})
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("groq: unexpected response shape") from None


class GeminiProvider(_HTTPProvider):
    """Generative Language API. Key in x-goog-api-key header. JSON output mode; Pydantic validates."""
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, rpm: int = 10, max_tokens: int = 2048, client: httpx.Client | None = None):
        super().__init__(model, rpm, client)
        self._secret = get_secret("GEMINI_API_KEY")
        self.max_tokens = max_tokens

    def complete(self, prompt: str, schema: dict | None = None) -> str:
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens,
                                     "responseMimeType": "application/json"}}
        data = self._post(self.URL.format(model=self.model), body, {"x-goog-api-key": self._secret})
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            # Blocked or empty candidates come back without content
            raise ProviderError("gemini: no content in response") from None


class OllamaProvider(_HTTPProvider):
    """Local Ollama. Nothing leaves the machine, so it may receive personal data.

    - num_ctx is always sent: Ollama's default window is small and it truncates silently.
    - The prompt is refused up front if it obviously can't fit, and the response is checked
      afterwards: if the model read as many tokens as the window holds, it was probably cut.
    - think=False disables reasoning output on thinking models (Qwen3, Granite 4.x); set
      think=None for models that reject the field.
    - schema (if given) is passed as `format`, so decoding is constrained to valid JSON of that shape.
    """
    name = "ollama"
    personal_data = True
    timeout = 600.0
    CHARS_PER_TOKEN = 3  # conservative estimate for the up-front check

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", num_ctx: int = 12288,
                 think: bool | None = False, rpm: int = 120, client: httpx.Client | None = None):
        super().__init__(model, rpm, client)
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.think = think

    def complete(self, prompt: str, schema: dict | None = None) -> str:
        if len(prompt) / self.CHARS_PER_TOKEN > self.num_ctx:
            raise ContextOverflowError(f"{self.name}: prompt ~{len(prompt) // self.CHARS_PER_TOKEN} tokens "
                                       f"> num_ctx {self.num_ctx}")
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                "format": inline_refs(schema) if schema else "json",
                "options": {"num_ctx": self.num_ctx, "temperature": 0}}
        if self.think is not None:
            body["think"] = self.think
        data = self._post(f"{self.base_url}/api/chat", body, {})
        if data.get("prompt_eval_count", 0) >= self.num_ctx:
            raise ContextOverflowError(f"{self.name}: prompt filled the {self.num_ctx}-token window (truncated)")
        try:
            return data["message"]["content"]
        except (KeyError, TypeError):
            raise ProviderError(f"{self.name}: unexpected response shape") from None


class FakeLLM:
    """Test double with the provider interface: returns/raises queued responses, records prompts."""

    def __init__(self, responses: list, name: str = "fake", personal_data: bool = False):
        self._responses = list(responses)
        self.name = name
        self.personal_data = personal_data
        self.prompts: list[str] = []
        self.schemas: list[dict | None] = []

    def complete(self, prompt: str, schema: dict | None = None) -> str:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def build_provider(name: str, spec: dict, client: httpx.Client | None = None):
    """Build one provider from config.yaml `llm.providers.<name>`."""
    kind = spec.get("kind", name)
    common = {"client": client} if client else {}
    if kind == "groq":
        return GroqProvider(spec["model"], rpm=spec.get("rpm", 25), max_tokens=spec.get("max_tokens", 2048), **common)
    if kind == "gemini":
        return GeminiProvider(spec["model"], rpm=spec.get("rpm", 10), max_tokens=spec.get("max_tokens", 2048), **common)
    if kind == "ollama":
        return OllamaProvider(spec["model"], base_url=spec.get("base_url", "http://127.0.0.1:11434"),
                              num_ctx=spec.get("num_ctx", 12288), think=spec.get("think", False),
                              rpm=spec.get("rpm", 120), **common)
    raise ValueError(f"unknown provider kind: {kind}")


__all__ = ["ContextOverflowError", "FakeLLM", "GeminiProvider", "GroqProvider", "OllamaProvider",
           "ProviderError", "RateLimitedError", "build_provider", "inline_refs"]
