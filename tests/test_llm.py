import json
from pathlib import Path

import httpx
import pytest

from cortex.agents.scorer import LLMScore, LLMScorer, ScoringInput
from cortex.core.audit import AuditLog
from cortex.core.models import ScoreResult
from cortex.core.resume import TailoredResume
from cortex.llm.providers import (
    ContextOverflowError,
    FakeLLM,
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    ProviderError,
    RateLimitedError,
    inline_refs,
)
from cortex.llm.router import AllProvidersFailed, PrivacyViolation, Router, build_route

GROQ_KEY, GEMINI_KEY = "gsk_TESTKEY_should_never_leak_123", "AIzaTESTKEY_should_never_leak_456"


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)


def mock(handler):
    """httpx client whose every request goes to handler(request) -> httpx.Response. No network."""
    seen = []

    def h(req):
        seen.append(req)
        return handler(req)
    return httpx.Client(transport=httpx.MockTransport(h)), seen


def audit(tmp_path):
    return AuditLog(str(tmp_path / "a.log"))


# ---------------- providers: request shape + parsing ----------------

def test_groq_request_and_parse():
    c, seen = mock(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}))
    p = GroqProvider("llama-x", client=c)
    assert p.complete("say json") == '{"ok": true}'
    req = seen[0]
    body = json.loads(req.content)
    assert req.url == GroqProvider.URL and req.headers["authorization"] == f"Bearer {GROQ_KEY}"
    assert body["model"] == "llama-x" and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}


def test_groq_extra_params_and_config_build():
    from cortex.core.config import load_config
    from cortex.llm.providers import build_provider
    c, seen = mock(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}))
    spec = load_config("config.yaml")["llm"]["providers"]["groq"]
    build_provider("groq", spec, client=c).complete("json please")
    body = json.loads(seen[0].content)
    assert body["model"] == spec["model"] and body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}


def test_config_pins_versions_not_latest_aliases():
    from cortex.core.config import load_config
    models = [p["model"] for p in load_config("config.yaml")["llm"]["providers"].values()]
    assert not [m for m in models if "latest" in m]


def test_gemini_key_in_header_not_url():
    c, seen = mock(lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"a":'}, {"text": "1}"}]}}]}))
    p = GeminiProvider("gemini-x", client=c)
    assert p.complete("p") == '{"a":1}'
    req = seen[0]
    assert GEMINI_KEY not in str(req.url) and req.headers["x-goog-api-key"] == GEMINI_KEY
    assert "models/gemini-x:generateContent" in str(req.url)
    assert json.loads(req.content)["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_blocked_response_is_provider_error():
    c, _ = mock(lambda r: httpx.Response(200, json={"candidates": [{"finishReason": "SAFETY"}]}))
    with pytest.raises(ProviderError, match="no content"):
        GeminiProvider("g", client=c).complete("p")


def test_ollama_request_shape():
    c, seen = mock(lambda r: httpx.Response(200, json={"message": {"content": "{}"}, "prompt_eval_count": 50}))
    schema = TailoredResume.model_json_schema()
    OllamaProvider("qwen3:8b", num_ctx=12288, client=c).complete("p", schema=schema)
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://127.0.0.1:11434/api/chat"
    assert body["options"]["num_ctx"] == 12288 and body["think"] is False and body["stream"] is False
    assert "$defs" not in json.dumps(body["format"]) and "$ref" not in json.dumps(body["format"])
    assert body["format"]["properties"]["roles"]["items"]["properties"]["employer"]


def test_ollama_think_omitted_when_none():
    c, seen = mock(lambda r: httpx.Response(200, json={"message": {"content": "{}"}}))
    OllamaProvider("m", think=None, client=c).complete("p")
    body = json.loads(seen[0].content)
    assert "think" not in body and body["format"] == "json"


def test_ollama_refuses_prompt_that_cannot_fit():
    c, seen = mock(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ContextOverflowError):
        OllamaProvider("m", num_ctx=1000, client=c).complete("x" * 4000)
    assert seen == []                                        # never sent


def test_ollama_detects_silent_truncation():
    c, _ = mock(lambda r: httpx.Response(200, json={"message": {"content": "{}"}, "prompt_eval_count": 4096}))
    with pytest.raises(ContextOverflowError, match="truncated"):
        OllamaProvider("m", num_ctx=4096, client=c).complete("short")


@pytest.mark.parametrize("status,exc", [(429, RateLimitedError), (500, ProviderError), (404, ProviderError)])
def test_http_errors(status, exc):
    c, _ = mock(lambda r: httpx.Response(status, text=f"bad request for key {GROQ_KEY}"))
    with pytest.raises(exc) as e:
        GroqProvider("m", client=c).complete("p")
    assert GROQ_KEY not in str(e.value)                     # redacted even if the API echoes it


def test_short_secret_does_not_mangle_errors(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "y")
    c, _ = mock(lambda r: httpx.Response(400, text="API key not valid"))
    with pytest.raises(ProviderError, match="API key not valid"):
        GroqProvider("m", client=c).complete("p")


def test_network_errors_become_provider_errors():
    def boom(r):
        raise httpx.ConnectError("refused")
    c, _ = mock(boom)
    with pytest.raises(ProviderError, match="ConnectError"):
        OllamaProvider("m", client=c).complete("p")


def test_keys_never_in_repr(monkeypatch):
    assert GROQ_KEY not in repr(GroqProvider("m")) and GEMINI_KEY not in repr(GeminiProvider("m"))
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqProvider("m")


def test_inline_refs():
    s = inline_refs(TailoredResume.model_json_schema())
    assert "$ref" not in json.dumps(s) and "$defs" not in s


# ---------------- router: fallback, privacy, audit ----------------

def test_router_falls_back_and_records_who_served(tmp_path):
    a = FakeLLM([RateLimitedError("groq: 429")], name="groq")
    b = FakeLLM(['{"x": 1}'], name="gemini")
    r = Router("scorer", [a, b], audit(tmp_path))
    assert r.complete("p") == '{"x": 1}' and r.served_by == "gemini"
    recs = [json.loads(x)["details"] for x in (tmp_path / "a.log").read_text().splitlines()]
    assert [(x["provider"], x["outcome"]) for x in recs] == [("groq", "error"), ("gemini", "ok")]
    assert recs[0]["error"] == "RateLimitedError" and "p" not in json.dumps(recs).replace('"provider"', "")


def test_router_all_fail(tmp_path):
    r = Router("scorer", [FakeLLM([ProviderError("x")], name="a"), FakeLLM([ProviderError("y")], name="b")],
               audit(tmp_path))
    with pytest.raises(AllProvidersFailed):
        r.complete("p")
    assert r.served_by is None


def test_router_does_not_swallow_programming_errors(tmp_path):
    r = Router("scorer", [FakeLLM([KeyError("bug")], name="a"), FakeLLM(["{}"], name="b")], audit(tmp_path))
    with pytest.raises(KeyError):
        r.complete("p")


def test_privacy_route_refuses_cloud_providers(tmp_path):
    local = FakeLLM([], name="ollama:q", personal_data=True)
    cloud = FakeLLM([], name="gemini")
    Router("tailor", [local], audit(tmp_path), personal_data=True)               # fine
    with pytest.raises(PrivacyViolation, match="gemini"):
        Router("tailor", [local, cloud], audit(tmp_path), personal_data=True)


def test_config_tailor_route_is_local_only(tmp_path):
    from cortex.core.config import load_config
    cfg = load_config("config.yaml")
    assert "tailor" in cfg["llm"]["personal_data_routes"]
    r = build_route("tailor", cfg, audit(tmp_path))
    assert all(p.personal_data for p in r.providers)
    with pytest.raises(PrivacyViolation):
        build_route("tailor", cfg, audit(tmp_path), only="gemini")              # can't pin a cloud model either
    with pytest.raises(PrivacyViolation):
        bad = {**cfg, "llm": {**cfg["llm"], "routes": {**cfg["llm"]["routes"], "tailor": ["groq"]}}}
        build_route("tailor", bad, audit(tmp_path))


def test_build_route_unknown_provider(tmp_path):
    from cortex.core.config import load_config
    with pytest.raises(ValueError, match="unknown provider"):
        build_route("scorer", load_config("config.yaml"), audit(tmp_path), only="openai")


def test_router_passes_inlined_schema(tmp_path):
    f = FakeLLM(["{}"], name="ollama:x", personal_data=True)
    Router("tailor", [f], audit(tmp_path), schema=TailoredResume.model_json_schema()).complete("p")
    assert "$defs" not in f.schemas[0]


# ---------------- end to end: scorer agent over a real-shaped HTTP provider ----------------

class NoWait:
    def wait(self):
        pass


def test_scorer_over_router_with_http_fallback(tmp_path):
    calls = []

    def groq(r):
        calls.append("groq")
        return httpx.Response(429)

    def gemini(r):
        calls.append("gemini")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": json.dumps({"score": 88, "reasons": ["Azure fit"], "red_flags": []})}]}}]})
    a = audit(tmp_path)
    router = Router("scorer", [GroqProvider("g", client=mock(groq)[0]), GeminiProvider("m", client=mock(gemini)[0])],
                    a, schema=LLMScore.model_json_schema())
    det = ScoreResult(track="data", score=70, band="P2", reasons=["base"])
    out = LLMScorer(router, NoWait(), a, {"P0": 85, "P1": 78, "P2": 70}).score(
        ScoringInput(source="t", external_id="1", title="Data Engineer", company="X"), det)
    assert (out.score, out.band, out.source) == (88, "P0", "llm") and calls == ["groq", "gemini"]
    text = Path(tmp_path / "a.log").read_text()
    assert GROQ_KEY not in text and GEMINI_KEY not in text and a.verify()
