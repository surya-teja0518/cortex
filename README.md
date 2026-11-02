# CORTEX

Multi-agent job-search system. Human-in-the-loop by design: it prepares, **you** submit.

## Architecture
![CORTEX architecture v2](docs/architecture-v2.png)

## Pipeline
`Sources → Sanitizer → Dedupe → Gates → Scorer → Tailor → Truth Guard → [HUMAN APPROVAL] → Packet → Tracker`

Rate limiter, audit log (hash-chained) and secrets wrap every external call.

## Results (first real evaluation, 2026-11-02)
An LLM job-relevance scorer (Groq, `openai/gpt-oss-120b`) lifted the ranking of jobs I actually applied to
from the **51st to the 61st percentile**, against a rule-based baseline at chance, on **362 real postings**
(60 applied).

| | Rules | LLM |
|---|---|---|
| Mean percentile rank of applied jobs (50 = random) | 50.8 | **60.9** |
| Pos-vs-neg AUC (15 negatives only) | 0.34 | 0.58 |

The LLM is above chance (z ≈ 2.9); its gap over the rules is borderline (z ≈ 1.9, approximate) and two
leakage checks are still open. Full method, confidence intervals, threats to validity and the questions
this raises: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

## Security principles
- Job text is untrusted: sanitized, injection-flagged, delimited before any LLM sees it
- Every LLM output validated with Pydantic
- Truth Guard blocks resume claims not in the master resume
- No auto-submit, no LinkedIn automation (account safety + ToS)
- Secrets via env only; tamper-evident audit log; `user_id` on every table

## For AI coding agents
Project context lives in `AGENTS.md` (single source of truth). `CLAUDE.md`, `GEMINI.md` and `.cursor/rules` just point to it.

## Run
```
pip install -e ".[dev,ui]" && pytest
cp .env.example .env
# export Google Sheets -> CSV (File > Download > CSV), put files in data/import/ (gitignored)
cortex import data/import/<master>.csv data/import/<batch>.csv   # master first, then batches oldest -> newest
cortex stats
cortex compare        # CORTEX deterministic score vs SYNAPSE Match
cortex eval           # recall@P0-P1, mean percentile rank; precision + AUC once there are >= 10 negatives
cortex tailor <job_id> # tailor -> Truth Guard -> pauses at the approval gate (approve in the dashboard)
cortex llm-check      # ping Groq / Gemini / local Ollama models from config.yaml
cortex bench          # local models on the real tailor task: Truth Guard pass rate + seconds per job
streamlit run src/cortex/ui/app.py --server.address 127.0.0.1
```
Status: v0.3. Implemented: tracker import, gates, deterministic scorer, fuzzy dedupe, review UI
(with Skip decisions), audit log, LLM scorer skeleton (provider interface, Pydantic-validated output,
retry + deterministic fallback; tested with a fake provider), offline eval with cached LLM scores,
master resume schema (`master_resume.example.yaml`; real file gitignored, contact details rejected) and a
deterministic Truth Guard (skills, employers, titles, dates, projects, metrics, certs checked per employer
scope; max 2 retries to the tailor, then reject). Truth Guard does not semantically verify paraphrased prose.
Per-job LangGraph (tailor -> Truth Guard -> human approval `interrupt` -> record) checkpointed to SQLite:
a crash resumes at the gate without re-running the LLM; approval only from the Streamlit dashboard.
LLM router: per-agent fallback chains (scorer: Groq -> Gemini -> local Ollama; tailor: local Ollama only, enforced in code
because the prompt carries the resume). Live provider calls are exercised on the developer's machine; CI uses mocked HTTP.
Not yet: resume PDF + cover note, real LLM eval numbers (pending runs on the developer machine), Apify adapter.
