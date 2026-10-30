# CORTEX — Project Context (AGENTS.md: read this first, every session, any coding agent)

## What this is
CORTEX is a multi-agent job-search system built by Surya Teja Vajjhala (data engineer / Oracle DBA, Hyderabad, ~4 YOE, targeting product-company Data Engineer / Database / AI-ML Engineer roles, switch target Dec 2026).

Three goals, in priority order:
1. **Use it on his own job search** (replaces the manual SYNAPSE Google-Sheet workflow).
2. **Capstone project for his resume** (AI Engineer roles). Must be defensible in interviews: every claim in the README must be true.
3. **Monetize later** (hence `user_id` on every table from day one).

Hard constraints: **zero runtime cost, near-zero maintenance, security first** (most vibe-coded apps skip security; this one must not). Free LLM tiers (Gemini, Groq) with Ollama as local fallback. SQLite via SQLAlchemy (Postgres later = config change). GitHub Actions cron or local cron for scheduling.

## Architecture (see docs/architecture-v2.png — LOCKED, do not redesign)
Sequential LangGraph state machine, not a parallel hub:

`Sources → Sanitizer → Dedupe → Deterministic Gates → Scorer → Tailor → Truth Guard → [HUMAN APPROVAL GATE] → Packet → Tracker`

- **Sources (v1, limited and safe):** SYNAPSE/Apify adapter, public ATS JSON (Greenhouse, Lever, Ashby), RSS. NO direct LinkedIn scraping.
- **Sanitizer:** strip control chars, flag prompt injection, wrap JD in `<untrusted_job_text>` delimiters. Flagged jobs go to manual review, never auto-processed. (`src/cortex/core/sanitize.py`)
- **Dedupe:** exact + fuzzy on (source, external_id) and company+title.
- **Deterministic gates (no LLM, run first):** title blocklist, YOE cap, location/remote rules, company blocklist. (`core/gates.py`, rules in `config.yaml`)
- **Scorer:** LLM relevance scoring + rules. If LLM is down/rate-limited: keep deterministic score, flag "LLM unscored", continue pipeline.
- **Tailor:** tailored resume PDF + cover note from templates. Output validated with Pydantic.
- **Truth Guard:** diff every claim (skills, experience, projects) against the master resume facts. Unsupported content → send back to Tailor with failure reason, **max 2 retries**, then Reject (logged).
- **Human Approval Gate:** explicit LangGraph `interrupt`. Nothing downstream runs without approval. **Approval only via the Streamlit dashboard in v1 (no Telegram approval).**
- **Packet Agent:** builds an approval packet (resume PDF, cover note, direct apply link, pre-filled answers). **NEVER auto-submits applications.** Surya submits manually.
- **Tracker Agent:** log application, track status, reminders, reports.
- **Cross-cutting on every external call:** Rate limiter, Secrets manager (gitignored `.env`, never logged; OS keyring later, optional — do NOT claim "encrypted"), hash-chained Audit logger, Pydantic validation, `user_id`.
- **Outputs:** Streamlit dashboard (local-only, with login before any exposure), Telegram notifier (output only), Eval harness (offline).
- **LangGraph checkpointing in SQLite** so a crash resumes instead of restarting.

## Security rules (non-negotiable)
- Job text is untrusted input. Never concatenate it into a prompt without delimiters. Never let it trigger tool calls.
- Every LLM output must validate against a Pydantic model; invalid → retry/fallback, never pass through.
- No secrets in code, logs, commits, or error messages. `.env` is gitignored.
- No LinkedIn automation, no auto-apply, no scraping behind logins (account-ban + ToS risk).
- Audit log is append-only and hash-chained; tampering must fail `verify()`.
- Dashboard binds to localhost; add auth before exposing anywhere.
- Never invent resume content. Truth Guard is the backstop, not the excuse.

## Current state (as of 2026-10-31)
Step 1 (MVP) and Step 2 (scorer skeleton + eval) merged to main; 85 tests passing.
- `core/`: `gates.py` (word-boundary title blocklist + YOE cap), `scoring.py` (deterministic SYNAPSE rules, config-driven), `dedupe.py` (fuzzy; same company + same city + same level tokens), `parsing.py`, `sanitize.py` (delimiter stripping repeats until stable), `audit.py` (hash chain), `ratelimit.py`, `secrets.py`, `config.py`, `models.py`
- `sources/synapse_import.py`: imports SYNAPSE tracker CSV exports (master + dated batches); idempotent upsert; never downgrades status (Applied stays Applied); sanitizes and flags injection; keeps SYNAPSE's own Match/Priority in `synapse_score`/`synapse_priority` for evaluation
- `db/schema.py`: `jobs` (full 23-column tracker fields + cortex score/band/gate/dup/flags), `approvals`, `skips`, `llm_score_cache`; `user_id` everywhere
- `cli.py`: `cortex import <csv...>`, `cortex stats`, `cortex compare` (CORTEX score vs SYNAPSE Match), `cortex eval`
- `review.py` + `ui/app.py`: Streamlit review queue (local-only; optional `CORTEX_UI_PASSWORD`) with "Skip selected" (skips = negative labels)
- `agents/scorer.py`: `ScoringInput` (structured tracker fields; description optional), provider protocol + `FakeProvider`. **Gemini/Groq are still stubs** (read keys, raise NotImplementedError; no live calls). All fields sanitized and inside one `<untrusted_job_text>` block; injection in ANY field -> no LLM call. Output = `LLMScore` (int or integral-float 0-100, no bools, extra fields forbidden). Bad output / provider error -> one retry -> deterministic score + `llm_unscored`. Rate-limited; audit logs outcome + error type only.
- `eval.py` + `cortex eval`: positives = applied/interview/offer; recall@P0-P1 and mean percentile rank, deterministic vs LLM; gated positives reported separately; LLM results cached by (user, job, provider, prompt sha256). Precision deliberately not reported until negatives (skips) exist.
- Tracker headers verified from the real sheets (23 cols A-W). Master `Exp Req` is free text ("~3-6 yrs"); batches use bare numbers or "Verify". Header is "Comp (est/disclosed)" in master, "Comp" in batches. Both handled.
- KNOWN FINDING: documented SYNAPSE scoring rules reproduce SYNAPSE's stored Match only loosely (on the fixture CORTEX scores ~6 points lower, band agreement 33%; deterministic recall@P0-P1 on fixture positives = 0%). Do NOT tune weights to hit Match blindly; evaluate against Surya's real apply/skip decisions.
Step 3 in progress: master resume schema + Truth Guard (branch feat/truth-guard).
NOT built yet: live Gemini/Groq calls, precision metrics, Apify adapter, tailor, approval gate (LangGraph interrupt), packet, tracker agent, scheduler, Telegram.

## Build order (each step must be usable on its own)
1. **MVP (DONE, v0.2):** SYNAPSE import → dedupe → gates → deterministic score → Streamlit review list. Next for this step: import real Batch016 export, then the older batches (master first).
2. LLM scorer + **eval harness** (compare LLM score vs deterministic score vs Surya's apply/skip decisions; precision/recall). This earns the resume line.
3. Tailor + Truth Guard + approval gate (LangGraph interrupt, SQLite checkpointer).
4. Packet + Telegram notifier + scheduler.
Credit note (Claude Code only): Surya has $100 cloud-session credit expiring **Nov 5, 2026 1:29 PM IST** — front-load steps 1–3. One narrow task per session, tests must pass before finishing.

## SYNAPSE (existing pipeline to reuse as a source)
- Scrapes via Apify actors (valig/linkedin-jobs-scraper, memo23/naukri-scraper); runs Tue and Fri 10 AM IST.
- Gates already used: title blocklist (Lead, Principal, Staff, Architect, Manager, Director, Head of, VP, SVP, Chief); reject if min YOE > 5.
- Deterministic scoring: base Data 64 / Database 61 / AI 62, plus adjustments for stack, location, applicant count.
- Priority bands: **P0 ≥ 85, P1 78–84, P2 70–77, P3 < 70**.
- Tracker: Google Drive sheet with a frozen master and dated batch sheets (Batch010–015), 23 columns. The Drive connector can read but not edit cells. **Read the actual header row from the sheet before designing the importer; do not guess column names.**
- Applications are never auto-submitted.

## Surya's resume tracks (inputs for Tailor / Truth Guard)
Three master resumes: Data Engineer (default), Oracle DBA, AI/ML. Source of truth for Truth Guard = a structured `master_resume.yaml` (to be created from his resumes: skills, roles, projects, metrics). Key facts to encode:
- CGI (PNC Data Streaming Platform), Jul 2025–present: Oracle DBA + data engineering, 8 production Oracle DBs, Kafka-based flows, Oracle DB KPI Automation Platform (Flask/React/Docker/OpenShift), ML change-risk prediction and IRVS (document verification) done as sprint work.
- Cognizant Oct 2024–May 2025 (Molina): PySpark, Azure Databricks, ADF, Power BI. Jul 2022–Sep 2024 (Wynn): Snowflake migration, SCD, Airflow, Streamlit ETL testing.
- Certs: DP-203, DP-900, Google Cloud Gemini Enterprise Agent Development.
- Metrics in resumes must be copied exactly from the resume files, never recomputed or embellished.

## Conventions
- Python 3.11+, type hints, Pydantic v2, SQLAlchemy 2.0, pytest, ruff.
- Small modules, one agent per file under `src/cortex/agents/`, sources under `src/cortex/sources/`.
- Every new module ships with tests. Run `pytest` before finishing a task.
- Commits: conventional style (`feat:`, `fix:`, `docs:`, `test:`).
- Keep the README honest: only claim what is implemented and tested.

## Working style with Surya
Direct, no hand-holding, push-hard feedback. Address him as Surya or Sir. Tell him the truth even when unwelcome. He is a capable data engineer; skip basics, but explain LangGraph/agent concepts since this is his path into AI engineering (he is also working through the "AI Engineering from Scratch" course; the target is a deployable AI app by Jan 2027).
