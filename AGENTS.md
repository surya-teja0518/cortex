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
- `agents/scorer.py`: `ScoringInput` (structured tracker fields; description optional), provider protocol + `FakeProvider`. Real providers live in `llm/`. All fields sanitized and inside one `<untrusted_job_text>` block; injection in ANY field -> no LLM call. Output = `LLMScore` (int or integral-float 0-100, no bools, extra fields forbidden). Bad output / provider error -> one retry -> deterministic score + `llm_unscored`. Rate-limited; audit logs outcome + error type only.
- `eval.py` + `cortex eval`: positives = applied/interview/offer; negatives = status exp_mismatch + Skip decisions (not_accepting/closed are unlabelled: employer's call). recall@P0-P1 and mean percentile rank always; precision@P0-P1 and pos-vs-neg AUC once >= 10 negatives (flagged ROUGH: exp_mismatch negatives favour the YOE-aware deterministic scorer). Gated positives/negatives reported separately; LLM results cached by (user, job, provider, prompt sha256).
- Tracker headers verified from the real sheets (23 cols A-W). Master `Exp Req` is free text ("~3-6 yrs"); batches use bare numbers or "Verify". Header is "Comp (est/disclosed)" in master, "Comp" in batches. Both handled.
- FIRST LLM RESULT (2026-11-02, groq gpt-oss-120b): mean pct rank 60.9 vs rules 50.8; AUC 0.58 vs 0.34 (15 negatives); recall@P0-P1 30% from a 17% top-band share (lift ~1.75x, z~2.6); prec 78% = base rate (80%), NOT evidence. Above chance z~2.9; gap vs rules borderline z~1.9. Notes leakage keyword check clean (applied 7.9% vs to_apply 9.3%). Open: stack_fit ablation (it is SYNAPSE's own verdict), paired bootstrap, groq_qwen comparison. Evidence + interview Q&A: docs/EVALUATION.md. Never state the claim without these caveats.
- REAL BASELINE (2026-11-02, master + Batch011, 362 jobs, 60 applied): deterministic recall@P0-P1 = 0%, mean percentile rank of applied jobs = 50.8 (= random). Bands: 0 P0, 3 P1, 128 P2, 239 P3: with tracker data (no applicant counts) the rules top out at 78, P0 is unreachable. This is the bar the LLM scorer must beat. Do NOT compare against SYNAPSE's stored Match: Surya chose applications by looking at it (circular).
- KNOWN FINDING: documented SYNAPSE scoring rules reproduce SYNAPSE's stored Match only loosely (on the fixture CORTEX scores ~6 points lower, band agreement 33%; deterministic recall@P0-P1 on fixture positives = 0%). Do NOT tune weights to hit Match blindly; evaluate against Surya's real apply/skip decisions.
Step 3: master resume schema + Truth Guard merged (PR #2). Real `master_resume.yaml` built from all 3 resumes (gitignored; Surya holds the copy). Decisions recorded in it: both CGI titles valid (title + alt_titles), Java/dbt dropped, Jenkins only in the KPI platform, "real-time" overclaims reworded, IRVS = CGI sprint work.
- `core/resume.py`: `MasterResume` (extra=forbid, no contact details, role `alt_titles`) and `TailoredResume` (tailor output contract).
- `agents/truth_guard.py`: deterministic claim checks scoped per employer/project; catches unit+plus metrics (2TB+) and plural terms (LLMs). `guarded_tailor`: 1 attempt + 2 retries, then reject. Limit: a number is checked to exist in the same scope, not bound to the same claim.
- Approval gate in progress (branch feat/approval-gate, 150 tests):
  - `agents/tailor.py`: prompt has 3 trust zones (<master_resume> trusted, <untrusted_job_text>, <truth_guard_feedback> with tags neutralised); returns parsed JSON; guarded_tailor validates + retries.
  - `graph.py`: LangGraph `tailor_and_guard -> approval (interrupt) -> record`, SqliteSaver checkpointer (`data/checkpoints.db`), one thread per run. Injection-flagged job -> manual_review (tailor never called). `submit_decision` validates `ApprovalDecision` (approve|reject, extra=forbid) before resuming; node validates again. No side effects before `interrupt()` (the node re-runs on resume); `record` is idempotent.
  - `tailored_resumes` table (status per run); `review.pending_approvals`; dashboard "Pending approvals" with Approve/Reject (the only approval path); `cortex tailor <job_id>`.
LLM layer (merged PR #5; model ids fixed on branch fix/llm-models):
- `llm/providers.py`: Groq (OpenAI-compatible, JSON mode), Gemini (key in `x-goog-api-key` header, JSON mime type), Ollama (local; always sends `num_ctx`; `think:false`; schema passed as `format` with $refs inlined; refuses prompts that can't fit and flags `prompt_eval_count >= num_ctx` as truncation). Per-provider rate limiters. Errors carry status + one-line redacted reason; keys never in repr/errors/logs. Request shapes verified only against mocks + one real Gemini 400; live calls are run on Surya's Mac.
- `llm/router.py`: per-agent fallback chain from `config.yaml llm.routes`; falls back only on ProviderError (not on programming errors); `personal_data_routes` (tailor) refuse non-local providers at construction (PrivacyViolation); audit logs provider/outcome/seconds, never prompts; `served_by`.
- 429 handling (branch fix/llm-429-backoff): providers wait the server's `retry-after` (else 2/4/8s) and retry up to 3 times; retry-after > 60s = daily quota -> raise at once so the router falls back. Found on the ablation run: removing the double limiter doubled speed to ~20 jobs/min and hit Groq's free-tier tokens-per-minute cap for gpt-oss-120b (reasoning tokens).
- Eval v2 (merged PR #8): report adds top-band share + lift and a paired bootstrap (LLM - rules, same applied jobs, seed 0); `--exclude-fields stack_fit` ablation (separate cache via prompt hash); progress line on stderr; aborts after 10 consecutive live failures (EvalAborted, cache kept); scorer/tailor/bench use `NoLimit` because the router already rate-limits per provider (was double-limiting, ~2x slower).
- CLI: `cortex llm-check` (ping every provider), `cortex eval --provider route|<name>`, `cortex tailor` uses the tailor route, `cortex bench --providers ollama_8b,ollama_granite --jobs 5` (real tailor + Truth Guard loop, no DB writes; pass rate, tries, seconds).
- Cloud models (picked 2026-11-02 from each account's own model list; pinned versions, never *-latest): Groq `openai/gpt-oss-120b` (reasoning_effort low) + `qwen/qwen3.8-27b` as `groq_qwen`; Gemini `gemini-3.8-flash`. The first guesses (llama-3.3-70b-versatile, gemini-2.5-flash) were retired: always check with `cortex llm-check`.
- Local models on Surya's Mac (M5, 16GB): qwen3:8b (tailor default), granite4.2:8b (challenger), qwen3:4b (scorer fallback). Measured: qwen3:8b ~25 tok/s, structured JSON + think:false OK.
NOT built yet: resume PDF + cover note, bullet-selection tailor mode (if bench shows rewrite is too slow), Apify adapter, packet, tracker agent, scheduler, Telegram.

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
