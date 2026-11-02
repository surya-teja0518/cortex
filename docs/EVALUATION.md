# CORTEX scorer evaluation: evidence, method and limits

> **Claim:** An LLM job-relevance scorer (Groq, `openai/gpt-oss-120b`) lifted the ranking of jobs I actually
> applied to from the **51st to the 61st percentile**, against a rule-based baseline at chance, on **362 real postings**.

This document is the evidence behind that sentence: what was measured, how, how sure we are, and what
could make it wrong. Every number below comes from one `cortex eval` run on 2026-11-02, and the open
items are listed as open.

---

## 1. Result

`cortex eval --provider groq` output (2026-11-02):

```
population=362 (ungated, deduped)  positives=60 (status in applied/interview/offer)  negatives=15 (exp_mismatch=15, skips=0)
scorer                recall@P0-P1  mean pct rank      prec@P0-P1  pos>neg AUC  llm_unscored
deterministic                   0%           50.8       n/a (n=0)         0.34             -
llm:groq                       30%           60.9      78% (n=23)         0.58            12
llm cache hits=238  live-scored=112  (llm_unscored jobs use the deterministic score)
note: 1 positive(s) were blocked by deterministic gates and are excluded above; for the full pipeline they are misses.
```

| Metric | Rules | LLM | Read it as |
|---|---|---|---|
| **Mean percentile rank of applied jobs** | 50.8 | **60.9** | **Headline metric.** Rules ≈ chance; the LLM ranks the jobs I chose clearly above chance. |
| Pos-vs-neg AUC | 0.34 | 0.58 | Direction is right, but only 15 negatives: not statistically reliable yet. |
| Recall@P0-P1 | 0% | 30% | **Real lift: ~1.75×.** The LLM bands only 17% of jobs P0/P1, yet they hold 30% of applied jobs (section 4.3). |
| Precision@P0-P1 | n/a | 78% (n=23) | **Not evidence.** It equals the base rate (section 4.2). |

## 2. Setup

| Item | Value |
|---|---|
| Data | Personal job tracker (SYNAPSE master sheet + Batch 011), imported with `cortex import` |
| Population | 362 jobs after de-duplication and deterministic gates (title blocklist, min-YOE cap 5) |
| Positives | 60 jobs with status Applied / Interview / Offer (63 applied rows before dedupe/gates) |
| Negatives | 15 jobs with status `exp_mismatch` (I looked and turned them down). `not_accepting`/`closed` are the employer's decision and left unlabelled. |
| Baseline | Deterministic SYNAPSE rules (`core/scoring.py`): base per track plus stack/location/experience adjustments |
| LLM | Groq `openai/gpt-oss-120b`, `reasoning_effort: low`, temperature 0, JSON mode |
| LLM input | Tracker fields only: title, company, location, experience text, `stack_fit`, work model, notes. No job description in this run. All fields sanitized and wrapped in `<untrusted_job_text>`. |
| LLM output | `{"score": 0-100, "reasons": [...], "red_flags": [...]}`, validated by Pydantic; invalid output gets 1 retry, then the job keeps its rule score (`llm_unscored`, 12 of 362 here) |
| Cost | $0 (Groq free tier). ~8 jobs/min with rate limiting, so a full run is ~45 min; reruns are free from the cache. |

## 3. Metrics and why these ones

- **Mean percentile rank (headline).** For each applied job, the percentage of all 362 jobs it outranks
  (ties count half), averaged. 50 = random ordering, 100 = every applied job at the top. It uses the
  whole ranking, needs **no negatives**, and doesn't depend on band thresholds. That matters because the
  rules' bands are miscalibrated: with tracker data they top out at 78, so P0 (≥ 85) is unreachable.
- **Pos-vs-neg AUC.** The probability that a random applied job outscores a random turned-down job (Mann-Whitney, ties = 0.5).
- **Recall / precision @P0-P1.** Operational metrics ("does the top of my queue contain what I'd apply to?"), but band-dependent and fragile at this sample size.

## 4. How sure are we?

### 4.1 Statistical uncertainty (approximate)
These are normal approximations, stated as such. Mean percentile rank: a random applied job's rank is
roughly uniform (SD ≈ 28.9), so with 60 positives the SE ≈ 3.7. AUC: Hanley–McNeil SE.

| | Estimate | ~95% CI | vs chance |
|---|---|---|---|
| Rules, mean pct rank | 50.8 | 43.5 – 58.1 | indistinguishable from random (z ≈ 0.2) |
| LLM, mean pct rank | 60.9 | 53.6 – 68.2 | **above random (z ≈ 2.9)** |
| LLM − rules gap | +10.1 | — | **borderline (z ≈ 1.9, unpaired approximation)**; a paired test on the same jobs is planned |
| Rules, AUC | 0.34 | 0.18 – 0.50 | at or below chance |
| LLM, AUC | 0.58 | 0.42 – 0.74 | includes 0.5: **not significant** with 15 negatives |

**What is safe to claim:** the LLM ranks applied jobs above chance, and the rules don't. **What is not
yet safe to claim:** that the gap between them is statistically significant, or any precision figure.

### 4.2 Why precision@P0-P1 = 78% is not evidence
Precision is computed over **labelled** jobs only, and 60 of the 75 labelled jobs (**80%**) are
positives. A random scorer would also get about 80%. 78% is the base rate, not skill. It becomes
meaningful only with many more negatives (Skip decisions in the review UI).

### 4.3 Recall@P0-P1 needs its denominator (checked 2026-11-02)
30% recall is lift only if the LLM bands well under 30% of all jobs as P0/P1. Band counts for the 350
LLM-scored jobs: **P0 21, P1 39, P2 58, P3 232**, so P0/P1 = 60 jobs = **17%** of the queue.

- 18 of 60 applied jobs (30%) sit in that top 17% → **lift ≈ 1.75×** over random banding.
- Equivalently, 30% of P0/P1 jobs were applied to, vs **16.6%** of all jobs.
- Random banding would put ~10.3 applied jobs in a 60-job top band; 18 is **z ≈ 2.6** above that (binomial approximation).
- Conservative: unlabelled `to_apply` jobs count as "not applied" here, though some may be applied to later.

Command used:
```bash
sqlite3 data/cortex.db "select json_extract(result_json,'\$.band') band, count(*) from llm_score_cache where provider='groq' group by band;"
```

## 5. Threats to validity (the questions an interviewer should ask)

| Threat | Status / mitigation |
|---|---|
| **Label leakage via `notes`.** If notes mention the outcome ("applied", "referral", "rejected"), the LLM could read the answer. | **Checked, no sign of leakage** (keyword check, below): outcome-like words appear in 5/63 applied (7.9%) vs 22/237 to_apply (9.3%) notes, i.e. not more often for applied jobs. Keyword-based, so not proof; a no-notes ablation remains the definitive test. |
| **`stack_fit` is SYNAPSE's own fit verdict** ("Azure-bullseye", "AWS-stretch"), produced by the system I used when deciding where to apply. | Known. The rules use it too, so the comparison is like for like, but some of the LLM's signal may be SYNAPSE's judgment re-read. Ablation planned: rerun without `stack_fit`. |
| **Circularity with SYNAPSE's Match score.** I chose applications partly by looking at SYNAPSE's Match/Priority. | That score is **never** used as a baseline or shown to the LLM. Comparing against it would be circular. |
| **Labels are preferences, not job quality.** "Applied" means "I chose it", not "it was a good job". | That's intended: the scorer's job is to predict my choices. It says nothing about interview or offer rates. |
| **Selection bias.** I only saw what SYNAPSE scraped and surfaced. | The population is my real queue, which is exactly where the scorer runs. Not a claim about the job market. |
| **Small, imbalanced sample.** 60 positives, 15 negatives. | Confidence intervals reported above. Negatives grow via Skip decisions. |
| **Fallbacks.** 12 jobs fell back to the rule score. | Kept in the LLM column (conservative: they drag it towards the baseline). |
| **Model drift.** Provider models change or get retired (two were retired during this project). | Exact model id pinned (no `-latest` aliases); results cached by provider + prompt hash. |
| **One run, one model.** | Second model (`qwen/qwen3.8-27b`) comparison planned; see section 7. |

Leakage check (run locally; prints counts only). Result 2026-11-02: applied 5/63, to_apply 22/237, closed 0/7, exp_mismatch 0/15, not_accepting 0/48.
```bash
sqlite3 data/cortex.db "select status, count(*) total, sum(lower(notes) like '%appl%' or lower(notes) like '%refer%' or lower(notes) like '%reject%' or lower(notes) like '%interview%') suspicious from jobs group by status;"
```

## 6. Likely interview questions, answered

**Why not just train a classifier on your labels?** 60 positives is too few to train and validate
reliably. The LLM is zero-shot, so it needs no training data. A logistic-regression baseline on the same
fields with cross-validation is a fair next comparison and hasn't been done yet.

**Why mean percentile rank and not accuracy?** There's no threshold-free accuracy without negatives, and
the rule bands are miscalibrated. Ranking is what the review queue actually uses.

**How do you know the LLM isn't just reading the answer?** I checked the free-text `notes` field: outcome
words like "applied" or "referral" are no more common for applied jobs (7.9%) than for unapplied ones (9.3%).
The remaining risk is `stack_fit`, which is SYNAPSE's own fit verdict. A rerun without it is the planned ablation.

**Is it reproducible?** Temperature 0, pinned model id, and every result cached by job, provider and
SHA-256 of the exact prompt. Rerunning the same config reproduces the report from the cache. A fresh run
can still differ slightly, because providers don't guarantee identical output at temperature 0.

**What does it cost?** $0: Groq free tier for scoring, and local Ollama models for anything containing
resume data. About 45 minutes for a full 362-job run because of free-tier rate limits; reruns are free.

**Is it safe to feed scraped job text to an LLM?** Every field is sanitized, wrapped in
`<untrusted_job_text>` delimiters, and checked for prompt-injection patterns. Flagged jobs never reach
the LLM. Output must validate against a strict Pydantic schema (no extra fields), or it's discarded.

**Why `gpt-oss-120b`?** It was the strongest general model available on the free Groq account. The
choice is configurable; the eval compares providers on identical inputs.

**Is 30% recall good?** On its own, no. What matters is that the LLM puts only 17% of jobs in P0/P1, and
those hold 30% of my applications: ~1.75× lift, about 18 hits where random banding gives ~10.

**What would make you drop the claim?** If removing `stack_fit` drops the LLM back to ~50, or if a paired
test shows no real gap vs the rules.

## 7. Open items (in priority order)

1. ~~`notes` leakage check~~ (done: no sign of leakage). A no-notes ablation is optional confirmation.
2. Ablation without `stack_fit` (SYNAPSE's own verdict): the most important remaining check.
   Tooling done: `cortex eval --provider groq --exclude-fields stack_fit` (separate cache, labelled in the report). Run pending.
3. Paired significance test for the LLM − rules gap. Tooling done: `cortex eval` now prints a paired
   bootstrap (same applied jobs, 2000 resamples, seed 0) with a 95% CI and one-sided P(diff ≤ 0). Number pending
   (it reads from the cache, so it costs nothing to rerun). ~~Band share for recall~~ (done: 17%, lift ≈ 1.75×; now a report column).
4. Second model (`groq_qwen`) on identical inputs.
5. More negatives via Skip decisions, so precision and AUC become meaningful.

## 8. Reproduce

```bash
cortex import "<master>.csv" "<batch>.csv"     # tracker CSV exports, master first
cortex llm-check                                 # every provider answers
cortex eval                                      # rules only
cortex eval --provider groq                      # rules vs LLM (cached per job + provider + prompt hash) + paired test
cortex eval --provider groq --exclude-fields stack_fit   # ablation: same model, stack_fit removed from the prompt
```
