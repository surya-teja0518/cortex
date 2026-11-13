"""Track tailor: pick the right one of Surya's curated resumes for a job, then put the most relevant
bullets first. No LLM, no rewriting: every word comes from his own resume.

Why this exists: the LLM tailor (agents/tailor.py) kept 1-2 bullets per role and flattened the skills
section into one list. Surya already has three hand-tuned resumes (Data Engineer, Oracle DBA, AI/ML),
so the better tailoring is selection plus ordering, not generation. The LLM tailor stays as an option.

- Track choice: the job's track (data / database / ai, from the tracker) via config `tailor.track_map`;
  a CLI/dashboard override wins.
- Reordering: each bullet is scored by how many of the job's tech terms (master skills + known tech
  vocabulary, found in title / stack_fit / notes / description) it mentions, plus shared keywords from the
  title. Stable sort: ties keep the resume's own order; nothing is dropped, added or edited.
- Job text is untrusted, but here it is only *compared* against bullets, never copied into the resume.
- Output still goes through guarded_tailor -> Truth Guard, which is the safety net if the track content
  in master_resume.yaml ever drifts from the master facts.
"""
import re
from collections.abc import Callable

from cortex.agents.scorer import ScoringInput
from cortex.agents.truth_guard import TruthGuard
from cortex.core.resume import MasterResume, TailoredResume, norm

DEFAULT_TRACK_MAP = {"data": "data_engineer", "database": "oracle_dba", "ai": "ai_ml"}
_WORD = re.compile(r"[a-z][a-z0-9+#.-]{3,}")
_STOP = {"with", "from", "that", "this", "your", "will", "have", "years", "year", "experience", "role",
         "team", "work", "engineer", "engineering", "senior", "junior", "india", "hyderabad", "remote",
         "hybrid", "onsite", "full", "time", "data"}   # 'data' is in every bullet: no signal


class TrackError(ValueError):
    pass


class TrackTailor:
    def __init__(self, master: MasterResume, track_map: dict[str, str] | None = None,
                 default_track: str = "data_engineer", override: str | None = None, original: bool = False):
        """original=True: Surya's own PDF goes into the packet as-is. The track is picked by classifying the job
        (core/skillfit.best_track), bullets keep his order, and based_on is 'original:<track>' so the packet
        copies resumes/<track>.pdf instead of rendering one."""
        self.original = original
        if not master.tracks:
            raise TrackError("master_resume.yaml has no `tracks:` section (your three curated resumes)")
        self.master = master
        self.track_map = {**DEFAULT_TRACK_MAP, **(track_map or {})}
        self.default_track = default_track
        if override is not None and override not in master.tracks:
            raise TrackError(f"unknown track {override!r}; have {sorted(master.tracks)}")
        self.override = override
        self._guard = TruthGuard(master)   # reused only for its tech-term detection
        self._matcher = None
        if original:
            from cortex.core.skillfit import SkillMatcher
            self._matcher = SkillMatcher(master)

    def track_for(self, job: ScoringInput, job_track: str | None) -> str:
        if self.override:
            return self.override
        name = self.track_map.get(job_track or "", self.default_track)
        name = name if name in self.master.tracks else self.default_track if self.default_track in self.master.tracks \
            else next(iter(self.master.tracks))
        if self._matcher is not None:
            name = self._matcher.best_track(job.title, job.description or "", name)
        return name

    def _job_terms(self, job: ScoringInput) -> tuple[set[str], set[str]]:
        text = " ".join(x for x in (job.title, job.stack_fit, job.notes, job.description) if x)
        terms = {self._guard._skill_canon.get(norm(t), norm(t)) for t in self._guard._terms_in(text)}
        words = {w for w in _WORD.findall((job.title or "").lower()) if w not in _STOP}
        return terms, words

    def _score(self, bullet: str, terms: set[str], words: set[str]) -> int:
        b_terms = {self._guard._skill_canon.get(norm(t), norm(t)) for t in self._guard._terms_in(bullet)}
        b_words = set(_WORD.findall(bullet.lower()))
        return 3 * len(b_terms & terms) + len(b_words & words)

    def build(self, job: ScoringInput, job_track: str | None = None) -> TailoredResume:
        name = self.track_for(job, job_track)
        tr = self.master.tracks[name]
        terms, words = self._job_terms(job)

        def ordered(bullets):   # stable: equal scores keep the curated order; original mode keeps his order
            return list(bullets) if self.original else sorted(bullets, key=lambda b: -self._score(b, terms, words))

        skills = list(dict.fromkeys(s for g in tr.skill_groups for s in g.skills))
        return TailoredResume.model_validate({
            "summary": tr.summary, "skills": skills,
            "skill_groups": [g.model_dump() for g in tr.skill_groups],
            "roles": [{**r.model_dump(), "bullets": ordered(r.bullets)} for r in tr.roles],
            "projects": [{"name": p.name, "bullets": ordered(p.bullets)} for p in tr.projects],
            "certifications": list(tr.certifications),
            "based_on": f"{'original' if self.original else 'track'}:{name}",
        })

    def for_job(self, wrapped_job: str, *, job: ScoringInput | None = None, job_track: str | None = None,
                **job_ref) -> Callable[[list[str]], TailoredResume]:
        """Same interface as Tailor.for_job, so the graph and guarded_tailor don't care which tailor runs.
        Feedback is ignored: the output is deterministic, so a Truth Guard failure means the track content
        itself is wrong and the run is (correctly) rejected after the retries."""
        if job is None:
            raise TrackError("track tailor needs the structured job (ScoringInput)")
        return lambda feedback: self.build(job, job_track)
