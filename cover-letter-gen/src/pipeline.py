"""3-pass orchestrator: Analyze -> Write -> Validate -> (Rewrite up to N).

v2 changes:
- Builds a `CanonicalFacts` from the profile once at init.
- Passes CanonicalFacts (read-only) into Analyzer and Validator.
- Routes low-confidence vacancies to universal-letter mode.
- Emits richer `GenerationResult` (selected_project, confidence,
  used_numbers, used_tech, attempts, semantic_validator_used).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .analyzer import analyze
from .facts import CanonicalFacts, VacancyFit, extract_canonical_facts, vacancy_fit
from .llm_client import LLMClient
from .models import Profile, Vacancy
from .validator import ValidationResult, Violation, validate_deterministic, validate_semantic
from .writer import write_letter


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    vacancy_id: str
    company: str
    title: str
    letter: Optional[str]
    analyzer_json: Optional[Dict[str, Any]]
    selected_project: Optional[str]
    confidence: float
    confidence_reason: str
    used_numbers: List[str]
    used_tech: List[str]
    universal_mode: bool
    semantic_validator_used: bool
    word_count: int
    passed: bool
    attempts: int
    violations: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vacancy_id": self.vacancy_id,
            "company": self.company,
            "title": self.title,
            "letter": self.letter,
            "analyzer_json": self.analyzer_json,
            "selected_project": self.selected_project,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "used_numbers": self.used_numbers,
            "used_tech": self.used_tech,
            "universal_mode": self.universal_mode,
            "semantic_validator_used": self.semantic_validator_used,
            "word_count": self.word_count,
            "passed": self.passed,
            "attempts": self.attempts,
            "violations": self.violations,
            "error": self.error,
        }


@dataclass
class PipelineConfig:
    min_words: int = 100
    max_words: int = 130
    max_writer_retries: int = 2
    use_semantic_validator: bool = True
    writer_temperature: float = 0.4
    writer_max_tokens: int = 400
    # Confidence threshold: vacancies below this trigger universal-letter mode.
    low_confidence_threshold: float = 0.5
    # Hard cutoff: below this we don't generate at all.
    skip_below_confidence: float = 0.2
    # Pre-Analyzer fit gate: skip the vacancy if it doesn't mention any of
    # the candidate's primary skills AND has fewer than this many tech-stack
    # tokens in common with `facts.allowed_tech`.
    min_tech_overlap: int = 1
    # Pre-Analyzer fit gate toggle.
    enforce_fit_gate: bool = True


class CoverLetterPipeline:
    """Stateful pipeline.

    Tracks `used_starts` across `.generate()` calls to encourage variety in
    the first sentence within a single batch.
    """

    def __init__(
        self,
        llm: LLMClient,
        profile: Profile,
        config: Optional[PipelineConfig] = None,
        *,
        forbidden_claims: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.profile = profile
        self.config = config or PipelineConfig()
        self.used_starts: List[str] = []
        self.facts: CanonicalFacts = extract_canonical_facts(
            profile, forbidden_claims=forbidden_claims
        )

    async def generate(self, vacancy: Vacancy) -> GenerationResult:
        # Pre-Analyzer fit gate — skip vacancies with no tech overlap. Saves
        # an LLM call AND avoids the model being tempted to invent matching
        # experience (the previous root cause of unknown_tech_term failures).
        if self.config.enforce_fit_gate:
            fit = vacancy_fit(self.facts, vacancy, self.profile.skills_primary)
            if not fit.primary_match and fit.overlap_count < self.config.min_tech_overlap:
                logger.info(
                    "Vacancy %s: no tech overlap (matched=%s) — skipping pre-Analyzer",
                    vacancy.id, fit.matched_terms,
                )
                return _error_result(
                    vacancy,
                    error="skipped_no_tech_overlap",
                    confidence_reason=(
                        "вакансия не упоминает Flutter и не пересекается с tech-stack резюме"
                    ),
                )

        try:
            analyzer_json = await analyze(self.llm, vacancy, self.facts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analyzer failed for vacancy %s", vacancy.id)
            return _error_result(vacancy, error=f"analyzer: {exc}")

        confidence = float(analyzer_json.get("confidence", 0.0))
        confidence_reason = str(analyzer_json.get("confidence_reason") or "")
        selected_project = str(analyzer_json.get("selected_project") or "")
        selected_numbers: List[str] = list(analyzer_json.get("selected_numbers") or [])
        # The validator's `allowed_numbers` is the full CanonicalFacts
        # whitelist, NOT just the Analyzer's stylistic subset. The opener
        # pool always injects `experience_years` (e.g. "3+ года"), and
        # Analyzer often picks only "impressive" numbers (5, 11000) — that
        # combination produced spurious `invented_number: '3'` violations
        # in v3.
        validator_allowed_numbers: List[str] = list(self.facts.allowed_numbers)
        for n in selected_numbers:
            if n not in validator_allowed_numbers:
                validator_allowed_numbers.append(n)

        # Hard skip on very low confidence — emit a result with no letter.
        if confidence < self.config.skip_below_confidence:
            logger.info(
                "Vacancy %s: confidence %.2f below skip threshold %.2f — skipping",
                vacancy.id, confidence, self.config.skip_below_confidence,
            )
            return GenerationResult(
                vacancy_id=vacancy.id,
                company=vacancy.company,
                title=vacancy.title,
                letter=None,
                analyzer_json=analyzer_json,
                selected_project=selected_project,
                confidence=confidence,
                confidence_reason=confidence_reason,
                used_numbers=[],
                used_tech=[],
                universal_mode=False,
                semantic_validator_used=False,
                word_count=0,
                passed=False,
                attempts=0,
                error="skipped_low_confidence",
            )

        universal_mode = confidence < self.config.low_confidence_threshold
        if universal_mode:
            logger.info(
                "Vacancy %s: confidence %.2f below %.2f — universal mode",
                vacancy.id, confidence, self.config.low_confidence_threshold,
            )

        feedback: Optional[str] = None
        hard_constraints: List[str] = []
        last_letter: str = ""
        last_result: Optional[ValidationResult] = None
        semantic_used = False
        # Set of (rule, evidence) tuples — used to detect repeat violations.
        prev_violation_keys: set[tuple[str, str]] = set()

        for attempt in range(1, self.config.max_writer_retries + 2):
            try:
                last_letter = await write_letter(
                    self.llm,
                    analyzer_json=analyzer_json,
                    facts=self.facts,
                    used_starts=self.used_starts,
                    feedback=feedback,
                    hard_constraints=hard_constraints or None,
                    universal_mode=universal_mode,
                    temperature=self.config.writer_temperature,
                    max_tokens=self.config.writer_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Writer failed (attempt %d) for vacancy %s", attempt, vacancy.id)
                return _error_result(
                    vacancy,
                    error=f"writer: {exc}",
                    analyzer_json=analyzer_json,
                    confidence=confidence,
                    confidence_reason=confidence_reason,
                    selected_project=selected_project,
                    universal_mode=universal_mode,
                    attempts=attempt,
                )

            det = validate_deterministic(
                last_letter,
                facts=self.facts,
                allowed_numbers=validator_allowed_numbers,
                min_words=self.config.min_words,
                max_words=self.config.max_words,
                universal_mode=universal_mode,
            )
            if not det.passed:
                feedback = det.format_feedback()
                # On repeated violations, escalate to hard_constraints so the
                # next prompt has a much louder prohibition.
                hard_constraints = _escalate_constraints(
                    det.violations, prev_violation_keys, hard_constraints,
                )
                last_result = det
                logger.info(
                    "Vacancy %s: attempt %d failed deterministic validation (%d violations)",
                    vacancy.id, attempt, len(det.violations),
                )
                continue

            if self.config.use_semantic_validator:
                semantic_used = True
                try:
                    sem = await validate_semantic(
                        self.llm,
                        last_letter,
                        analyzer_json,
                        validator_allowed_numbers,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Semantic validator errored, treating as passed: %s", exc)
                    sem = ValidationResult(passed=True, violations=[], word_count=det.word_count)

                if not sem.passed:
                    feedback = sem.format_feedback()
                    # Preserve det.used_numbers/used_tech (semantic doesn't compute them).
                    sem.used_numbers = det.used_numbers
                    sem.used_tech = det.used_tech
                    last_result = sem
                    logger.info(
                        "Vacancy %s: attempt %d failed semantic validation (%d violations)",
                        vacancy.id, attempt, len(sem.violations),
                    )
                    continue
                # Semantic passed — merge det's used_* into the result.
                sem.used_numbers = det.used_numbers
                sem.used_tech = det.used_tech
                last_result = sem
            else:
                last_result = det

            # Success: record opener for variety.
            opener = last_letter.strip().split(".", 1)[0]
            if opener:
                self.used_starts.append(opener[:80])

            return GenerationResult(
                vacancy_id=vacancy.id,
                company=vacancy.company,
                title=vacancy.title,
                letter=last_letter,
                analyzer_json=analyzer_json,
                selected_project=selected_project,
                confidence=confidence,
                confidence_reason=confidence_reason,
                used_numbers=list(last_result.used_numbers),
                used_tech=list(last_result.used_tech),
                universal_mode=universal_mode,
                semantic_validator_used=semantic_used,
                word_count=det.word_count,
                passed=True,
                attempts=attempt,
            )

        # Out of retries.
        violations = [v.to_dict() for v in (last_result.violations if last_result else [])]
        return GenerationResult(
            vacancy_id=vacancy.id,
            company=vacancy.company,
            title=vacancy.title,
            letter=last_letter or None,
            analyzer_json=analyzer_json,
            selected_project=selected_project,
            confidence=confidence,
            confidence_reason=confidence_reason,
            used_numbers=list(last_result.used_numbers) if last_result else [],
            used_tech=list(last_result.used_tech) if last_result else [],
            universal_mode=universal_mode,
            semantic_validator_used=semantic_used,
            word_count=(last_result.word_count if last_result else 0),
            passed=False,
            attempts=self.config.max_writer_retries + 1,
            violations=violations,
            error="validation_failed_after_retries",
        )

    async def generate_batch(
        self,
        vacancies: List[Vacancy],
        *,
        max_concurrent: int = 5,
    ) -> List[GenerationResult]:
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(v: Vacancy) -> GenerationResult:
            async with sem:
                return await self.generate(v)

        return await asyncio.gather(*(_one(v) for v in vacancies))


def _escalate_constraints(
    current_violations: List[Violation],
    prev_keys: set[tuple[str, str]],
    existing_hard: List[str],
) -> List[str]:
    """Convert repeated violations into hard-constraint lines for the next retry.

    A violation that repeats with the same (rule, evidence) tuple is treated
    as evidence that polite feedback isn't working — the next prompt gets a
    loud, explicit prohibition. Mutates `prev_keys` to remember what we've
    seen so far.
    """
    new_hard: List[str] = list(existing_hard)
    seen_hard = {line.lower() for line in new_hard}
    for v in current_violations:
        key = (v.rule, v.evidence)
        if key in prev_keys:
            # Repeated violation — escalate.
            line = _hard_line_for(v)
            if line and line.lower() not in seen_hard:
                new_hard.append(line)
                seen_hard.add(line.lower())
        prev_keys.add(key)
    return new_hard


def _hard_line_for(v: Violation) -> str:
    """Map a violation to a short, loud prohibition line for the prompt."""
    if v.rule == "meta_leak":
        return f"НЕ пиши слово «{v.evidence}» — это служебная разметка, а не часть письма."
    if v.rule == "anglicism":
        return f"НЕ пиши слово «{v.evidence}» латиницей — оно не из списка разрешённых технологий."
    if v.rule == "invented_number":
        return f"НЕ используй число «{v.evidence}» — его нет в «Разрешённых числах»."
    if v.rule == "forbidden_claim":
        return f"НЕ упоминай «{v.evidence}» — этого нет в резюме."
    if v.rule == "forbidden_phrase":
        return f"НЕ используй фразу «{v.evidence}»."
    if v.rule == "library_name":
        return f"НЕ называй библиотеку «{v.evidence}» — пиши обобщённо (DI, HTTP-клиент, state management)."
    if v.rule == "unknown_tech_term":
        return f"НЕ упоминай «{v.evidence}» — этой технологии нет в моём опыте."
    if v.rule == "too_long":
        return "Письмо должно быть КОРОТКИМ — не больше одного-двух абзацев."
    if v.rule == "wrong_paragraph_count":
        return "Структура должна быть РОВНО такой, как просили в системном сообщении."
    return ""


def _error_result(
    vacancy: Vacancy,
    *,
    error: str,
    analyzer_json: Optional[Dict[str, Any]] = None,
    confidence: float = 0.0,
    confidence_reason: str = "",
    selected_project: Optional[str] = None,
    universal_mode: bool = False,
    attempts: int = 0,
) -> GenerationResult:
    return GenerationResult(
        vacancy_id=vacancy.id,
        company=vacancy.company,
        title=vacancy.title,
        letter=None,
        analyzer_json=analyzer_json,
        selected_project=selected_project,
        confidence=confidence,
        confidence_reason=confidence_reason,
        used_numbers=[],
        used_tech=[],
        universal_mode=universal_mode,
        semantic_validator_used=False,
        word_count=0,
        passed=False,
        attempts=attempts,
        error=error,
    )
