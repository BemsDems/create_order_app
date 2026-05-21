"""3-pass orchestrator: Analyze -> Write -> Validate -> (Rewrite up to N)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .analyzer import analyze, profile_to_compact_dict
from .llm_client import LLMClient
from .models import Profile, Vacancy
from .validator import ValidationResult, validate_deterministic, validate_semantic
from .writer import write_letter


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    vacancy_id: str
    company: str
    title: str
    letter: Optional[str]
    analyzer_json: Optional[Dict[str, Any]]
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


class CoverLetterPipeline:
    """Stateful pipeline.

    Tracks `used_starts` across `.generate()` calls to encourage variety in
    the first sentence within a single batch.
    """

    def __init__(self, llm: LLMClient, profile: Profile, config: Optional[PipelineConfig] = None):
        self.llm = llm
        self.profile = profile
        self.config = config or PipelineConfig()
        self.used_starts: List[str] = []
        self._profile_dict = profile_to_compact_dict(profile)

    async def generate(self, vacancy: Vacancy) -> GenerationResult:
        try:
            analyzer_json = await analyze(self.llm, vacancy, self.profile)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analyzer failed for vacancy %s", vacancy.id)
            return GenerationResult(
                vacancy_id=vacancy.id,
                company=vacancy.company,
                title=vacancy.title,
                letter=None,
                analyzer_json=None,
                word_count=0,
                passed=False,
                attempts=0,
                error=f"analyzer: {exc}",
            )

        allowed_numbers: List[str] = list(analyzer_json.get("allowed_numbers") or [])

        feedback: Optional[str] = None
        last_letter: str = ""
        last_result: Optional[ValidationResult] = None
        for attempt in range(1, self.config.max_writer_retries + 2):
            try:
                last_letter = await write_letter(
                    self.llm,
                    analyzer_json=analyzer_json,
                    profile_dict=self._profile_dict,
                    used_starts=self.used_starts,
                    feedback=feedback,
                    temperature=self.config.writer_temperature,
                    max_tokens=self.config.writer_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Writer failed (attempt %d) for vacancy %s", attempt, vacancy.id)
                return GenerationResult(
                    vacancy_id=vacancy.id,
                    company=vacancy.company,
                    title=vacancy.title,
                    letter=None,
                    analyzer_json=analyzer_json,
                    word_count=0,
                    passed=False,
                    attempts=attempt,
                    error=f"writer: {exc}",
                )

            det = validate_deterministic(
                last_letter,
                allowed_numbers=allowed_numbers,
                min_words=self.config.min_words,
                max_words=self.config.max_words,
            )
            if not det.passed:
                feedback = det.format_feedback()
                last_result = det
                logger.info(
                    "Vacancy %s: attempt %d failed deterministic validation (%d violations)",
                    vacancy.id, attempt, len(det.violations),
                )
                continue

            if self.config.use_semantic_validator:
                try:
                    sem = await validate_semantic(
                        self.llm, last_letter, analyzer_json, allowed_numbers
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Semantic validator errored, treating as passed: %s", exc)
                    sem = ValidationResult(passed=True, violations=[], word_count=det.word_count)

                if not sem.passed:
                    feedback = sem.format_feedback()
                    last_result = sem
                    logger.info(
                        "Vacancy %s: attempt %d failed semantic validation (%d violations)",
                        vacancy.id, attempt, len(sem.violations),
                    )
                    continue
                last_result = sem
            else:
                last_result = det

            # Success: record the opener for variety in subsequent letters.
            opener = last_letter.strip().split(".", 1)[0]
            if opener:
                self.used_starts.append(opener[:80])

            return GenerationResult(
                vacancy_id=vacancy.id,
                company=vacancy.company,
                title=vacancy.title,
                letter=last_letter,
                analyzer_json=analyzer_json,
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
