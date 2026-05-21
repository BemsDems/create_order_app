"""End-to-end pipeline test with a mocked LLM client.

Verifies that:
- Analyzer JSON is consumed correctly.
- Writer signature is stripped if the model adds it.
- Validator failures trigger a retry with feedback.
- Successful generation records the opener in `used_starts`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from src.llm_client import LLMConfig
from src.models import Language, Position, Profile, Project, Vacancy
from src.pipeline import CoverLetterPipeline, PipelineConfig
from src.prompts.analyzer import ANALYZER_SYSTEM
from src.prompts.validator import VALIDATOR_SYSTEM
from src.prompts.writer import WRITER_SYSTEM


def _make_profile() -> Profile:
    return Profile(
        name="Тест Тестов",
        experience_years=3,
        experience_months=2,
        summary="Flutter-разработчик с 3+ годами опыта.",
        skills_primary=["Flutter", "Dart", "BLoC", "Clean Architecture"],
        positions=[
            Position(
                title="Flutter-разработчик",
                company="OtherCode",
                duration_months=9,
                projects=[
                    Project(
                        name="OtherMark",
                        description="B2B/ERP-система для маркировки товаров.",
                        tech_stack=["Flutter", "BLoC", "Clean Architecture", "gRPC"],
                        achievements=[
                            "Спроектировал архитектуру на Clean Architecture с DI.",
                            "Реализовал 5 ключевых B2B-модулей со сложной бизнес-логикой.",
                            "Кодовая база превысила 11 000 строк.",
                        ],
                    ),
                ],
            ),
        ],
        languages=[Language(name="Русский", level_code="C2")],
    )


def _make_vacancy() -> Vacancy:
    return Vacancy(
        id="vac-1",
        title="Flutter-разработчик",
        company="DNS",
        description="Корпоративное приложение для сотрудников магазинов. Clean Architecture, BLoC, сложная бизнес-логика, ролевая модель.",
        requirements=["Flutter", "BLoC", "Clean Architecture"],
        work_format="remote",
    )


_ANALYZER_RESPONSE = {
    "vacancy_type": "b2b_erp",
    "top_requirements": ["Flutter", "BLoC", "Clean Architecture"],
    "best_project": {
        "name": "OtherMark",
        "why_relevant": "ERP-система с ролями и сложной бизнес-логикой как у DNS.",
    },
    "evidence": [
        {"requirement": "Clean Architecture", "candidate_proof": "Спроектировал на Clean Architecture с DI"},
        {"requirement": "BLoC", "candidate_proof": "5 модулей на BLoC + Cubit"},
        {"requirement": "сложная бизнес-логика", "candidate_proof": "5 ERP-модулей со сложной логикой"},
    ],
    "hook_phrase": "корпоративное приложение для сотрудников",
    "honest_gaps": [],
    "allowed_numbers": ["3", "5", "11000"],
}


_GOOD_LETTER = (
    "3+ года разработки B2B-систем на Flutter. В OtherMark спроектировал "
    "ERP-систему на Clean Architecture с DI — 5 модулей от управления "
    "компаниями до производственных заданий со сложной бизнес-логикой и "
    "валидацией бизнес-форматов. Кодовая база превысила 11000 строк, "
    "модульная архитектура упростила поддержку и тестирование сразу "
    "нескольких независимых команд разработки. Реализовал внутреннюю "
    "дизайн-систему с кастомными таблицами и формами для единообразия "
    "интерфейса в продукте и быстрой сборки новых экранов под новые сценарии.\n\n"
    "Опыт работы с корпоративными инструментами для сотрудников подкреплён "
    "реальной практикой в OtherMark. BLoC и Clean Architecture применялись "
    "в системе с ролевой моделью и сложной бизнес-логикой. Похожие задачи "
    "решал в production-проекте с 5 модулями и 3 годами развития."
)


_BAD_LETTER = (
    "Опыт разработки B2B-систем на Flutter. В OtherMark спроектировал ERP "
    "с complex business logic — 5 модулей. Готов применить этот опыт.\n\n"
    "Ваш продукт требует похожего подхода."
)


class FakeLLMClient:
    """Deterministic mock that routes by system prompt."""

    def __init__(self, responses: Dict[str, List[str]]):
        # `responses` maps system-prompt -> queue of replies.
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: List[Dict[str, Any]] = []
        # Required by LLMClient interface — pipeline only uses .generate / .close.
        self.config = LLMConfig(api_key="test", endpoint="test", model="test")
        self.stats = type("S", (), {"attempts": 0, "last_error": None, "json_mode_supported": True})()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> str:
        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        })
        queue = self.responses.get(system_prompt)
        if not queue:
            raise AssertionError(f"No mock response for system_prompt={system_prompt[:60]!r}")
        return queue.pop(0)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_happy_path():
    llm = FakeLLMClient({
        ANALYZER_SYSTEM: [json.dumps(_ANALYZER_RESPONSE, ensure_ascii=False)],
        WRITER_SYSTEM: [_GOOD_LETTER],
        VALIDATOR_SYSTEM: [json.dumps({"passed": True, "violations": []})],
    })
    profile = _make_profile()
    pipeline = CoverLetterPipeline(llm, profile, config=PipelineConfig())
    result = await pipeline.generate(_make_vacancy())

    assert result.passed, result.violations
    assert result.letter is not None
    assert result.attempts == 1
    assert 100 <= result.word_count <= 130
    # Opener got recorded for next-letter variety.
    assert pipeline.used_starts and "3+ года" in pipeline.used_starts[0]


@pytest.mark.asyncio
async def test_retry_on_deterministic_failure():
    # First writer attempt is bad (anglicism + forbidden phrase),
    # second attempt returns the good letter.
    llm = FakeLLMClient({
        ANALYZER_SYSTEM: [json.dumps(_ANALYZER_RESPONSE, ensure_ascii=False)],
        WRITER_SYSTEM: [_BAD_LETTER, _GOOD_LETTER],
        VALIDATOR_SYSTEM: [json.dumps({"passed": True, "violations": []})],
    })
    profile = _make_profile()
    pipeline = CoverLetterPipeline(llm, profile, config=PipelineConfig())
    result = await pipeline.generate(_make_vacancy())

    assert result.passed
    assert result.attempts == 2
    # The second writer call must have received feedback referring to violations.
    writer_calls = [c for c in llm.calls if c["system"] == WRITER_SYSTEM]
    assert len(writer_calls) == 2
    assert "Нарушения" in writer_calls[1]["user"]


@pytest.mark.asyncio
async def test_gives_up_after_max_retries():
    profile = _make_profile()
    cfg = PipelineConfig(max_writer_retries=1)  # => 2 attempts total
    llm = FakeLLMClient({
        ANALYZER_SYSTEM: [json.dumps(_ANALYZER_RESPONSE, ensure_ascii=False)],
        WRITER_SYSTEM: [_BAD_LETTER, _BAD_LETTER],
        VALIDATOR_SYSTEM: [],
    })
    pipeline = CoverLetterPipeline(llm, profile, config=cfg)
    result = await pipeline.generate(_make_vacancy())

    assert not result.passed
    assert result.error == "validation_failed_after_retries"
    assert result.attempts == 2
    assert result.violations  # surfaced for debugging


@pytest.mark.asyncio
async def test_writer_signature_is_stripped():
    profile = _make_profile()
    letter_with_sig = _GOOD_LETTER + "\n\nС уважением,\nТест Тестов"
    llm = FakeLLMClient({
        ANALYZER_SYSTEM: [json.dumps(_ANALYZER_RESPONSE, ensure_ascii=False)],
        WRITER_SYSTEM: [letter_with_sig],
        VALIDATOR_SYSTEM: [json.dumps({"passed": True, "violations": []})],
    })
    pipeline = CoverLetterPipeline(llm, profile, config=PipelineConfig())
    result = await pipeline.generate(_make_vacancy())
    assert result.passed
    assert "С уважением" not in (result.letter or "")


@pytest.mark.asyncio
async def test_analyzer_normalizes_allowed_numbers():
    """Analyzer should always include the candidate's years of experience in allowed_numbers."""
    response = dict(_ANALYZER_RESPONSE)
    response["allowed_numbers"] = []  # model "forgot" to fill them in
    llm = FakeLLMClient({
        ANALYZER_SYSTEM: [json.dumps(response, ensure_ascii=False)],
        WRITER_SYSTEM: [_GOOD_LETTER],
        VALIDATOR_SYSTEM: [json.dumps({"passed": True, "violations": []})],
    })
    profile = _make_profile()
    pipeline = CoverLetterPipeline(llm, profile, config=PipelineConfig())
    result = await pipeline.generate(_make_vacancy())
    # Even though analyzer returned empty list, normalization should have added "3".
    assert result.passed
    assert "3" in result.analyzer_json["allowed_numbers"]
    # And it should have absorbed "11000" + "5" from the achievements text.
    assert "5" in result.analyzer_json["allowed_numbers"]
    assert "11000" in result.analyzer_json["allowed_numbers"]
