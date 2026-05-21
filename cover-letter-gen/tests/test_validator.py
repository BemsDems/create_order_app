"""Tests for the deterministic validator."""

from __future__ import annotations

from src.validator import validate_deterministic


ALLOWED = ["3", "5", "11000", "4", "6", "20"]


GOOD_LETTER = """\
3+ года разработки B2B-систем на Flutter. В OtherMark спроектировал ERP-систему \
на Clean Architecture с DI — 5 модулей от управления компаниями до производственных \
заданий со сложной бизнес-логикой и валидацией бизнес-форматов. Кодовая база превысила \
11000 строк, модульная архитектура упростила поддержку и тестирование сразу нескольких \
независимых команд разработки. Реализовал внутреннюю дизайн-систему с кастомными \
таблицами и формами для единообразия интерфейса в продукте и быстрой сборки новых \
экранов под новые бизнес-сценарии.

Опыт работы с корпоративными инструментами для сотрудников подкреплён реальной \
практикой в OtherMark. BLoC и Clean Architecture применялись в системе с ролевой \
моделью и сложной бизнес-логикой. Похожие задачи решал в production-проекте \
с 6 типами ролей и 20 переиспользуемыми UI-компонентами.\
"""


def test_good_letter_passes():
    result = validate_deterministic(GOOD_LETTER, ALLOWED)
    assert result.passed, [v.to_dict() for v in result.violations]
    assert 100 <= result.word_count <= 130


def test_invented_number_is_flagged():
    text = GOOD_LETTER.replace("11000 строк", "сотен тысяч пользователей и 999 строк")
    result = validate_deterministic(text, ALLOWED)
    assert not result.passed
    rules = {v.rule for v in result.violations}
    assert "invented_number" in rules


def test_forbidden_phrase_is_flagged():
    text = GOOD_LETTER + "\n\nГотов применить опыт в вашей команде."
    result = validate_deterministic(text, ALLOWED)
    assert not result.passed
    rules = {v.rule for v in result.violations}
    # Both forbidden phrase and either paragraph-count or extra-content errors.
    assert "forbidden_phrase" in rules


def test_library_name_is_flagged():
    text = GOOD_LETTER.replace("Clean Architecture с DI", "Clean Architecture с GetIt")
    result = validate_deterministic(text, ALLOWED)
    assert not result.passed
    assert any(v.rule == "library_name" for v in result.violations)


def test_paragraph_count_is_enforced():
    one_para = GOOD_LETTER.replace("\n\n", " ")
    result = validate_deterministic(one_para, ALLOWED)
    assert not result.passed
    assert any(v.rule == "wrong_paragraph_count" for v in result.violations)


def test_too_short_letter_is_flagged():
    short = "3+ года разработки на Flutter. В OtherMark реализовал 5 модулей.\n\nКодовая база 11000 строк."
    result = validate_deterministic(short, ALLOWED)
    assert not result.passed
    assert any(v.rule == "too_short" for v in result.violations)


def test_first_sentence_must_contain_a_number():
    text = "Опыт разработки на Flutter в течение нескольких лет в B2B-системах. " + GOOD_LETTER.split(". ", 1)[1]
    result = validate_deterministic(text, ALLOWED)
    rules = {v.rule for v in result.violations}
    assert "no_years_in_opener" in rules


def test_number_with_space_normalization():
    # "11 000" should be treated identically to "11000".
    text = GOOD_LETTER.replace("11000", "11 000")
    result = validate_deterministic(text, ALLOWED)
    assert result.passed, [v.to_dict() for v in result.violations]


def test_anglicism_inside_russian_is_flagged():
    text = GOOD_LETTER.replace(
        "сложной бизнес-логикой",
        "complex business logic",
        1,
    )
    result = validate_deterministic(text, ALLOWED)
    assert not result.passed
    rules = {v.rule for v in result.violations}
    assert "anglicism" in rules


def test_allowed_tech_terms_do_not_trigger_anglicism():
    # Tech tokens that ARE in the whitelist should not be flagged as anglicisms.
    # Plain "Flutter / Dart / BLoC" already appear in GOOD_LETTER.
    result = validate_deterministic(GOOD_LETTER, ALLOWED)
    assert not any(v.rule == "anglicism" for v in result.violations)
