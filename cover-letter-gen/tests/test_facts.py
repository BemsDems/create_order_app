"""Tests for the canonical-facts extractor."""

from __future__ import annotations

from src.facts import extract_canonical_facts, vacancy_fit
from src.models import Position, Profile, Project, Vacancy


def _profile() -> Profile:
    return Profile(
        name="Иван Иванов",
        experience_years=3,
        experience_months=2,
        summary="Flutter-разработчик с опытом B2B-систем.",
        skills_primary=["Flutter", "Dart", "BLoC", "Clean Architecture"],
        skills_secondary=["Firebase", "JWT"],
        positions=[
            Position(
                title="Flutter-разработчик",
                company="OtherCode",
                industry="ERP",
                projects=[
                    Project(
                        name="OtherMark",
                        description="B2B/ERP-система для маркировки.",
                        tech_stack=["Flutter", "BLoC", "Clean Architecture", "gRPC"],
                        achievements=[
                            "Спроектировал архитектуру с DI.",
                            "Реализовал 5 B2B-модулей.",
                            "Кодовая база превысила 11 000 строк.",
                        ],
                    ),
                    Project(
                        name="DIOM",
                        description="Финтех-приложение.",
                        tech_stack=["Flutter", "JWT", "Secure Storage"],
                        achievements=[
                            "Реализовал 4 способа входа и систему прав для 6 ролей.",
                            "20 переиспользуемых UI-компонентов.",
                        ],
                    ),
                ],
            ),
        ],
    )


def test_extract_canonical_facts_collects_numbers_globally():
    facts = extract_canonical_facts(_profile())
    nums = set(facts.allowed_numbers)
    # 3 (years), 5, 11000, 4, 6, 20
    assert nums.issuperset({"3", "5", "11000", "4", "6", "20"})


def test_extract_canonical_facts_collects_tech_globally():
    facts = extract_canonical_facts(_profile())
    assert "Flutter" in facts.allowed_tech
    assert "JWT" in facts.allowed_tech
    assert "gRPC" in facts.allowed_tech
    assert "Firebase" in facts.allowed_tech  # from secondary skills


def test_extract_canonical_facts_per_project():
    facts = extract_canonical_facts(_profile())
    om = facts.project("OtherMark")
    diom = facts.project("DIOM")
    assert om is not None and diom is not None
    assert set(om.allowed_numbers) == {"5", "11000"}
    assert set(diom.allowed_numbers) == {"4", "6", "20"}


def test_forbidden_claims_grounded_filters_out_resume_words():
    p = _profile()
    facts = extract_canonical_facts(
        p,
        forbidden_claims=["финтех", "high-load", "сотни пользователей"],
    )
    grounded = facts.forbidden_claims_grounded()
    # "финтех" appears in DIOM.description → it's NOT forbidden.
    assert "финтех" not in {g.lower() for g in grounded}
    # "high-load" and "сотни пользователей" don't appear in profile → forbidden.
    assert "high-load" in grounded
    assert "сотни пользователей" in grounded


def test_default_forbidden_claims_used_when_none_passed():
    facts = extract_canonical_facts(_profile())
    # Default list is non-empty and contains "финтех" — but profile has "финтех",
    # so it's filtered from grounded.
    assert len(facts.forbidden_claims) > 0
    grounded = facts.forbidden_claims_grounded()
    assert "финтех" not in {g.lower() for g in grounded}
    assert "high-load" in grounded


# ---------- v3 ----------


def test_vacancy_fit_matches_primary_skill():
    facts = extract_canonical_facts(_profile())
    vac = Vacancy(
        id="v1",
        title="Flutter Developer",
        description="Looking for a Flutter dev with Dart experience.",
    )
    fit = vacancy_fit(facts, vac, _profile().skills_primary)
    assert fit.primary_match is True
    assert fit.overlap_count >= 1
    assert "Flutter" in fit.matched_terms


def test_vacancy_fit_skips_unrelated_backend_vacancy():
    facts = extract_canonical_facts(_profile())
    vac = Vacancy(
        id="v2",
        title="Backend Go Engineer",
        description="Looking for a Go backend developer with Kafka and Redis.",
        requirements=["3+ years Go", "Kafka", "PostgreSQL", "Redis"],
    )
    fit = vacancy_fit(facts, vac, _profile().skills_primary)
    # Profile has none of Go/Kafka/Redis → no primary match, no overlap.
    assert fit.primary_match is False
    assert fit.overlap_count == 0


def test_vacancy_fit_uses_requirements_and_tags():
    facts = extract_canonical_facts(_profile())
    vac = Vacancy(
        id="v3",
        title="Mobile dev",
        description="Native mobile developer.",
        requirements=["Опыт работы с Flutter обязателен"],
        tags=["dart", "mobile"],
    )
    fit = vacancy_fit(facts, vac, _profile().skills_primary)
    assert fit.primary_match is True


# ---------- v4 ----------


def test_vacancy_fit_matches_multi_word_tech():
    """Multi-word tech tokens like 'Clean Architecture' must be matched —
    in v3 they were silently invisible to the fit gate."""
    facts = extract_canonical_facts(_profile())
    vac = Vacancy(
        id="v4",
        title="Mobile Engineer",
        description=(
            "We use Clean Architecture and Secure Storage with gRPC. "
            "Flutter not strictly required, but Dart helpful."
        ),
    )
    fit = vacancy_fit(facts, vac, _profile().skills_primary)
    assert "Clean Architecture" in fit.matched_terms
    assert fit.overlap_count >= 1


def test_vacancy_fit_matches_multi_word_primary_skill():
    """Primary skills can be multi-word too (e.g. 'Clean Architecture')."""
    facts = extract_canonical_facts(_profile())
    vac = Vacancy(
        id="v4b",
        title="Backend dev",
        description="We follow Clean Architecture rigorously.",
    )
    primary = ["Clean Architecture"]
    fit = vacancy_fit(facts, vac, primary)
    assert fit.primary_match is True


def test_vacancy_fit_single_word_still_exact_match():
    """Sanity: single-word tech must NOT match as a substring inside a
    longer word — 'dart' should not match 'darts'."""
    facts = extract_canonical_facts(_profile())
    vac = Vacancy(
        id="v4c",
        title="Game dev",
        description="Throwing darts in our office tournament.",
    )
    fit = vacancy_fit(facts, vac, _profile().skills_primary)
    # 'darts' must NOT be matched as 'Dart' tech.
    matched_lower = {t.lower() for t in fit.matched_terms}
    assert "dart" not in matched_lower


def test_position_industry_lands_in_profile_text_lower():
    """Industry='Финтех' on the position must make 'финтех' a known word
    so the forbidden_claim grounding doesn't flag it."""
    p = Profile(
        name="X",
        experience_years=3,
        skills_primary=["Flutter"],
        positions=[
            Position(
                title="Flutter-разработчик",
                company="Bank",
                industry="Финтех",
                projects=[
                    Project(
                        name="App",
                        description="Banking application.",
                        tech_stack=["Flutter"],
                        achievements=["Сделал что-то полезное."],
                    ),
                ],
            ),
        ],
    )
    facts = extract_canonical_facts(p, forbidden_claims=["финтех"])
    assert "финтех" in facts.profile_text_lower
    # Because "финтех" is now present in the profile text, it must be
    # filtered out of the grounded forbidden list.
    assert "финтех" not in facts.forbidden_claims_grounded()
