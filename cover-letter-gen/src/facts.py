"""Canonical facts extracted deterministically from a Profile.

This is the SINGLE SOURCE OF TRUTH for what the LLM is allowed to claim.

Rationale: in v1 the Analyzer LLM produced `allowed_numbers` itself, which
made anti-hallucination self-referential — if the Analyzer hallucinated a
number, the Writer was then free to use it. In v2, the Analyzer no longer
*produces* a whitelist; it only *selects* facts/numbers from a whitelist
that this module built from the resume YAML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .models import Profile, Project, Vacancy


# Default smell-phrases — common hallucinations seen in cover letters.
# A claim is flagged iff it appears in the letter AND nowhere in the resume.
DEFAULT_FORBIDDEN_CLAIMS: List[str] = [
    "финтех",
    "финтех-проекты",
    "финтех-платформ",
    "банковские транзакции",
    "международные платежи",
    "платёжные сервисы",
    "платежные сервисы",
    "high-load",
    "highload",
    "production-нагрузка",
    "сотни пользователей",
    "тысячи пользователей",
    "миллионы пользователей",
    "сотни тысяч пользователей",
]


@dataclass
class ProjectFacts:
    """Per-project canonical facts."""

    name: str
    company: str
    industry: str
    description: str
    tech_stack: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    allowed_numbers: List[str] = field(default_factory=list)


@dataclass
class CanonicalFacts:
    """All deterministic facts about the candidate.

    Built once per Profile at pipeline init; passed read-only into the
    Analyzer (as context) and the validator (as ground truth).
    """

    candidate_name: str
    experience_years: int
    experience_months: int
    summary: str
    profile_text_lower: str          # for forbidden_claim grounding
    allowed_numbers: List[str]       # global whitelist (years + every number across all projects)
    allowed_tech: Set[str]           # tech terms ever mentioned (case-insensitive)
    allowed_project_names: Set[str]
    allowed_company_names: Set[str]
    projects: Dict[str, ProjectFacts] = field(default_factory=dict)
    forbidden_claims: List[str] = field(default_factory=list)

    def project(self, name: str) -> ProjectFacts | None:
        # Case-insensitive lookup; the LLM may produce slightly different casing.
        for key, facts in self.projects.items():
            if key.lower() == name.lower():
                return facts
        return None

    def forbidden_claims_grounded(self) -> List[str]:
        """Subset of `forbidden_claims` that do NOT appear in the resume.

        If "финтех" appears in the resume itself, it's not forbidden — the
        candidate IS in fintech.
        """
        return [
            phrase for phrase in self.forbidden_claims
            if phrase.lower() not in self.profile_text_lower
        ]


# Number tokens: 1-6 digits with optional thousands-style space ("11 000").
_NUMBER_RE = re.compile(r"(?<!\w)(\d[\d\s]{0,4}\d|\d)\+?(?!\w)")


def extract_canonical_facts(
    profile: Profile,
    *,
    forbidden_claims: List[str] | None = None,
) -> CanonicalFacts:
    """Build a `CanonicalFacts` object from a Profile.

    All extraction is regex/string-level — no LLM involved.
    """
    all_numbers: List[str] = []
    all_tech: Set[str] = set()
    all_projects: Set[str] = set()
    all_companies: Set[str] = set()
    project_facts: Dict[str, ProjectFacts] = {}
    text_chunks: List[str] = [profile.summary or ""]

    if profile.experience_years:
        all_numbers.append(str(profile.experience_years))

    for skill in profile.skills_primary + profile.skills_secondary:
        if skill.strip():
            all_tech.add(skill.strip())

    for position in profile.positions:
        if position.company.strip():
            all_companies.add(position.company.strip())
        if position.industry.strip():
            text_chunks.append(position.industry)
        for project in position.projects:
            facts = _project_facts_from(project, position.company, position.industry)
            project_facts[project.name] = facts
            if project.name.strip():
                all_projects.add(project.name.strip())
            all_tech.update(facts.tech_stack)
            for n in facts.allowed_numbers:
                if n not in all_numbers:
                    all_numbers.append(n)
            text_chunks.append(project.description or "")
            text_chunks.extend(project.achievements)

    text_chunks.extend(all_projects)
    text_chunks.extend(all_companies)
    profile_text = " ".join(c for c in text_chunks if c)

    return CanonicalFacts(
        candidate_name=profile.name,
        experience_years=profile.experience_years,
        experience_months=profile.experience_months,
        summary=profile.summary,
        profile_text_lower=profile_text.lower(),
        allowed_numbers=_dedup(all_numbers),
        allowed_tech=all_tech,
        allowed_project_names=all_projects,
        allowed_company_names=all_companies,
        projects=project_facts,
        forbidden_claims=list(forbidden_claims) if forbidden_claims is not None else list(DEFAULT_FORBIDDEN_CLAIMS),
    )


def _project_facts_from(project: Project, company: str, industry: str) -> ProjectFacts:
    numbers: List[str] = []
    text = " ".join([project.description or "", *project.achievements])
    for raw in _NUMBER_RE.findall(text):
        normalized = re.sub(r"\s+", "", raw)
        if normalized and normalized not in numbers:
            numbers.append(normalized)
    return ProjectFacts(
        name=project.name,
        company=company,
        industry=industry,
        description=project.description,
        tech_stack=list(project.tech_stack),
        achievements=list(project.achievements),
        allowed_numbers=numbers,
    )


def _dedup(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# Words tokenizer for the fit-gate (incl. cyrillic, latin, digits, '+/#').
_FIT_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9+/#-]*", re.UNICODE)


@dataclass
class VacancyFit:
    """Deterministic match between a vacancy and the profile.

    overlap_count is the number of distinct allowed-tech tokens that appear
    in the vacancy text. primary_match is true iff the profile's primary
    skill (e.g. "Flutter") appears in the vacancy text.
    """

    overlap_count: int
    primary_match: bool
    matched_terms: List[str]


def vacancy_fit(facts: CanonicalFacts, vacancy: Vacancy, primary_skills: List[str]) -> VacancyFit:
    """Score the vacancy against the candidate's tech.

    Used as a pre-Analyzer gate: if the vacancy doesn't mention even one of
    the candidate's primary skills (Flutter, Dart, Mobile, ...), skip it
    without an LLM call. Saves tokens AND avoids the model being tempted
    to invent matching experience.

    Multi-word tech tokens (e.g. "Clean Architecture", "Secure Storage")
    are matched as substrings against the full lowercased vacancy text —
    single-word tokenization can't see them.
    """
    text_parts: List[str] = [vacancy.title or "", vacancy.description or ""]
    text_parts.extend(vacancy.requirements or [])
    text_parts.extend(vacancy.tags or [])
    text = " ".join(p for p in text_parts if p)
    text_lower = text.lower()
    tokens_lower = {m.group(0).lower() for m in _FIT_WORD_RE.finditer(text)}

    matched: List[str] = []
    matched_set: Set[str] = set()
    for tech in facts.allowed_tech:
        if not tech.strip():
            continue
        tech_lower = tech.lower()
        if tech_lower in matched_set:
            continue
        if " " in tech_lower:
            # Multi-word tokens — substring match on the full lowercased text.
            if tech_lower in text_lower:
                matched.append(tech)
                matched_set.add(tech_lower)
        else:
            # Single-word tokens — exact word match to avoid false positives
            # like "dart" matching inside "darts".
            if tech_lower in tokens_lower:
                matched.append(tech)
                matched_set.add(tech_lower)

    primary_match = False
    for skill in primary_skills:
        if not skill.strip():
            continue
        skill_lower = skill.lower()
        if " " in skill_lower:
            if skill_lower in text_lower:
                primary_match = True
                break
        else:
            if skill_lower in tokens_lower:
                primary_match = True
                break

    return VacancyFit(
        overlap_count=len(matched),
        primary_match=primary_match,
        matched_terms=matched,
    )
