"""Pass 1: Analyzer. T=0, JSON-only.

Turns (vacancy + resume + CanonicalFacts) into a structured fit-analysis.

Key v2 change: the Analyzer no longer *produces* a whitelist of allowed
numbers/facts. The whitelist comes from the resume deterministically; the
Analyzer only *selects* a subset of it for this specific vacancy and
reports its confidence that the selected project is a good fit.
"""

from __future__ import annotations

import json
from typing import Any, Dict


ANALYZER_SYSTEM = """\
Ты — аналитик соответствия резюме и вакансии. Возвращай СТРОГО валидный JSON по схеме ниже. Никакого текста до или после JSON. Без markdown-обёрток ```json.

Схема:
{
  "vacancy_type": "b2b_erp" | "fintech" | "consumer" | "social" | "marketplace" | "tools" | "other",
  "top_requirements": [string, ...],          // 1-3 ключевых требования вакансии
  "selected_project": string,                  // РОВНО одно из имён проектов из CANONICAL_FACTS.projects
  "confidence": number,                        // 0.0-1.0, насколько selected_project релевантен этой вакансии
  "confidence_reason": string,                 // одно предложение, обоснование оценки
  "selected_numbers": [string, ...],           // 2-4 числа ИЗ CANONICAL_FACTS.allowed_numbers, которые планируешь использовать
  "selected_achievements": [string, ...],      // 2-3 достижения ДОСЛОВНО из CANONICAL_FACTS.projects[selected_project].achievements
  "hook_phrase": string,                       // 4-10 слов из ОПИСАНИЯ ВАКАНСИИ, к которым письмо отвечает во втором абзаце
  "honest_gaps": [string]                      // 0-2 честных пробела (опционально)
}

ЖЁСТКИЕ ПРАВИЛА:
1. selected_project — ТОЛЬКО из списка CANONICAL_FACTS.projects. Никаких «новых» названий проектов.
2. selected_numbers — ТОЛЬКО из CANONICAL_FACTS.allowed_numbers. Никаких других чисел.
3. selected_achievements — копируй дословно из CANONICAL_FACTS.projects[selected_project].achievements. Не переписывай, не сокращай.
4. confidence:
   - 0.8-1.0: проект и вакансия в одной предметной области, явные пересечения навыков.
   - 0.5-0.8: общая технологическая база совпадает, домены пересекаются частично.
   - 0.3-0.5: только базовый стек (Flutter, BLoC) совпадает, домен сильно разный.
   - 0.0-0.3: вообще не похоже.
5. hook_phrase — это цитата/перефраз из вакансии (то, на что будем отвечать), а НЕ перечисление наших навыков.
6. Если в резюме нет проекта, релевантного вакансии — выбирай ближайший по стеку и ставь честный низкий confidence.

Ответ — ОДИН JSON-объект, без комментариев, без префиксов.
"""


def build_analyzer_user(
    vacancy_block: str,
    canonical_facts_block: str,
) -> str:
    return (
        "ВАКАНСИЯ\n"
        "========\n"
        f"{vacancy_block}\n\n"
        "CANONICAL_FACTS (твой ЕДИНСТВЕННЫЙ источник истины)\n"
        "====================================================\n"
        f"{canonical_facts_block}\n\n"
        "Верни JSON по схеме."
    )


def render_vacancy_block(data: Dict[str, Any]) -> str:
    title = data.get("title") or "—"
    company = data.get("company") or "—"
    work_format = data.get("work_format") or "не указан"
    location = data.get("location") or "—"
    description = (data.get("description") or "").strip() or "—"
    requirements = data.get("requirements") or []
    parts = [
        f"Должность: {title}",
        f"Компания: {company}",
        f"Формат работы: {work_format}",
        f"Локация: {location}",
        "Описание:",
        description,
    ]
    if requirements:
        parts.append("Требования:")
        for r in requirements[:15]:
            parts.append(f"- {r}")
    return "\n".join(parts)


def render_canonical_facts_block(facts_dict: Dict[str, Any]) -> str:
    """Compact JSON rendering of CanonicalFacts.

    Listing projects with their achievements gives the Analyzer exactly the
    text it must select from (no rewriting allowed).
    """
    return json.dumps(facts_dict, ensure_ascii=False, indent=2, sort_keys=False)


def canonical_facts_to_dict(facts) -> Dict[str, Any]:
    """Convert a `CanonicalFacts` instance into a compact dict for the prompt.

    Importing lazily here to avoid a cycle (`facts` imports `models`,
    `prompts.analyzer` would import `facts`).
    """
    return {
        "candidate_name": facts.candidate_name,
        "experience_years": facts.experience_years,
        "allowed_numbers": facts.allowed_numbers,
        "allowed_tech": sorted(facts.allowed_tech),
        "projects": {
            name: {
                "company": p.company,
                "industry": p.industry,
                "description": p.description,
                "tech_stack": p.tech_stack,
                "achievements": p.achievements,
                "allowed_numbers": p.allowed_numbers,
            }
            for name, p in facts.projects.items()
        },
    }
