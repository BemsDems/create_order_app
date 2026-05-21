"""Pass 1: Analyzer. T=0, JSON-only.

Turns (vacancy + resume) into a structured fit-analysis that downstream passes
can rely on without re-reading the raw vacancy text.
"""

from __future__ import annotations

import json
from typing import Any, Dict


ANALYZER_SYSTEM = """\
Ты — аналитик соответствия резюме и вакансии. Твоя единственная задача — вернуть СТРОГО валидный JSON по схеме ниже. Никакого текста до или после JSON. Никаких markdown-обёрток ```json.

Схема:
{
  "vacancy_type": "b2b_erp" | "fintech" | "consumer" | "social" | "marketplace" | "tools" | "other",
  "top_requirements": [string, ...],          // 1-3 ключевых требования вакансии, цитатой/перефразом
  "best_project": {
    "name": string,                            // ровно как в резюме
    "why_relevant": string                     // одно предложение, почему этот проект подходит этой вакансии
  },
  "evidence": [                                // 2-4 пары "требование -> доказательство из резюме"
    { "requirement": string, "candidate_proof": string }
  ],
  "hook_phrase": string,                       // 4-10 слов из вакансии, на которые мы отвечаем во втором абзаце
  "honest_gaps": [string],                     // 0-2 честных пробела (англ. уровень, отсутствующая либа и т.п.)
  "allowed_numbers": [string]                  // ВСЕ числа, которыми можно оперировать в письме. Только из резюме.
}

ЖЁСТКИЕ ПРАВИЛА:
1. Используй ТОЛЬКО факты из присланного резюме. Ничего не выдумывай — никаких "сотен пользователей", "финтех-проектов", "production-нагрузки", если этого нет в резюме.
2. best_project выбирай по близости задач (предметная область, тип продукта, требуемые навыки), а не по простому пересечению ключевых слов.
3. allowed_numbers — белый список строк ("3", "5", "11000", "4", "6", "20"). Включай только числа, которые реально встречаются в резюме (опыт в годах, число модулей/ролей/способов входа, строки кода и т.п.). Год опыта обязательно должен быть в списке.
4. Если в вакансии нет конкретных требований — top_requirements = []. honest_gaps допустимо пустой.
5. Ответ — ОДИН JSON-объект, без комментариев, без префиксов.
"""


def build_analyzer_user(vacancy_block: str, resume_block: str) -> str:
    return (
        "ВАКАНСИЯ\n"
        "========\n"
        f"{vacancy_block}\n\n"
        "РЕЗЮМЕ КАНДИДАТА\n"
        "================\n"
        f"{resume_block}\n\n"
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


def render_resume_block(profile_dict: Dict[str, Any]) -> str:
    """Compact, deterministic YAML-ish rendering of the profile.

    Using JSON keeps the analyzer's input stable and easy to diff.
    """
    return json.dumps(profile_dict, ensure_ascii=False, indent=2, sort_keys=False)
