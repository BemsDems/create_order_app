"""Pass 2: Writer. T=0.4, plain text.

Receives the Analyzer's JSON + the profile and produces the letter text.
Short prompt by design — most of the rules live in deterministic
post-checks, not in negative-instruction soup.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


WRITER_SYSTEM = """\
Ты пишешь сопроводительные письма Middle Flutter-разработчику. Цель — вызвать у рекрутера желание открыть резюме и пригласить на созвон.

ФОРМАТ
- Язык: русский. Технические термины (Flutter, BLoC, gRPC, JWT, REST, Web, iOS, Android, Clean Architecture, Firebase, API) пишутся латиницей.
- Длина: 100–130 слов, без подписи.
- Ровно два абзаца, разделённых пустой строкой.
  Абзац 1 (3–4 предложения): кто ты + проект из best_project + 2–3 конкретных факта/цифры.
  Абзац 2 (2–3 предложения): одно органичное упоминание hook_phrase и мост к твоему опыту.

ИСТОЧНИК ФАКТОВ
- Все числа в письме — ТОЛЬКО из allowed_numbers. Никаких "сотен пользователей", "production-нагрузки", "финтех-проектов", если этого нет в allowed_numbers/evidence.
- Все названия проектов и технологий — только из evidence/резюме.
- Никаких выдуманных доменов (платёжные сервисы, банковские системы), если их нет в резюме.

СТИЛЬ
- Активные глаголы: спроектировал, реализовал, настроил, переработал, внедрил.
- Без канцеляризмов: "благодаря", "в рамках", "легли в основу".
- Без штампов: "привычная задача", "напрямую соответствует", "Готов применить", "Буду рад обсудить".
- Без названий библиотек (GetIt, Injectable, Riverpod, Dio, Provider, MobX) — высокий уровень: "DI", "HTTP-клиент".
- Не консультируй компанию: не пиши "Ваш / Ваше / Ваша / Вакансия предполагает / требует / нужен".
- honest_gaps в письмо НЕ выноси и не оправдывайся.

ПЕРСОНАЛИЗАЦИЯ
- В абзаце 2 ОБЯЗАТЕЛЬНО органичное упоминание hook_phrase (или сильное перефразирование БЕЗ слов "ваш/ваше/ваша/вакансия"). Допустимы конструкции: "для подобных задач", "в задачах работы с …", "опыт применим к …".

КОНЦОВКА
- Заканчивай на факте или опыте. Не пиши "Готов", "Буду рад", "Хотел бы".

ВЫХОД
- Только текст письма. Без заголовков, без подписи, без префиксов "Здравствуйте", без markdown.
"""


def build_writer_user(
    analyzer_json: Dict[str, Any],
    resume_block: str,
    *,
    used_starts: Optional[list[str]] = None,
    feedback: Optional[str] = None,
) -> str:
    parts = [
        "JSON-анализ вакансии:",
        json.dumps(analyzer_json, ensure_ascii=False, indent=2),
        "",
        "Резюме (для справки, не пересказывай целиком):",
        resume_block,
        "",
        "Напиши сопроводительное письмо. 100–130 слов. Не выходи за allowed_numbers.",
    ]
    if used_starts:
        joined = "; ".join(f'"{s}"' for s in used_starts[-5:])
        parts.append(
            f"Эти варианты первой фразы уже использованы в этом батче: {joined}. "
            "Сформулируй начало по-другому, сохранив указание лет опыта."
        )
    if feedback:
        parts.append("")
        parts.append("Предыдущая попытка не прошла валидацию. Исправь нарушения:")
        parts.append(feedback)
    return "\n".join(parts)
