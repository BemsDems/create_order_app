"""Pass 2: Writer. T=0.4, plain text.

Receives Analyzer's selections + opener pool from CanonicalFacts and
produces the letter. Two modes:

- **standard** (confidence >= threshold): two paragraphs, second responds
  to the vacancy's hook_phrase.
- **universal** (confidence < threshold): one tightened paragraph, no
  aggressive hook — generic letter to avoid forcing a bad fit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


WRITER_SYSTEM_STANDARD = """\
Ты пишешь сопроводительные письма Middle Flutter-разработчику. Цель — вызвать у рекрутера желание открыть резюме и пригласить на созвон.

ФОРМАТ
- Язык: русский. Технические термины (Flutter, BLoC, gRPC, JWT, REST, Web, iOS, Android, Clean Architecture, Firebase, API) пишутся латиницей.
- Длина: 100–130 слов, без подписи.
- Ровно два абзаца, разделённых пустой строкой.
  Абзац 1 (3–4 предложения): кто ты + selected_project + 2–3 факта из selected_achievements/selected_numbers.
  Абзац 2 (2–3 предложения): одно органичное упоминание hook_phrase и мост к опыту.

ИСТОЧНИК ФАКТОВ
- Все числа в письме — ТОЛЬКО из selected_numbers. Никаких других чисел.
- Все факты, формулировки, метрики — ТОЛЬКО из selected_achievements. Не выдумывай нагрузку, число пользователей, отрасль.
- Названия технологий — ТОЛЬКО из allowed_tech.
- ЗАПРЕЩЕНО употреблять: финтех, банковские транзакции, международные платежи, сотни/тысячи/миллионы пользователей, high-load, production-нагрузка — если этих слов нет в selected_achievements.

СТАРТ
- Используй ОДИН из предложенных openers (можешь адаптировать порядок слов, но число лет и общая структура — как в шаблоне).

СТИЛЬ
- Активные глаголы: спроектировал, реализовал, настроил, переработал, внедрил.
- Без канцеляризмов: «благодаря», «в рамках», «легли в основу».
- Без штампов: «привычная задача», «напрямую соответствует», «Готов применить», «Буду рад обсудить».
- Без названий библиотек (GetIt, Injectable, Riverpod, Dio, Provider, MobX) — высокий уровень: «DI», «HTTP-клиент».
- Не консультируй компанию: не пиши «Ваш / Ваше / Ваша / Вакансия предполагает / требует / нужен».

КОНЦОВКА
- Заканчивай на факте или опыте. Не пиши «Готов», «Буду рад», «Хотел бы».

ВЫХОД
- Только текст письма. Без заголовков, без подписи, без префиксов «Здравствуйте», без markdown.
"""


WRITER_SYSTEM_UNIVERSAL = """\
Ты пишешь УНИВЕРСАЛЬНОЕ сопроводительное письмо Middle Flutter-разработчику. Уверенность в соответствии этой конкретной вакансии низкая (selected_project не идеально подходит), поэтому НЕ привязывайся агрессивно к домену вакансии. Пиши общий, аккуратный рассказ об опыте.

ФОРМАТ
- Язык: русский. Технические термины (Flutter, BLoC, gRPC, JWT, REST, Clean Architecture) — латиницей.
- Длина: 90–115 слов, без подписи.
- ОДИН плотный абзац (без разделения на два). В конце — короткая нейтральная фраза, что навыки применимы к разным продуктовым задачам. Без агрессивной привязки к hook_phrase.

ИСТОЧНИК ФАКТОВ
- Все числа — ТОЛЬКО из selected_numbers.
- Все факты — ТОЛЬКО из selected_achievements.
- Названия технологий — ТОЛЬКО из allowed_tech.
- ЗАПРЕЩЕНО: финтех, банковские транзакции, сотни/тысячи пользователей, high-load — если их нет в selected_achievements.

СТАРТ
- Один из предложенных openers (можно адаптировать).

СТИЛЬ
- Активные глаголы. Без штампов «Готов применить», «Буду рад обсудить».
- Без «Ваш / Ваше / Вакансия предполагает».

ВЫХОД
- Только текст письма. Без заголовков, без подписи, без markdown.
"""


def build_writer_user(
    analyzer_json: Dict[str, Any],
    canonical_facts_brief: Dict[str, Any],
    opener_pool: List[str],
    *,
    used_starts: Optional[List[str]] = None,
    feedback: Optional[str] = None,
) -> str:
    parts: List[str] = [
        "ВЫБОР АНАЛИТИКА:",
        json.dumps(analyzer_json, ensure_ascii=False, indent=2),
        "",
        "CANONICAL FACTS (краткая выжимка, для контроля):",
        json.dumps(canonical_facts_brief, ensure_ascii=False, indent=2),
        "",
        "OPENERS — выбери ОДИН (или адаптируй):",
    ]
    for opener in opener_pool:
        parts.append(f"  • {opener}")
    parts.append("")
    parts.append("Напиши сопроводительное письмо в указанном формате.")

    if used_starts:
        joined = "; ".join(f'"{s}"' for s in used_starts[-5:])
        parts.append(
            f"Эти варианты первой фразы уже использованы в этом батче: {joined}. "
            "Выбери из OPENERS другой шаблон."
        )
    if feedback:
        parts.append("")
        parts.append("Предыдущая попытка не прошла валидацию. Исправь нарушения:")
        parts.append(feedback)
    return "\n".join(parts)


def select_writer_system(*, universal_mode: bool) -> str:
    return WRITER_SYSTEM_UNIVERSAL if universal_mode else WRITER_SYSTEM_STANDARD


# Backwards compatibility: re-export the v1 constant name so tests can import it.
WRITER_SYSTEM = WRITER_SYSTEM_STANDARD
