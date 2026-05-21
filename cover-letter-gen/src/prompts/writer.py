"""Pass 2: Writer. T=0.4, plain text.

v3 changes (vs v2):
- System and user prompts contain NO English JSON labels (`selected_project`,
  `allowed_tech`, `openers`, `confidence`, ...). Small models love to copy
  those into the letter. Everything is referenced in Russian.
- System prompt has NO literal length digits ("100–130 слов" → just
  "одно письмо в один-два абзаца"). Length is enforced by the validator —
  the model used to read those digits and write them out as "90", "115"
  numbers in the letter itself.
- User prompt is plain Russian text, not JSON. Same reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Both system prompts deliberately avoid English JSON labels and digit
# length ranges. Length is enforced downstream by the validator.

WRITER_SYSTEM_STANDARD = """\
Ты пишешь сопроводительное письмо Middle Flutter-разработчику. Цель — вызвать у рекрутера желание открыть резюме и пригласить на созвон.

ФОРМАТ
- Язык: русский. Латиницей пишутся только названия технологий (например: Flutter, BLoC, gRPC, JWT, REST, Web, iOS, Android, Clean Architecture, Firebase, API).
- Два абзаца, разделённых пустой строкой. Первый — о себе и проекте; второй — про связь с этой вакансией.
- Без подписи, без приветствия, без заголовков, без markdown, без любых служебных меток.

ИСТОЧНИК ФАКТОВ
- В письме можно упоминать ТОЛЬКО факты из секции «Факты для упоминания» в пользовательском сообщении.
- Все числа в письме — ТОЛЬКО из секции «Разрешённые числа». Если нужного числа нет в этом списке — переформулируй без числа.
- Все названия технологий — ТОЛЬКО из секции «Разрешённые технологии».
- НЕ выдумывай отрасль, нагрузку, количество пользователей, размер команды, домен продукта.

ЗАПРЕЩЕНО
- Слова «финтех», «банковские транзакции», «международные платежи», «сотни / тысячи / миллионы пользователей», «high-load», «production-нагрузка» — если их нет в «Факты для упоминания».
- Технологии вне «Разрешённых технологий» (например, Go, Kafka, Redis, NodeJS — даже если они упомянуты в вакансии).
- Названия библиотек: GetIt, Injectable, Riverpod, Dio, Provider, MobX. Пиши обобщённо: «DI», «HTTP-клиент», «state management».
- Слова «Ваш / Ваше / Ваша», «Вакансия предполагает / требует». Не консультируй компанию.
- Штампы: «Готов применить», «Буду рад обсудить», «Хотел бы», «привычная задача», «напрямую соответствует».
- Канцеляризмы: «благодаря», «в рамках», «легли в основу».
- Любые служебные слова из пользовательского сообщения: «openers», «confidence», «achievements», «facts», «hook», «selected», «candidate». Их в письме быть не должно.

СТИЛЬ
- Активные глаголы: спроектировал, реализовал, настроил, переработал, внедрил.
- Заканчивай на конкретном факте или опыте, а не на «Готов» / «Буду рад».

ВЫХОД
- Только готовый текст письма, ничего больше.
"""


WRITER_SYSTEM_UNIVERSAL = """\
Ты пишешь УНИВЕРСАЛЬНОЕ сопроводительное письмо Middle Flutter-разработчику. Вакансия не идеально совпадает с опытом, поэтому НЕ привязывайся агрессивно к её домену — пиши общий, аккуратный рассказ об опыте.

ФОРМАТ
- Язык: русский. Латиницей только названия технологий (Flutter, BLoC, gRPC, JWT, REST, Clean Architecture).
- ОДИН плотный абзац. Без пустых строк внутри. Без подписи, без приветствия, без заголовков, без markdown.

ИСТОЧНИК ФАКТОВ
- Можно упоминать ТОЛЬКО факты из секции «Факты для упоминания».
- Числа — ТОЛЬКО из секции «Разрешённые числа».
- Технологии — ТОЛЬКО из секции «Разрешённые технологии».

ЗАПРЕЩЕНО
- Слова «финтех», «банковские транзакции», «сотни / тысячи / миллионы пользователей», «high-load» — если их нет в «Факты для упоминания».
- Технологии вне «Разрешённых технологий» (включая Go, Kafka, Redis, NodeJS).
- Названия библиотек (GetIt, Riverpod, Dio, Provider, MobX) — пиши обобщённо.
- Слова «Ваш / Ваше», «Вакансия предполагает».
- Штампы «Готов применить», «Буду рад обсудить».
- Любые служебные слова из пользовательского сообщения: «openers», «confidence», «achievements», «facts», «hook», «selected», «candidate».

СТИЛЬ
- Активные глаголы. Заканчивай нейтральной фразой, что навыки применимы к разным продуктовым задачам.

ВЫХОД
- Только готовый текст письма, ничего больше.
"""


def build_writer_user(
    analyzer_json: Dict[str, Any],
    canonical_facts_brief: Dict[str, Any],
    opener_pool: List[str],
    *,
    used_starts: Optional[List[str]] = None,
    feedback: Optional[str] = None,
    hard_constraints: Optional[List[str]] = None,
) -> str:
    """Build the Writer user prompt as plain Russian text — no JSON.

    The user prompt deliberately uses Russian section headings only. Small
    models tend to copy English JSON labels into the output verbatim, so
    we don't show them any.

    Args:
        analyzer_json: full Analyzer output (after grounding).
        canonical_facts_brief: small dict from `build_canonical_facts_brief`
            with `selected_project_name`, `selected_project_tech`,
            `allowed_tech`.
        opener_pool: 1-3 candidate opening sentences.
        used_starts: openers already used in this batch.
        feedback: violation summary from previous retry, plain text.
        hard_constraints: extra strict lines (e.g. "СТРОГО не пиши X")
            added at the end of the message after the 2nd retry.
    """
    project_name = str(canonical_facts_brief.get("selected_project_name") or "")
    project_tech = list(canonical_facts_brief.get("selected_project_tech") or [])
    allowed_tech = list(canonical_facts_brief.get("allowed_tech") or [])

    selected_numbers = list(analyzer_json.get("selected_numbers") or [])
    selected_achievements = list(analyzer_json.get("selected_achievements") or [])
    hook = str(analyzer_json.get("hook_phrase") or "").strip()

    # Tech list shown to the writer = project tech first (most relevant),
    # then the rest. Trimmed to ~12 to keep prompt small.
    tech_for_prompt: List[str] = []
    for t in project_tech + allowed_tech:
        if t and t not in tech_for_prompt:
            tech_for_prompt.append(t)
    tech_for_prompt = tech_for_prompt[:12]

    parts: List[str] = []
    parts.append("Напиши сопроводительное письмо по правилам из системного сообщения.")
    parts.append("")
    parts.append(f"Проект, который надо упомянуть: {project_name or 'не задан'}.")

    if selected_achievements:
        parts.append("")
        parts.append("Факты для упоминания (выбери 2-3, можешь немного перефразировать):")
        for fact in selected_achievements:
            parts.append(f"- {fact}")

    if selected_numbers:
        parts.append("")
        parts.append("Разрешённые числа (использовать ТОЛЬКО эти, другие — запрещены):")
        parts.append("- " + ", ".join(selected_numbers))

    if tech_for_prompt:
        parts.append("")
        parts.append("Разрешённые технологии (использовать ТОЛЬКО эти):")
        parts.append("- " + ", ".join(tech_for_prompt))

    if hook:
        parts.append("")
        parts.append(f"Фраза из вакансии, к которой относится второй абзац: «{hook}».")

    if opener_pool:
        parts.append("")
        parts.append("Можешь начать одной из этих фраз (адаптируй порядок слов, но число лет сохрани):")
        for opener in opener_pool:
            parts.append(f"- {opener}")

    if used_starts:
        joined = "; ".join(f'«{s}»' for s in used_starts[-5:] if s)
        if joined:
            parts.append("")
            parts.append(
                f"Эти варианты первой фразы уже использовались в текущем батче: {joined}. "
                "Возьми из предложенных другой шаблон."
            )

    if feedback:
        parts.append("")
        parts.append("Предыдущая попытка письма не прошла проверку. Исправь нарушения:")
        parts.append(feedback)

    if hard_constraints:
        parts.append("")
        parts.append("СТРОГО (нарушение → автоматическое отклонение):")
        for line in hard_constraints:
            parts.append(f"- {line}")

    parts.append("")
    parts.append("Верни ТОЛЬКО текст письма, без любых пояснений.")
    return "\n".join(parts)


def select_writer_system(*, universal_mode: bool) -> str:
    return WRITER_SYSTEM_UNIVERSAL if universal_mode else WRITER_SYSTEM_STANDARD


# Backwards compatibility for v1 tests / external imports.
WRITER_SYSTEM = WRITER_SYSTEM_STANDARD
