"""Pass 3: Validator (semantic). T=0, JSON.

Hard rules (length, forbidden phrases, numeric whitelist) are enforced
deterministically in `src/validator.py`. This LLM pass only catches the
soft, semantic violations a regex can't see:

- whether the hook_phrase is actually addressed,
- whether the letter implicitly invents domain expertise,
- whether the second paragraph reads as advice to the company.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


VALIDATOR_SYSTEM = """\
Ты — строгий редактор. Проверь сопроводительное письмо ТОЛЬКО на семантические нарушения и верни ОДИН JSON-объект.

Схема:
{
  "passed": boolean,
  "violations": [
    { "rule": string, "evidence": string, "fix_hint": string }
  ]
}

Семантические правила:
1. invented_facts — есть ли в письме факты или домены, которых НЕТ в evidence или резюме (например "платёжные сервисы", "сотни пользователей", "production-нагрузка" если их нет в данных).
2. advice_to_company — второй абзац звучит как совет компании ("Вам нужен", "Ваш продукт требует", "Вакансия предполагает"), а не как факт о кандидате.
3. hook_not_addressed — hook_phrase из JSON-анализа никак не отражён в письме (даже перефразом).
4. weak_ending — последнее предложение — слабое ("Готов применить", "Буду рад обсудить", "Хотел бы").
5. tone_consulting — кандидат пишет так, будто консультирует, а не претендует на роль.

Не проверяй длину, числа, англицизмы и список запрещённых слов — это делает отдельный детерминированный валидатор.

passed = true только если violations = [].
Ответ — ОДИН JSON-объект, без markdown, без префиксов.
"""


def build_validator_user(
    letter_text: str,
    analyzer_json: Dict[str, Any],
    allowed_numbers: List[str],
) -> str:
    return (
        f"allowed_numbers: {json.dumps(allowed_numbers, ensure_ascii=False)}\n\n"
        f"analyzer_json: {json.dumps(analyzer_json, ensure_ascii=False)}\n\n"
        "ПИСЬМО:\n"
        "---\n"
        f"{letter_text.strip()}\n"
        "---\n\n"
        "Верни JSON."
    )
