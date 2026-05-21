"""Pass 3: validator.

Two layers:

1. **Deterministic checks** (this module, pure Python): length, paragraph count,
   forbidden phrases, anglicism patterns, library names, numeric whitelist.
   These are cheap, reliable, and run before any LLM call.

2. **Semantic checks** (LLM, optional): detects hook-not-addressed,
   advice-to-company, weak ending. Run only if deterministic checks pass.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .prompts.validator import VALIDATOR_SYSTEM, build_validator_user


logger = logging.getLogger(__name__)


FORBIDDEN_PHRASES: List[str] = [
    "Готов применить",
    "Готов применять",
    "Готов включиться",
    "Готов приступить",
    "Буду рад обсудить",
    "Буду рад",
    "Хотел бы обсудить",
    "напрямую соответствует",
    "привычная задача",
    "благодаря",
    "в рамках",
    "легли в основу",
    "Ваш ",
    "Ваше ",
    "Ваша ",
    "Вашего ",
    "Вашему ",
    "Вакансия предполагает",
]


FORBIDDEN_LIBRARIES: List[str] = [
    "GetIt",
    "get_it",
    "Injectable",
    "Riverpod",
    "Provider",
    "MobX",
    "Dio",
    "Retrofit",
]


# Tech terms allowed as English even inside Russian sentences.
# Matched case-insensitively against bare lowercase tokens by `_find_anglicisms`.
ALLOWED_TECH_TERMS = {
    "Flutter", "Dart", "BLoC", "Cubit", "gRPC", "JWT", "REST", "API",
    "Web", "iOS", "Android", "Firebase", "FCM", "Clean", "Architecture",
    "GraphQL", "SQL", "SDK", "OTP", "B2B", "ERP", "CI/CD", "Git",
    "WebView", "SQLite", "URL", "HTTP", "HTTPS", "UI", "UX",
    # Common IT loan-words that are routinely used as-is in Russian.
    "production", "backend", "frontend", "mobile", "open", "source",
    "legacy", "deploy", "release", "build", "pipeline",
}


@dataclass
class Violation:
    rule: str
    evidence: str
    fix_hint: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"rule": self.rule, "evidence": self.evidence, "fix_hint": self.fix_hint}


@dataclass
class ValidationResult:
    passed: bool
    violations: List[Violation] = field(default_factory=list)
    word_count: int = 0

    def format_feedback(self) -> str:
        if not self.violations:
            return ""
        lines = ["Нарушения:"]
        for v in self.violations:
            line = f"- [{v.rule}] {v.evidence}"
            if v.fix_hint:
                line += f" — {v.fix_hint}"
            lines.append(line)
        return "\n".join(lines)


def validate_deterministic(
    letter: str,
    allowed_numbers: List[str],
    *,
    min_words: int = 100,
    max_words: int = 130,
) -> ValidationResult:
    """Run cheap, regex-level checks. Returns all violations found (not first-only)."""
    violations: List[Violation] = []
    text = letter.strip()

    # 1. Length.
    words = _word_count(text)
    if words < min_words:
        violations.append(Violation(
            rule="too_short",
            evidence=f"{words} слов (нужно {min_words}-{max_words})",
            fix_hint="Расширь абзац 1 ещё одним фактом из evidence.",
        ))
    elif words > max_words:
        violations.append(Violation(
            rule="too_long",
            evidence=f"{words} слов (нужно {min_words}-{max_words})",
            fix_hint="Сократи общие фразы, оставь только конкретику.",
        ))

    # 2. Paragraph count.
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) != 2:
        violations.append(Violation(
            rule="wrong_paragraph_count",
            evidence=f"{len(paragraphs)} абзаца (нужно ровно 2)",
            fix_hint="Раздели текст на ровно 2 абзаца пустой строкой.",
        ))

    # 3. Forbidden phrases (case-insensitive).
    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in lower:
            violations.append(Violation(
                rule="forbidden_phrase",
                evidence=phrase.strip(),
                fix_hint=f"Удали или перефразируй фразу «{phrase.strip()}».",
            ))

    # 4. Forbidden library names.
    for lib in FORBIDDEN_LIBRARIES:
        if re.search(rf"\b{re.escape(lib)}\b", text):
            violations.append(Violation(
                rule="library_name",
                evidence=lib,
                fix_hint=f"Замени «{lib}» обобщённым термином (DI / HTTP-клиент / state management).",
            ))

    # 5. Years-of-experience in the first sentence.
    # Looks for a digit at a word boundary (so "2" inside "B2B" doesn't count)
    # or an explicit "N+" / "N лет/года/год" pattern.
    first_sentence = _first_sentence(text)
    if first_sentence and not _opener_has_years(first_sentence):
        violations.append(Violation(
            rule="no_years_in_opener",
            evidence=first_sentence[:80],
            fix_hint="В первое предложение добавь число лет опыта (например, «3+ года»).",
        ))

    # 6. Numeric whitelist.
    allowed_set = {n.strip() for n in allowed_numbers if n}
    found_numbers = _extract_numbers(text)
    for n in found_numbers:
        if n not in allowed_set:
            violations.append(Violation(
                rule="invented_number",
                evidence=n,
                fix_hint=f"Число «{n}» нет в allowed_numbers — удали или замени на число из списка.",
            ))

    # 7. Minimum number of numeric facts.
    if len({n for n in found_numbers if n in allowed_set}) < 2:
        violations.append(Violation(
            rule="too_few_numbers",
            evidence=f"использовано {len(found_numbers)} чисел из allowed_numbers (нужно минимум 2)",
            fix_hint="Добавь ещё одну метрику из allowed_numbers (например, число модулей или строк кода).",
        ))

    # 8. Russian-with-anglicism heuristic.
    angl = _find_anglicisms(text)
    for word in angl:
        violations.append(Violation(
            rule="anglicism",
            evidence=word,
            fix_hint=f"Замени «{word}» русским эквивалентом.",
        ))

    return ValidationResult(passed=not violations, violations=violations, word_count=words)


async def validate_semantic(
    llm: LLMClient,
    letter: str,
    analyzer_json: Dict[str, Any],
    allowed_numbers: List[str],
) -> ValidationResult:
    """Run the LLM-based semantic validator. Returns its parsed result.

    On parse failure, returns a passing result with a logged warning — the
    deterministic layer is the source of truth.
    """
    user_prompt = build_validator_user(letter, analyzer_json, allowed_numbers)
    raw = await llm.generate(
        system_prompt=VALIDATOR_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=600,
        json_mode=True,
    )
    parsed = _try_parse_json(raw)
    if parsed is None:
        logger.warning("Semantic validator returned unparseable JSON (treating as passed): %r", raw[:200])
        return ValidationResult(passed=True, violations=[], word_count=_word_count(letter))

    violations_raw = parsed.get("violations") or []
    violations = []
    for v in violations_raw:
        if not isinstance(v, dict):
            continue
        violations.append(Violation(
            rule=str(v.get("rule", "semantic")),
            evidence=str(v.get("evidence", "")),
            fix_hint=str(v.get("fix_hint", "")),
        ))
    passed = bool(parsed.get("passed", not violations))
    return ValidationResult(
        passed=passed and not violations,
        violations=violations,
        word_count=_word_count(letter),
    )


_WORD_RE = re.compile(r"[\w’'-]+", re.UNICODE)

# A digit token at word boundaries, optionally followed by '+'.
# Matches "3", "3+", "11000" but NOT "2" inside "B2B".
_OPENER_YEARS_RE = re.compile(r"(?:(?<=\s)|^)\d+\+?(?=\s|\b)")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _first_sentence(text: str) -> str:
    # Сплитим по точке/восклицательному/вопросительному. Очень простая
    # эвристика: достаточно для нашего жанра.
    match = re.search(r"^[^\.\!\?\n]+", text)
    return match.group(0) if match else ""


def _opener_has_years(first_sentence: str) -> bool:
    """True iff the opener mentions a year count.

    Accepts "3+", "3 года", "3+ лет", a standalone digit token like "5 модулей"
    — i.e. any digit at a word boundary. Rejects digits embedded in identifiers
    such as "B2B".
    """
    return bool(_OPENER_YEARS_RE.search(first_sentence))


_NUMBER_TOKEN_RE = re.compile(r"(?<!\w)(\d[\d\s]{0,4}\d|\d)\+?(?!\w)")


def _extract_numbers(text: str) -> List[str]:
    """Find all numeric tokens, normalizing whitespace inside them.

    "11 000" and "11000" both normalize to "11000".
    """
    out: List[str] = []
    for raw in _NUMBER_TOKEN_RE.findall(text):
        normalized = re.sub(r"\s+", "", raw)
        if normalized:
            out.append(normalized)
    return out


# Very conservative English-word detector. Triggers only on lowercase ASCII
# tokens of length >= 4 that are NOT in the allowed tech-term whitelist.
_LATIN_WORD_RE = re.compile(r"\b[a-z][a-z]{3,}\b")


def _find_anglicisms(text: str) -> List[str]:
    found: List[str] = []
    allowed_lower = {t.lower() for t in ALLOWED_TECH_TERMS}
    for word in _LATIN_WORD_RE.findall(text):
        if word in allowed_lower:
            continue
        # Allow English words that are part of project names already capitalized
        # in the resume — but those would be PascalCase, not lowercase.
        found.append(word)
    return found


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed
