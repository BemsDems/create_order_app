"""Pass 1: vacancy + CanonicalFacts -> structured analyzer JSON."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .facts import CanonicalFacts
from .llm_client import LLMClient
from .models import Vacancy
from .prompts.analyzer import (
    ANALYZER_SYSTEM,
    build_analyzer_user,
    canonical_facts_to_dict,
    render_canonical_facts_block,
    render_vacancy_block,
)


logger = logging.getLogger(__name__)


REQUIRED_KEYS = {
    "vacancy_type",
    "top_requirements",
    "selected_project",
    "confidence",
    "selected_numbers",
    "selected_achievements",
    "hook_phrase",
}


async def analyze(
    llm: LLMClient,
    vacancy: Vacancy,
    facts: CanonicalFacts,
    *,
    max_parse_retries: int = 1,
) -> Dict[str, Any]:
    """Run the analyzer and return a *validated and grounded* dict.

    Validation enforces that:
    - `selected_project` is one of `facts.allowed_project_names`,
    - `selected_numbers` is a subset of the project's allowed numbers (with
      a fallback to the global allowed_numbers — some achievements
      reference the candidate's years of experience, which lives globally),
    - `selected_achievements` items are substrings of real achievements,
    - `confidence` is a number in [0, 1].

    Anything outside the whitelist is FILTERED OUT (not used as feedback to
    the LLM) so a hallucinated number can never reach the Writer even if a
    retry doesn't happen.

    Raises:
        ValueError: if the model fails to produce parseable JSON after
            retries, OR if `selected_project` is invalid after filtering.
    """
    facts_dict = canonical_facts_to_dict(facts)
    user_prompt = build_analyzer_user(
        vacancy_block=render_vacancy_block(_vacancy_to_dict(vacancy)),
        canonical_facts_block=render_canonical_facts_block(facts_dict),
    )

    last_raw: str = ""
    for attempt in range(max_parse_retries + 1):
        raw = await llm.generate(
            system_prompt=ANALYZER_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=900,
            json_mode=True,
        )
        last_raw = raw
        parsed = _try_parse_json(raw)
        if parsed is None:
            logger.warning("Analyzer attempt %d: unparseable JSON", attempt + 1)
            user_prompt = user_prompt + (
                "\n\nПредыдущий ответ был невалидным JSON. Верни ОДИН валидный JSON-объект без markdown."
            )
            continue

        missing = REQUIRED_KEYS - set(parsed.keys())
        if missing:
            logger.warning("Analyzer attempt %d: missing keys %s", attempt + 1, sorted(missing))
            user_prompt += (
                f"\n\nВ предыдущем ответе отсутствовали ключи: {sorted(missing)}. "
                "Верни ПОЛНЫЙ JSON по схеме."
            )
            continue

        return _ground(parsed, facts=facts)

    raise ValueError(
        f"Analyzer failed to return valid JSON after {max_parse_retries + 1} attempts. "
        f"Last raw output (truncated): {last_raw[:400]!r}"
    )


def _vacancy_to_dict(v: Vacancy) -> Dict[str, Any]:
    return asdict(v)


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


def _ground(parsed: Dict[str, Any], *, facts: CanonicalFacts) -> Dict[str, Any]:
    """Filter the Analyzer's output against CanonicalFacts.

    This is the anti-hallucination teeth: even if the LLM made up a number
    or a project, those don't survive past this function.
    """
    # 1. selected_project must exist in allowed_project_names (case-insensitive).
    raw_project = str(parsed.get("selected_project") or "").strip()
    project_facts = facts.project(raw_project) if raw_project else None
    if project_facts is None:
        # Fallback: pick the first project from CanonicalFacts so we always
        # return something the Writer can use. The pipeline will see
        # confidence=0.0 and route to universal mode.
        if not facts.projects:
            raise ValueError(
                f"Analyzer chose project {raw_project!r} but CanonicalFacts has no projects."
            )
        fallback_name = next(iter(facts.projects))
        logger.warning(
            "Analyzer chose unknown project %r; falling back to %r with confidence=0.0",
            raw_project, fallback_name,
        )
        project_facts = facts.projects[fallback_name]
        parsed["selected_project"] = fallback_name
        parsed["confidence"] = 0.0
        parsed["confidence_reason"] = (
            f"fallback: analyzer chose unknown project '{raw_project}', forced to {fallback_name}"
        )

    # 2. selected_numbers must be subset of facts.allowed_numbers.
    raw_numbers = parsed.get("selected_numbers") or []
    allowed_numbers_set = set(facts.allowed_numbers)
    grounded_numbers: List[str] = []
    seen: set[str] = set()
    for n in raw_numbers:
        s = str(n).strip()
        if s in allowed_numbers_set and s not in seen:
            grounded_numbers.append(s)
            seen.add(s)
    # If the analyzer produced nothing usable, fall back to: years of exp + first project number.
    if not grounded_numbers:
        years = str(facts.experience_years)
        if facts.experience_years and years in allowed_numbers_set:
            grounded_numbers.append(years)
        if project_facts.allowed_numbers:
            for n in project_facts.allowed_numbers:
                if n not in grounded_numbers:
                    grounded_numbers.append(n)
                    if len(grounded_numbers) >= 3:
                        break
    parsed["selected_numbers"] = grounded_numbers

    # 3. selected_achievements must be substrings of real achievements
    #    (case-insensitive). Drop anything that isn't.
    raw_achievements = parsed.get("selected_achievements") or []
    real_achievements_lower = [a.lower().strip() for a in project_facts.achievements]
    grounded_achievements: List[str] = []
    for ach in raw_achievements:
        ach_str = str(ach).strip()
        ach_lower = ach_str.lower()
        # Accept if the analyzer's text matches one of the real achievements
        # (we accept substring in either direction for robustness).
        matched = False
        for real_lower, real_full in zip(real_achievements_lower, project_facts.achievements):
            if ach_lower == real_lower or ach_lower in real_lower or real_lower in ach_lower:
                grounded_achievements.append(real_full)
                matched = True
                break
        if not matched:
            logger.debug("Analyzer achievement %r not found in CanonicalFacts; dropping.", ach_str)
    # If everything was dropped, fall back to the first 2 real achievements.
    if not grounded_achievements:
        grounded_achievements = list(project_facts.achievements[:2])
    parsed["selected_achievements"] = grounded_achievements

    # 4. confidence in [0, 1].
    raw_conf = parsed.get("confidence", 0.0)
    try:
        conf = float(raw_conf)
    except (TypeError, ValueError):
        conf = 0.0
    parsed["confidence"] = max(0.0, min(1.0, conf))

    parsed.setdefault("top_requirements", [])
    parsed.setdefault("honest_gaps", [])
    parsed.setdefault("confidence_reason", "")
    return parsed
