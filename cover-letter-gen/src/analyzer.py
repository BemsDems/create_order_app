"""Pass 1: vacancy/profile -> structured analysis JSON."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .models import Profile, Vacancy
from .prompts.analyzer import (
    ANALYZER_SYSTEM,
    build_analyzer_user,
    render_resume_block,
    render_vacancy_block,
)


logger = logging.getLogger(__name__)


REQUIRED_KEYS = {
    "vacancy_type",
    "top_requirements",
    "best_project",
    "evidence",
    "hook_phrase",
    "honest_gaps",
    "allowed_numbers",
}


async def analyze(
    llm: LLMClient,
    vacancy: Vacancy,
    profile: Profile,
    *,
    max_parse_retries: int = 1,
) -> Dict[str, Any]:
    """Run the analyzer LLM call and return a validated dict.

    Raises:
        ValueError: if the model fails to produce parseable JSON after retries
            or if required keys are missing.
    """
    profile_dict = profile_to_compact_dict(profile)
    user_prompt = build_analyzer_user(
        vacancy_block=render_vacancy_block(_vacancy_to_dict(vacancy)),
        resume_block=render_resume_block(profile_dict),
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

        return _normalize(parsed, profile=profile)

    raise ValueError(
        f"Analyzer failed to return valid JSON after {max_parse_retries + 1} attempts. "
        f"Last raw output (truncated): {last_raw[:400]!r}"
    )


def _vacancy_to_dict(v: Vacancy) -> Dict[str, Any]:
    return asdict(v)


def profile_to_compact_dict(profile: Profile) -> Dict[str, Any]:
    return {
        "name": profile.name,
        "experience_years": profile.experience_years,
        "experience_months": profile.experience_months,
        "summary": profile.summary,
        "skills_primary": profile.skills_primary,
        "skills_secondary": profile.skills_secondary,
        "languages": [
            {"name": l.name, "level_code": l.level_code} for l in profile.languages
        ],
        "positions": [
            {
                "title": p.title,
                "company": p.company,
                "industry": p.industry,
                "duration_months": p.duration_months,
                "projects": [
                    {
                        "name": pr.name,
                        "description": pr.description,
                        "tech_stack": pr.tech_stack,
                        "achievements": pr.achievements,
                    }
                    for pr in p.projects
                ],
            }
            for p in profile.positions
        ],
    }


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    # Strip common markdown wrappers.
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


def _normalize(parsed: Dict[str, Any], *, profile: Profile) -> Dict[str, Any]:
    """Coerce/clean the analyzer's output and harden allowed_numbers."""
    # Ensure allowed_numbers always includes the candidate's years of experience
    # so the writer can safely include it in the opening sentence.
    raw_numbers = parsed.get("allowed_numbers") or []
    allowed: List[str] = []
    seen: set[str] = set()
    for n in raw_numbers:
        s = str(n).strip()
        if s and s not in seen:
            allowed.append(s)
            seen.add(s)

    years = str(profile.experience_years)
    if profile.experience_years and years not in seen:
        allowed.append(years)
        seen.add(years)

    # Also auto-add numbers that appear in resume achievements/descriptions —
    # the analyzer sometimes misses them.
    auto_numbers = _extract_numbers_from_profile(profile)
    for s in auto_numbers:
        if s not in seen:
            allowed.append(s)
            seen.add(s)

    parsed["allowed_numbers"] = allowed
    parsed.setdefault("top_requirements", [])
    parsed.setdefault("honest_gaps", [])
    parsed.setdefault("evidence", [])
    return parsed


_NUMBER_RE = re.compile(r"(?<!\w)(\d[\d\s]{0,5}\d|\d)(?!\w)")


def _extract_numbers_from_profile(profile: Profile) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    texts: List[str] = [profile.summary]
    for pos in profile.positions:
        for pr in pos.projects:
            texts.append(pr.description)
            texts.extend(pr.achievements)
    for text in texts:
        if not text:
            continue
        for match in _NUMBER_RE.findall(text):
            s = re.sub(r"\s+", "", str(match))
            if s and s not in seen:
                out.append(s)
                seen.add(s)
    return out
