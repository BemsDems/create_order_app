"""Pass 2: Analyzer JSON + CanonicalFacts -> letter text."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .facts import CanonicalFacts
from .llm_client import LLMClient
from .prompts.opener_pool import select_openers
from .prompts.writer import build_writer_user, select_writer_system


def build_canonical_facts_brief(
    facts: CanonicalFacts, selected_project: str
) -> Dict[str, Any]:
    """Compact dict of facts handed to the Writer — only what's needed."""
    proj = facts.project(selected_project)
    return {
        "candidate_name": facts.candidate_name,
        "selected_project_name": proj.name if proj else selected_project,
        "selected_project_tech": list(proj.tech_stack) if proj else [],
        "allowed_tech": sorted(facts.allowed_tech),
    }


async def write_letter(
    llm: LLMClient,
    analyzer_json: Dict[str, Any],
    facts: CanonicalFacts,
    *,
    used_starts: Optional[List[str]] = None,
    feedback: Optional[str] = None,
    hard_constraints: Optional[List[str]] = None,
    universal_mode: bool = False,
    temperature: float = 0.4,
    max_tokens: int = 400,
) -> str:
    system_prompt = select_writer_system(universal_mode=universal_mode)
    selected_project = str(analyzer_json.get("selected_project") or "")
    brief = build_canonical_facts_brief(facts, selected_project)
    opener_pool = select_openers(facts.experience_years, used_starts or [], n=2)

    user_prompt = build_writer_user(
        analyzer_json=analyzer_json,
        canonical_facts_brief=brief,
        opener_pool=opener_pool,
        used_starts=used_starts,
        feedback=feedback,
        hard_constraints=hard_constraints,
    )
    raw = await llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=False,
    )
    return _strip_signature_lines(raw)


_SIGNATURE_PREFIXES: tuple[str, ...] = (
    "с уважением",
    "с наилучшими",
    "спасибо за внимание",
    "спасибо за рассмотрение",
    "благодарю за внимание",
    "благодарю за рассмотрение",
    "до связи",
    "best regards",
    "kind regards",
    "regards,",
    "sincerely",
)


def _strip_signature_lines(text: str) -> str:
    """Remove a trailing signature block if the model added one.

    Walks the text from the bottom up and cuts off everything from the
    first line matching `_SIGNATURE_PREFIXES` onward.
    """
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip().lower()
        if any(stripped.startswith(prefix) for prefix in _SIGNATURE_PREFIXES):
            lines = lines[:idx]
            break

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines).strip()
