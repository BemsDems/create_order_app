"""Pass 2: structured-JSON -> letter text."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .prompts.analyzer import render_resume_block
from .prompts.writer import WRITER_SYSTEM, build_writer_user


async def write_letter(
    llm: LLMClient,
    analyzer_json: Dict[str, Any],
    profile_dict: Dict[str, Any],
    *,
    used_starts: Optional[List[str]] = None,
    feedback: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 400,
) -> str:
    user_prompt = build_writer_user(
        analyzer_json=analyzer_json,
        resume_block=render_resume_block(profile_dict),
        used_starts=used_starts,
        feedback=feedback,
    )
    raw = await llm.generate(
        system_prompt=WRITER_SYSTEM,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=False,
    )
    return _strip_signature_lines(raw)


def _strip_signature_lines(text: str) -> str:
    """Remove a trailing 'С уважением, …\\n<name>' block if the model added one.

    Walks the text from the bottom up and cuts off everything from the line
    that starts with 'С уважением' onward.
    """
    lines = text.splitlines()
    # Strip trailing whitespace-only lines.
    while lines and not lines[-1].strip():
        lines.pop()

    # Find the last 'С уважением' line and drop it + anything after it.
    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip().lower()
        if stripped.startswith("с уважением"):
            lines = lines[:idx]
            break

    # Re-strip trailing whitespace exposed by the cut.
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines).strip()
