"""Prompt strings for the 3-pass pipeline.

Kept in plain Python modules (not Jinja) — the small amount of templating
we need is simple string substitution on a few well-defined slots.
"""

from .analyzer import (
    ANALYZER_SYSTEM,
    build_analyzer_user,
    canonical_facts_to_dict,
    render_canonical_facts_block,
    render_vacancy_block,
)
from .opener_pool import OPENER_TEMPLATES, select_openers
from .validator import VALIDATOR_SYSTEM, build_validator_user
from .writer import (
    WRITER_SYSTEM,
    WRITER_SYSTEM_STANDARD,
    WRITER_SYSTEM_UNIVERSAL,
    build_writer_user,
    select_writer_system,
)

__all__ = [
    "ANALYZER_SYSTEM",
    "build_analyzer_user",
    "canonical_facts_to_dict",
    "render_canonical_facts_block",
    "render_vacancy_block",
    "OPENER_TEMPLATES",
    "select_openers",
    "WRITER_SYSTEM",
    "WRITER_SYSTEM_STANDARD",
    "WRITER_SYSTEM_UNIVERSAL",
    "build_writer_user",
    "select_writer_system",
    "VALIDATOR_SYSTEM",
    "build_validator_user",
]
