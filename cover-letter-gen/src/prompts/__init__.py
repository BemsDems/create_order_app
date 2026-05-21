"""Prompt strings for the 3-pass pipeline.

Kept in plain Python modules (not Jinja) — the small amount of templating
we need is simple string substitution on a few well-defined slots.
"""

from .analyzer import ANALYZER_SYSTEM, build_analyzer_user
from .writer import WRITER_SYSTEM, build_writer_user
from .validator import VALIDATOR_SYSTEM, build_validator_user

__all__ = [
    "ANALYZER_SYSTEM",
    "build_analyzer_user",
    "WRITER_SYSTEM",
    "build_writer_user",
    "VALIDATOR_SYSTEM",
    "build_validator_user",
]
