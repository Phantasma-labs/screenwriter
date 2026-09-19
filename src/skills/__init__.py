# src/skills/__init__.py
from __future__ import annotations

from src.skills.base import SkillProfile
from src.skills.short_film import SHORT_FILM


class UnknownSkillError(KeyError):
    """Raised when a skill name isn't in the registry."""


_REGISTRY: dict[str, SkillProfile] = {skill.name: skill for skill in (SHORT_FILM,)}


def get_skill(name: str) -> SkillProfile:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise UnknownSkillError(f"Unknown skill '{name}'. Available: {sorted(_REGISTRY)}") from exc


def available_skills() -> list[str]:
    return sorted(_REGISTRY)
