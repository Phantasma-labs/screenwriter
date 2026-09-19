# src/skills/__init__.py
from __future__ import annotations

from src.skills.base import SkillProfile
from src.skills.commercial import COMMERCIAL
from src.skills.documentary import DOCUMENTARY
from src.skills.infomedia import INFOMEDIA
from src.skills.learning_dev import LEARNING_DEV
from src.skills.product_shot import PRODUCT_SHOT
from src.skills.short_film import SHORT_FILM


class UnknownSkillError(KeyError):
    """Raised when a skill name isn't in the registry."""


_ALL_SKILLS = (
    SHORT_FILM,
    DOCUMENTARY,
    COMMERCIAL,
    PRODUCT_SHOT,
    LEARNING_DEV,
    INFOMEDIA,
)

_REGISTRY: dict[str, SkillProfile] = {skill.name: skill for skill in _ALL_SKILLS}


def get_skill(name: str) -> SkillProfile:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise UnknownSkillError(f"Unknown skill '{name}'. Available: {sorted(_REGISTRY)}") from exc


def available_skills() -> list[str]:
    return sorted(_REGISTRY)
