# tests/test_skills.py
from __future__ import annotations

import pytest

from src.skills import UnknownSkillError, available_skills, get_skill
from src.skills.base import MASTER_SCENE_RULES, T2I_PROMPT_GUIDELINES, OutputFormat, SkillProfile


def test_master_scene_rules_mentions_scene_headings():
    assert "INT." in MASTER_SCENE_RULES
    assert "EXT." in MASTER_SCENE_RULES


def test_t2i_guidelines_mentions_nano_banana():
    assert "Nano Banana" in T2I_PROMPT_GUIDELINES


def test_short_film_is_fountain_output():
    skill = get_skill("short_film")
    assert isinstance(skill, SkillProfile)
    assert skill.output_format == OutputFormat.FOUNTAIN
    assert len(skill.outline_template) >= 4
    assert len(skill.review_criteria) >= 3


def test_short_film_system_prompt_includes_master_scene_rules():
    prompt = get_skill("short_film").system_prompt()
    assert "INT." in prompt


def test_short_film_review_system_prompt_includes_all_criteria():
    skill = get_skill("short_film")
    review_prompt = skill.review_system_prompt()
    for criterion in skill.review_criteria:
        assert criterion in review_prompt


def test_short_film_outline_prompt_includes_every_beat():
    skill = get_skill("short_film")
    prompt = skill.outline_prompt()
    for beat in skill.outline_template:
        assert beat in prompt


# --- append to tests/test_skills.py ---
ALL_SKILLS = [
    "short_film",
    "documentary",
    "commercial",
    "product_shot",
    "learning_dev",
    "infomedia",
]


def test_available_skills_lists_all_six():
    assert available_skills() == sorted(ALL_SKILLS)


@pytest.mark.parametrize("name", ALL_SKILLS)
def test_get_skill_returns_complete_profile(name):
    skill = get_skill(name)
    assert skill.name == name
    assert skill.persona
    assert len(skill.outline_template) >= 4
    assert len(skill.review_criteria) >= 3


@pytest.mark.parametrize(
    "name", ["documentary", "commercial", "product_shot", "learning_dev", "infomedia"]
)
def test_production_skills_are_dual_column(name):
    assert get_skill(name).output_format == OutputFormat.DUAL_COLUMN


def test_get_skill_unknown_raises():
    with pytest.raises(UnknownSkillError):
        get_skill("not_a_real_skill")
