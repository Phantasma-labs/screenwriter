# src/utils.py
from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def extract_json_object(raw: str, model: type[ModelT]) -> ModelT | None:
    """Finds the first {...} block in raw text and validates it against model.
    Returns None if no valid block is found - callers must supply a fallback."""
    match = _OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return model.model_validate(data)
    except ValidationError:
        return None


def extract_json_array(raw: str, item_model: type[ModelT]) -> list[ModelT]:
    """Finds the first [...] block in raw text and validates each entry
    against item_model, skipping entries that fail validation."""
    match = _ARRAY_RE.search(raw)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    items: list[ModelT] = []
    for entry in data:
        try:
            items.append(item_model.model_validate(entry))
        except ValidationError:
            continue
    return items
