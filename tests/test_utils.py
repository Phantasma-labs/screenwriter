# tests/test_utils.py
from __future__ import annotations

from pydantic import BaseModel

from src.utils import extract_json_array, extract_json_object


class _Widget(BaseModel):
    name: str
    count: int


def test_extract_json_object_valid():
    raw = 'Sure, here you go: {"name": "gizmo", "count": 3} thanks'
    assert extract_json_object(raw, _Widget) == _Widget(name="gizmo", count=3)


def test_extract_json_object_no_json_returns_none():
    assert extract_json_object("no json here", _Widget) is None


def test_extract_json_object_invalid_schema_returns_none():
    assert extract_json_object('{"name": "gizmo"}', _Widget) is None


def test_extract_json_object_malformed_json_returns_none():
    assert extract_json_object('{"name": "gizmo", "count": }', _Widget) is None


def test_extract_json_array_valid_skips_bad_entries():
    raw = '[{"name": "a", "count": 1}, {"name": "b"}, {"name": "c", "count": 3}]'
    result = extract_json_array(raw, _Widget)
    assert result == [_Widget(name="a", count=1), _Widget(name="c", count=3)]


def test_extract_json_array_no_array_returns_empty():
    assert extract_json_array("nothing here", _Widget) == []


def test_extract_json_array_not_a_list_returns_empty():
    assert extract_json_array('{"name": "a", "count": 1}', _Widget) == []
