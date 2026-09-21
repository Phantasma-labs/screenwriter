# src/formatters/t2i.py
from __future__ import annotations


def t2i_box(label: str, prompt: str) -> str:
    return f"**{label}:**\n```text\n{prompt}\n```\n"
