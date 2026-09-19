# Screenwriter Agent CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Screenwriter Agent CLI — a LangGraph-orchestrated Python CLI that turns a topic plus optional `.md`/`.txt`/`.pdf` context into a reviewed screenplay (Fountain + human-readable Markdown), a character bible, and a location bible, each with copy-paste Nano Banana Pro T2I prompts.

**Architecture:** A `StateGraph` pipeline (`ingest → interview loop → researcher → outliner → writer ⇄ reviewer → character_bible → location_bible ⇄ bible_reviewer → finalize`) built from small, dependency-injected node factories, backed by an ephemeral Chroma RAG store for long context and Ollama's cloud API for all LLM/embedding calls. `main.py` drives the compiled graph via `.stream()`, handling LangGraph `interrupt()`/`Command(resume=...)` for the terminal Q&A interview, and writes four output files per run.

**Tech Stack:** Python 3.10+, LangGraph (`StateGraph`, `interrupt`, `InMemorySaver`), `langchain-core`, `langchain-ollama` (Ollama cloud API only), `chromadb` (ephemeral, in-memory), `tavily-python` → `ddgs` fallback, `pypdf`, `python-dotenv`, Pydantic v2, `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-09-19-screenwriter-cli-design.md` (read this alongside the plan — §12 "Implementation addenda" documents API details, e.g. `InMemorySaver`, the two-node interviewer split, and the `AVBeat` structured-draft representation, that this plan implements directly).

## Global Constraints

- Python 3.10+; every source file starts with `from __future__ import annotations`.
- Import order in every file: standard library, then third-party, then internal (`src.*`).
- Strict type annotations on every function signature and class attribute. No bare `Any` or unparameterized `dict`/`list` — **the one sanctioned exception** is `NodeUpdate = dict[str, Any]` in `src/graph/state.py`, used only as the LangGraph node-return type (LangGraph's functional-update pattern genuinely returns heterogeneous dicts).
- `ScreenplayState` is a `TypedDict`; all structured LLM outputs (`ReviewFeedback`, `AVBeat`, `CharacterBibleEntry`, `LocationBibleEntry`, `InterviewDecision`) are Pydantic v2 `BaseModel`s.
- Ruff line length 100, target Python 3.10.
- 100% offline tests: never call `.invoke()`/`.embed_query()`/`.embed_documents()` on a real network-backed client, never call the real Tavily/DDGS/Chroma-over-network APIs. Every LLM-calling node factory accepts an injectable `llm: BaseChatModel | None` (default `get_llm(...)`); every embeddings-consuming path accepts an injectable embeddings object; `tools/search.py` takes injectable `tavily_fn`/`ddgs_fn`. Instantiating a real `ChatOllama`/`OllamaEmbeddings` is safe in tests (no network touched by construction) — only `.invoke()`/`.embed_*()` are forbidden without a fake.
- Every revision-style loop (interview, screenplay revise, bible revise) has an explicit terminal cap already threaded through state/settings — never add an uncapped loop.
- Ollama's **cloud API only** (`OLLAMA_BASE_URL` default `https://ollama.com`, `OLLAMA_API_KEY` required at the point a real client is constructed) — no local daemon assumption, no OpenAI, no image ingestion.
- All six skills render to strict Fountain (`<stem>.fountain`); the five Dual-Column skills additionally render a Markdown A/V table (`<stem>.md`). T2I prompts target Nano Banana Pro per `T2I_PROMPT_GUIDELINES` in `src/skills/base.py`.
- Every run writes exactly four files when `--output` is given: `<stem>.fountain`, `<stem>.md`, `<stem>.characters.md`, `<stem>.locations.md`; without `--output`, all four are printed to stdout.

## File Structure

```text
screenwriter/                        (= L:\AIStudio\ScreenWriter, project root)
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py            # Settings, load_settings, get_llm, get_embeddings (Ollama cloud only)
│   ├── utils.py              # extract_json_object / extract_json_array (shared LLM-JSON parsing)
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py           # SourceDocument, ParsedContext, ParserError, parse_context_files
│   │   ├── text_parser.py    # parse_text_file (.md/.txt)
│   │   ├── pdf_parser.py     # parse_pdf_file (.pdf via pypdf)
│   │   └── chunker.py        # Chunk, chunk_text, chunk_parsed_context
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── store.py          # EphemeralVectorStore, register_store/get_store/clear_store
│   │   └── retriever.py      # retrieve_context(run_id, query, k)
│   ├── skills/
│   │   ├── __init__.py       # registry: get_skill, available_skills, UnknownSkillError
│   │   ├── base.py           # SkillProfile, OutputFormat, MASTER_SCENE_RULES, T2I_PROMPT_GUIDELINES
│   │   ├── short_film.py
│   │   ├── documentary.py
│   │   ├── commercial.py
│   │   ├── product_shot.py
│   │   ├── learning_dev.py
│   │   └── infomedia.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search.py         # search() - Tavily -> DDGS fallback, DI'd fn params
│   ├── formatters/
│   │   ├── __init__.py
│   │   ├── dual_column.py    # AVBeat, parse_av_beats, render_dual_column_table
│   │   ├── fountain.py       # render_fountain, validate_fountain, render_dual_column_as_fountain
│   │   └── bible.py          # CharacterBibleEntry/LocationBibleEntry, parse_*, render_*
│   └── graph/
│       ├── __init__.py
│       ├── state.py          # ScreenplayState, InterviewTurn, NodeUpdate, new_initial_state
│       ├── edges.py           # route_after_interview_ask/wait, route_after_reviewer/bible_reviewer
│       ├── workflow.py        # build_workflow(...)
│       └── nodes/
│           ├── __init__.py
│           ├── ingest.py
│           ├── interviewer.py       # make_interview_ask_node, interview_wait_node
│           ├── researcher.py
│           ├── outliner.py
│           ├── writer.py
│           ├── review_shared.py     # ReviewFeedback, FALLBACK_REVIEW_FEEDBACK, format_critique
│           ├── reviewer.py
│           ├── character_bible.py
│           ├── location_bible.py
│           ├── bible_reviewer.py
│           └── finalize.py
└── tests/
    ├── __init__.py
    ├── conftest.py           # FakeChatModel, autouse Ollama env fixture
    ├── test_config.py
    ├── test_utils.py
    ├── test_parsers.py
    ├── test_chunker.py
    ├── test_rag.py
    ├── test_search_fallback.py
    ├── test_skills.py
    ├── test_formatters.py
    ├── test_graph_state.py
    ├── test_graph_nodes.py
    ├── test_interview_node.py
    ├── test_graph_edges.py
    ├── test_workflow.py
    ├── test_main.py
    └── test_workflow_e2e.py
```

---

### Task 1: Project Scaffolding & Git Init

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`
- Create: `src/__init__.py`, `src/parsers/__init__.py`, `src/rag/__init__.py`, `src/skills/__init__.py`, `src/tools/__init__.py`, `src/formatters/__init__.py`, `src/graph/__init__.py`, `src/graph/nodes/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces: an installable, ruff/pytest-configured empty package tree every later task adds real modules into.

- [ ] **Step 1: Create the package skeleton**

All `__init__.py` files are empty (0 bytes) — they exist only to make each directory an importable package:

```bash
mkdir -p src/parsers src/rag src/skills src/tools src/formatters src/graph/nodes tests
touch src/__init__.py src/parsers/__init__.py src/rag/__init__.py src/skills/__init__.py \
      src/tools/__init__.py src/formatters/__init__.py src/graph/__init__.py \
      src/graph/nodes/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "screenwriter"
version = "0.1.0"
description = "LangGraph-orchestrated screenplay writer/reviewer with ephemeral RAG context and Nano Banana Pro T2I production bibles."
requires-python = ">=3.10"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `requirements.txt`**

```text
langgraph>=0.6
langchain-core>=0.3
langchain-ollama>=0.3
chromadb>=0.5
tavily-python>=0.5
ddgs>=1.0
pypdf>=4.0
python-dotenv>=1.0
pydantic>=2.6
pytest>=8.0
pytest-cov>=5.0
ruff>=0.6
```

- [ ] **Step 4: Write `.env.example`**

```text
OLLAMA_MODEL_NAME=deepseek-v4.1-flash:cloud
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=
OLLAMA_EMBED_MODEL=nomic-embed-text
TAVILY_API_KEY=
```

- [ ] **Step 5: Write `.gitignore`**

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env
*.egg-info/
htmlcov/
.coverage
```

- [ ] **Step 6: Install dependencies and verify the environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import langgraph, langchain_core, langchain_ollama, chromadb, tavily, ddgs, pypdf, dotenv, pydantic"
```

Expected: no import errors.

- [ ] **Step 7: Init git and commit**

```bash
git init
git add pyproject.toml requirements.txt .env.example .gitignore src tests
git commit -m "chore: scaffold screenwriter CLI project structure"
```

---

### Task 2: Config Layer (`src/config.py`)

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`
- Modify: `tests/conftest.py` (create — autouse Ollama env fixture used by every later test file)

**Interfaces:**
- Produces: `class Settings` (frozen dataclass, fields: `ollama_model_name: str`, `ollama_base_url: str`, `ollama_api_key: str`, `ollama_embed_model: str`, `tavily_api_key: str | None`, `rag_chunk_size: int`, `rag_chunk_overlap: int`, `rag_top_k: int`, `rag_min_chars_to_index: int`, `max_interview_questions: int`, `review_pass_score: float`, `max_bible_revisions: int`); `class ConfigError(RuntimeError)`; `load_settings() -> Settings`; `get_llm(settings: Settings | None = None, temperature: float = 0.7) -> ChatOllama`; `get_embeddings(settings: Settings | None = None) -> OllamaEmbeddings`.

- [ ] **Step 1: Verify the installed `langchain-ollama` accepts `client_kwargs`**

This confirms the exact constructor field before we write code against it (the spec flagged this as the one detail not pinned by first-party doc snippets):

```bash
python -c "from langchain_ollama import ChatOllama; print(sorted(ChatOllama.model_fields.keys()))"
```

Expected: `client_kwargs` appears in the printed field list. If it doesn't, inspect the printed list for the actual field name used to pass extra options to the underlying `ollama.Client` (e.g. `headers`, `sync_client_kwargs`) and use that name instead of `client_kwargs` in Step 3 below and in `get_embeddings`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_config.py
from __future__ import annotations

import pytest

from src.config import ConfigError, get_embeddings, get_llm, load_settings


def test_load_settings_uses_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL_NAME", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    settings = load_settings()
    assert settings.ollama_model_name == "deepseek-v4.1-flash:cloud"
    assert settings.ollama_base_url == "https://ollama.com"
    assert settings.ollama_api_key == ""
    assert settings.rag_chunk_size == 1000
    assert settings.rag_chunk_overlap == 150
    assert settings.max_interview_questions == 5
    assert settings.review_pass_score == 8.0


def test_load_settings_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "custom-model:cloud")
    monkeypatch.setenv("RAG_TOP_K", "9")
    settings = load_settings()
    assert settings.ollama_model_name == "custom-model:cloud"
    assert settings.rag_top_k == 9


def test_get_llm_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_llm()


def test_get_llm_builds_chat_ollama_with_configured_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    llm = get_llm()
    assert llm.model == "deepseek-v4.1-flash:cloud"
    assert llm.base_url == "https://ollama.com"


def test_get_embeddings_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        get_embeddings()


def test_get_embeddings_builds_with_configured_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embeddings = get_embeddings()
    assert embeddings.model == "nomic-embed-text"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 4: Implement `src/config.py`**

```python
# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing at the point it's needed."""


@dataclass(frozen=True)
class Settings:
    ollama_model_name: str
    ollama_base_url: str
    ollama_api_key: str
    ollama_embed_model: str
    tavily_api_key: str | None
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_top_k: int
    rag_min_chars_to_index: int
    max_interview_questions: int
    review_pass_score: float
    max_bible_revisions: int


def load_settings() -> Settings:
    return Settings(
        ollama_model_name=os.environ.get("OLLAMA_MODEL_NAME", "deepseek-v4.1-flash:cloud"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "https://ollama.com"),
        ollama_api_key=os.environ.get("OLLAMA_API_KEY", ""),
        ollama_embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY") or None,
        rag_chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", "1000")),
        rag_chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "150")),
        rag_top_k=int(os.environ.get("RAG_TOP_K", "5")),
        rag_min_chars_to_index=int(os.environ.get("RAG_MIN_CHARS_TO_INDEX", "4000")),
        max_interview_questions=int(os.environ.get("MAX_INTERVIEW_QUESTIONS", "5")),
        review_pass_score=float(os.environ.get("REVIEW_PASS_SCORE", "8.0")),
        max_bible_revisions=int(os.environ.get("MAX_BIBLE_REVISIONS", "1")),
    )


def _require_api_key(settings: Settings) -> str:
    if not settings.ollama_api_key:
        raise ConfigError(
            "OLLAMA_API_KEY is required to call Ollama's cloud API. Set it in your .env file."
        )
    return settings.ollama_api_key


def get_llm(settings: Settings | None = None, temperature: float = 0.7) -> ChatOllama:
    settings = settings or load_settings()
    api_key = _require_api_key(settings)
    return ChatOllama(
        model=settings.ollama_model_name,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    )


def get_embeddings(settings: Settings | None = None) -> OllamaEmbeddings:
    settings = settings or load_settings()
    api_key = _require_api_key(settings)
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    )
```

- [ ] **Step 5: Create `tests/conftest.py` with the autouse Ollama env fixture**

Every later test needs a non-empty `OLLAMA_API_KEY` so `get_llm()`/`get_embeddings()` can construct real (never-invoked) clients without raising `ConfigError`; this fixture sets sane defaults once, and individual tests (like the two `_raises_without_api_key` tests above) explicitly `monkeypatch.delenv` to override it:

```python
# tests/conftest.py
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_MODEL_NAME", "deepseek-v4.1-flash:cloud")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add src/config.py tests/test_config.py tests/conftest.py
git commit -m "feat: add Ollama-cloud-only config layer"
```

---

### Task 3: Shared JSON Utilities (`src/utils.py`)

**Files:**
- Create: `src/utils.py`
- Test: `tests/test_utils.py`

**Interfaces:**
- Consumes: nothing internal (only `pydantic.BaseModel`).
- Produces: `extract_json_object(raw: str, model: type[ModelT]) -> ModelT | None`; `extract_json_array(raw: str, item_model: type[ModelT]) -> list[ModelT]`. Reused by every reviewer-style node and by `formatters/dual_column.py`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils'`

- [ ] **Step 3: Implement `src/utils.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_utils.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: add shared robust JSON extraction for LLM outputs"
```

---

### Task 4: Parsers - base.py + text_parser.py

**Files:**
- Create: `src/parsers/base.py`, `src/parsers/text_parser.py`
- Test: `tests/test_parsers.py`

**Interfaces:**
- Produces: `SUPPORTED_EXTENSIONS: set[str]`; `class ParserError(RuntimeError)`; `@dataclass SourceDocument {file_path: str, text: str, warnings: list[str]}`; `@dataclass ParsedContext {documents: list[SourceDocument], combined_text: str, warnings: list[str]}`; `parse_context_files(file_paths: list[str]) -> ParsedContext`; `parse_text_file(file_path: str) -> SourceDocument`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parsers.py
from __future__ import annotations

import pytest

from src.parsers.base import ParserError, parse_context_files
from src.parsers.text_parser import parse_text_file


def test_parse_text_file_reads_markdown(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Heading\n\nSome body text.", encoding="utf-8")
    doc = parse_text_file(str(md_file))
    assert doc.text == "# Heading\n\nSome body text."
    assert doc.warnings == []


def test_parse_text_file_missing_file_returns_warning():
    doc = parse_text_file("does/not/exist.md")
    assert doc.text == ""
    assert len(doc.warnings) == 1
    assert "Could not read" in doc.warnings[0]


def test_parse_context_files_combines_multiple_sources(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("Content A", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("Content B", encoding="utf-8")
    context = parse_context_files([str(a), str(b)])
    assert "Content A" in context.combined_text
    assert "Content B" in context.combined_text
    assert len(context.documents) == 2
    assert context.warnings == []


def test_parse_context_files_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "image.png"
    bad.write_text("not text")
    with pytest.raises(ParserError):
        parse_context_files([str(bad)])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_parsers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parsers.base'`

- [ ] **Step 3: Implement `src/parsers/base.py`**

```python
# src/parsers/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


class ParserError(RuntimeError):
    """Raised for unsupported file extensions."""


@dataclass
class SourceDocument:
    file_path: str
    text: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedContext:
    documents: list[SourceDocument]
    combined_text: str
    warnings: list[str]


def parse_context_files(file_paths: list[str]) -> ParsedContext:
    from src.parsers.pdf_parser import parse_pdf_file
    from src.parsers.text_parser import parse_text_file

    documents: list[SourceDocument] = []
    warnings: list[str] = []
    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ParserError(
                f"Unsupported file type '{ext}' for {file_path}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        doc = parse_pdf_file(file_path) if ext == ".pdf" else parse_text_file(file_path)
        documents.append(doc)
        warnings.extend(doc.warnings)

    combined_text = "\n\n".join(
        f"### Source: {doc.file_path}\n{doc.text}" for doc in documents if doc.text
    )
    return ParsedContext(documents=documents, combined_text=combined_text, warnings=warnings)
```

- [ ] **Step 4: Implement `src/parsers/text_parser.py`**

```python
# src/parsers/text_parser.py
from __future__ import annotations

from pathlib import Path

from src.parsers.base import SourceDocument


def parse_text_file(file_path: str) -> SourceDocument:
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return SourceDocument(
            file_path=file_path, text="", warnings=[f"Could not read {file_path}: {exc}"]
        )
    return SourceDocument(file_path=file_path, text=text)
```

- [ ] **Step 5: Create a stub `src/parsers/pdf_parser.py` so `base.py`'s lazy import resolves**

Task 5 replaces this with the real implementation; for now it only needs to exist and be importable so `parse_context_files` doesn't `ImportError` when it takes the `.pdf` branch (untested until Task 5, so this stub is never exercised by Task 4's tests):

```python
# src/parsers/pdf_parser.py
from __future__ import annotations

from src.parsers.base import SourceDocument


def parse_pdf_file(file_path: str) -> SourceDocument:
    raise NotImplementedError("Implemented in Task 5")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_parsers.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add src/parsers/base.py src/parsers/text_parser.py src/parsers/pdf_parser.py tests/test_parsers.py
git commit -m "feat: add text/markdown context parsing and dispatcher"
```

---

### Task 5: Parsers - pdf_parser.py

**Files:**
- Modify: `src/parsers/pdf_parser.py` (replace the Task 4 stub)
- Modify: `tests/test_parsers.py` (append PDF tests)

**Interfaces:**
- Consumes: `SourceDocument` from `src.parsers.base`.
- Produces: `parse_pdf_file(file_path: str) -> SourceDocument`, now fully implemented.

- [ ] **Step 1: Append the failing tests to `tests/test_parsers.py`**

```python
# --- append to tests/test_parsers.py ---
from unittest.mock import MagicMock, patch

from src.parsers.pdf_parser import parse_pdf_file


def _make_fake_page(text: str) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def test_parse_pdf_file_extracts_pages():
    fake_reader = MagicMock()
    fake_reader.pages = [_make_fake_page("Page one text"), _make_fake_page("Page two text")]
    with patch("src.parsers.pdf_parser.pypdf.PdfReader", return_value=fake_reader):
        doc = parse_pdf_file("fake.pdf")
    assert "[Page 1]" in doc.text
    assert "Page one text" in doc.text
    assert "[Page 2]" in doc.text
    assert "Page two text" in doc.text
    assert doc.warnings == []


def test_parse_pdf_file_handles_open_failure():
    with patch("src.parsers.pdf_parser.pypdf.PdfReader", side_effect=OSError("corrupt file")):
        doc = parse_pdf_file("broken.pdf")
    assert doc.text == ""
    assert len(doc.warnings) == 1
    assert "Could not open PDF" in doc.warnings[0]


def test_parse_pdf_file_handles_page_extraction_failure():
    bad_page = MagicMock()
    bad_page.extract_text.side_effect = RuntimeError("bad page")
    fake_reader = MagicMock()
    fake_reader.pages = [bad_page]
    with patch("src.parsers.pdf_parser.pypdf.PdfReader", return_value=fake_reader):
        doc = parse_pdf_file("fake.pdf")
    assert doc.text == ""
    assert "Failed to extract page 1" in doc.warnings[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_parsers.py -k pdf -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement `src/parsers/pdf_parser.py`**

```python
# src/parsers/pdf_parser.py
from __future__ import annotations

import pypdf

from src.parsers.base import SourceDocument


def parse_pdf_file(file_path: str) -> SourceDocument:
    try:
        reader = pypdf.PdfReader(file_path)
    except Exception as exc:  # noqa: BLE001 - any pypdf failure must not crash ingestion
        return SourceDocument(
            file_path=file_path, text="", warnings=[f"Could not open PDF {file_path}: {exc}"]
        )

    page_texts: list[str] = []
    warnings: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            page_texts.append(f"[Page {index + 1}]\n{page.extract_text() or ''}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to extract page {index + 1} of {file_path}: {exc}")
    return SourceDocument(file_path=file_path, text="\n\n".join(page_texts), warnings=warnings)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_parsers.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/parsers/pdf_parser.py tests/test_parsers.py
git commit -m "feat: implement PDF context parsing via pypdf"
```

---

### Task 6: Chunker (`src/parsers/chunker.py`)

**Files:**
- Create: `src/parsers/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `ParsedContext`, `SourceDocument` from `src.parsers.base`.
- Produces: `@dataclass Chunk {text: str, source_file: str, chunk_index: int}`; `chunk_text(text: str, source_file: str, chunk_size: int, overlap: int) -> list[Chunk]`; `chunk_parsed_context(context: ParsedContext, chunk_size: int, overlap: int) -> list[Chunk]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunker.py
from __future__ import annotations

import pytest

from src.parsers.base import ParsedContext, SourceDocument
from src.parsers.chunker import chunk_parsed_context, chunk_text


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("", "a.md", chunk_size=10, overlap=2) == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    chunks = chunk_text("short text", "a.md", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].chunk_index == 0


def test_chunk_text_splits_with_overlap():
    text = "a" * 25
    chunks = chunk_text(text, "a.md", chunk_size=10, overlap=3)
    assert [c.text for c in chunks] == ["a" * 10, "a" * 10, "a" * 10, "a" * 4]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert all(c.source_file == "a.md" for c in chunks)


def test_chunk_text_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", "a.md", chunk_size=5, overlap=5)


def test_chunk_parsed_context_chunks_every_document():
    context = ParsedContext(
        documents=[
            SourceDocument(file_path="a.md", text="a" * 15),
            SourceDocument(file_path="b.md", text="b" * 5),
        ],
        combined_text="",
        warnings=[],
    )
    chunks = chunk_parsed_context(context, chunk_size=10, overlap=2)
    sources = {c.source_file for c in chunks}
    assert sources == {"a.md", "b.md"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.parsers.chunker'`

- [ ] **Step 3: Implement `src/parsers/chunker.py`**

```python
# src/parsers/chunker.py
from __future__ import annotations

from dataclasses import dataclass

from src.parsers.base import ParsedContext


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int


def chunk_text(text: str, source_file: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(Chunk(text=text[start:end], source_file=source_file, chunk_index=index))
        if end == length:
            break
        start = end - overlap
        index += 1
    return chunks


def chunk_parsed_context(context: ParsedContext, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in context.documents:
        chunks.extend(chunk_text(doc.text, doc.file_path, chunk_size, overlap))
    return chunks
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/parsers/chunker.py tests/test_chunker.py
git commit -m "feat: add overlapping-window chunker for RAG indexing"
```

---

### Task 7: RAG - store.py + retriever.py

**Files:**
- Create: `src/rag/store.py`, `src/rag/retriever.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `Chunk` from `src.parsers.chunker`.
- Produces: `class EphemeralVectorStore` (`__init__(embeddings, collection_name=None)`, `add_chunks(chunks: list[Chunk]) -> None`, `similarity_search(query: str, k: int) -> list[str]`); `register_store(run_id: str, store: EphemeralVectorStore) -> None`; `get_store(run_id: str) -> EphemeralVectorStore | None`; `clear_store(run_id: str) -> None`; `retrieve_context(run_id: str, query: str, k: int) -> list[str]`. Consumed by `graph/nodes/ingest.py`, `outliner.py`, `writer.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rag.py
from __future__ import annotations

from src.parsers.chunker import Chunk
from src.rag.retriever import retrieve_context
from src.rag.store import EphemeralVectorStore, clear_store, get_store, register_store


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vowels = sum(1 for c in text.lower() if c in "aeiou")
        first = ord(text[0]) if text else 0.0
        return [float(len(text)), float(vowels), float(first)]


def test_add_chunks_and_similarity_search_returns_closest_text():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-1")
    chunks = [
        Chunk(text="The quick brown fox", source_file="a.md", chunk_index=0),
        Chunk(text="Quantum physics lecture notes", source_file="b.md", chunk_index=0),
    ]
    store.add_chunks(chunks)
    results = store.similarity_search("The quick brown fox", k=1)
    assert results == ["The quick brown fox"]


def test_similarity_search_empty_collection_returns_empty_list():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-2")
    assert store.similarity_search("anything", k=3) == []


def test_add_chunks_noop_on_empty_list():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-3")
    store.add_chunks([])
    assert store.similarity_search("x", k=1) == []


def test_register_and_get_store_roundtrip():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-4")
    register_store("run-123", store)
    try:
        assert get_store("run-123") is store
    finally:
        clear_store("run-123")
    assert get_store("run-123") is None


def test_get_store_unknown_run_id_returns_none():
    assert get_store("no-such-run") is None


def test_retrieve_context_no_run_id_returns_empty():
    assert retrieve_context("", "query", 3) == []


def test_retrieve_context_unknown_run_id_returns_empty():
    assert retrieve_context("missing-run", "query", 3) == []


def test_retrieve_context_delegates_to_store():
    store = EphemeralVectorStore(embeddings=FakeEmbeddings(), collection_name="test-collection-5")
    store.add_chunks([Chunk(text="Hello world", source_file="a.md", chunk_index=0)])
    register_store("run-abc", store)
    try:
        assert retrieve_context("run-abc", "Hello world", 1) == ["Hello world"]
    finally:
        clear_store("run-abc")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.store'`

- [ ] **Step 3: Implement `src/rag/store.py`**

```python
# src/rag/store.py
from __future__ import annotations

import uuid
from typing import Protocol

import chromadb

from src.parsers.chunker import Chunk


class EmbeddingsFn(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class EphemeralVectorStore:
    def __init__(self, embeddings: EmbeddingsFn, collection_name: str | None = None) -> None:
        self._client = chromadb.EphemeralClient()
        self._embeddings = embeddings
        self._collection = self._client.create_collection(
            name=collection_name or f"run-{uuid.uuid4().hex}"
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._embeddings.embed_documents([c.text for c in chunks])
        self._collection.add(
            ids=[f"{c.source_file}:{c.chunk_index}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source_file": c.source_file, "chunk_index": c.chunk_index} for c in chunks
            ],
        )

    def similarity_search(self, query: str, k: int) -> list[str]:
        count = self._collection.count()
        if count == 0:
            return []
        query_vector = self._embeddings.embed_query(query)
        results = self._collection.query(query_embeddings=[query_vector], n_results=min(k, count))
        documents = results.get("documents") or [[]]
        return documents[0]


_ACTIVE_STORES: dict[str, EphemeralVectorStore] = {}


def register_store(run_id: str, store: EphemeralVectorStore) -> None:
    _ACTIVE_STORES[run_id] = store


def get_store(run_id: str) -> EphemeralVectorStore | None:
    return _ACTIVE_STORES.get(run_id)


def clear_store(run_id: str) -> None:
    _ACTIVE_STORES.pop(run_id, None)
```

- [ ] **Step 4: Implement `src/rag/retriever.py`**

```python
# src/rag/retriever.py
from __future__ import annotations

from src.rag.store import get_store


def retrieve_context(run_id: str, query: str, k: int) -> list[str]:
    if not run_id:
        return []
    store = get_store(run_id)
    if store is None:
        return []
    return store.similarity_search(query, k)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_rag.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add src/rag/store.py src/rag/retriever.py tests/test_rag.py
git commit -m "feat: add ephemeral Chroma RAG store and retriever"
```

---

### Task 8: Search Tool (`src/tools/search.py`)

**Files:**
- Create: `src/tools/search.py`
- Test: `tests/test_search_fallback.py`

**Interfaces:**
- Consumes: `Settings` from `src.config`.
- Produces: `class SearchResult(TypedDict) {title: str, url: str, snippet: str}`; `search(query: str, settings: Settings | None = None, tavily_fn=..., ddgs_fn=...) -> list[SearchResult]`. Consumed by `graph/nodes/researcher.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_search_fallback.py
from __future__ import annotations

from src.config import Settings
from src.tools.search import SearchResult, search


def _settings(tavily_api_key: str | None = None) -> Settings:
    return Settings(
        ollama_model_name="m", ollama_base_url="https://ollama.com", ollama_api_key="k",
        ollama_embed_model="e", tavily_api_key=tavily_api_key,
        rag_chunk_size=1000, rag_chunk_overlap=150, rag_top_k=5, rag_min_chars_to_index=4000,
        max_interview_questions=5, review_pass_score=8.0, max_bible_revisions=1,
    )


def test_search_uses_tavily_when_key_present():
    calls = []

    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        calls.append((query, api_key))
        return [SearchResult(title="T", url="u", snippet="s")]

    def fake_ddgs(query: str) -> list[SearchResult]:
        raise AssertionError("DDGS should not be called when Tavily succeeds")

    results = search(
        "test query", settings=_settings("tavily-key"), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == [SearchResult(title="T", url="u", snippet="s")]
    assert calls == [("test query", "tavily-key")]


def test_search_falls_back_to_ddgs_when_no_tavily_key():
    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        raise AssertionError("Tavily should not be called without a key")

    def fake_ddgs(query: str) -> list[SearchResult]:
        return [SearchResult(title="D", url="u2", snippet="s2")]

    results = search(
        "test query", settings=_settings(None), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == [SearchResult(title="D", url="u2", snippet="s2")]


def test_search_falls_back_to_ddgs_when_tavily_raises():
    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        raise RuntimeError("network down")

    def fake_ddgs(query: str) -> list[SearchResult]:
        return [SearchResult(title="D", url="u2", snippet="s2")]

    results = search(
        "test query", settings=_settings("tavily-key"), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == [SearchResult(title="D", url="u2", snippet="s2")]


def test_search_returns_empty_when_both_fail():
    def fake_tavily(query: str, api_key: str) -> list[SearchResult]:
        raise RuntimeError("network down")

    def fake_ddgs(query: str) -> list[SearchResult]:
        raise RuntimeError("ddgs also down")

    results = search(
        "test query", settings=_settings("tavily-key"), tavily_fn=fake_tavily, ddgs_fn=fake_ddgs
    )
    assert results == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_search_fallback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.search'`

- [ ] **Step 3: Implement `src/tools/search.py`**

```python
# src/tools/search.py
from __future__ import annotations

import logging
from typing import Callable, TypedDict

from src.config import Settings, load_settings

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


TavilySearchFn = Callable[[str, str], list[SearchResult]]
DDGSSearchFn = Callable[[str], list[SearchResult]]


def _tavily_search(query: str, api_key: str) -> list[SearchResult]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=5)
    return [
        SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
        for r in response.get("results", [])
    ]


def _ddgs_search(query: str) -> list[SearchResult]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=5))
    return [
        SearchResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
        for r in raw_results
    ]


def search(
    query: str,
    settings: Settings | None = None,
    tavily_fn: TavilySearchFn = _tavily_search,
    ddgs_fn: DDGSSearchFn = _ddgs_search,
) -> list[SearchResult]:
    settings = settings or load_settings()
    if settings.tavily_api_key:
        try:
            return tavily_fn(query, settings.tavily_api_key)
        except Exception as exc:  # noqa: BLE001 - any Tavily failure falls back
            logger.warning("Tavily search failed (%s); falling back to DDGS.", exc)
    try:
        return ddgs_fn(query)
    except Exception as exc:  # noqa: BLE001 - both providers failed
        logger.warning("DDGS search failed (%s); returning no results.", exc)
        return []
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_search_fallback.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/search.py tests/test_search_fallback.py
git commit -m "feat: add Tavily-to-DDGS research search fallback"
```

---

### Task 9: Skills - base.py + registry + short_film

**Files:**
- Create: `src/skills/base.py`, `src/skills/short_film.py`, `src/skills/__init__.py` (replace empty stub)
- Test: `tests/test_skills.py`

**Interfaces:**
- Produces: `class OutputFormat(str, Enum) {FOUNTAIN, DUAL_COLUMN}`; `MASTER_SCENE_RULES: str`; `T2I_PROMPT_GUIDELINES: str`; `@dataclass(frozen=True) SkillProfile {name, display_name, output_format, persona, outline_template: list[str], review_criteria: list[str]}` with methods `system_prompt() -> str`, `review_system_prompt() -> str`, `outline_prompt() -> str`; `class UnknownSkillError(KeyError)`; `get_skill(name: str) -> SkillProfile`; `available_skills() -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_skills.py
from __future__ import annotations

from src.skills import get_skill
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_skills.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.skills.base'`

- [ ] **Step 3: Implement `src/skills/base.py`**

```python
# src/skills/base.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutputFormat(str, Enum):
    FOUNTAIN = "fountain"
    DUAL_COLUMN = "dual_column"


MASTER_SCENE_RULES = """MASTER SCENE FORMAT RULES (Fountain-compatible):
- Scene headings (sluglines) are ALL CAPS: "INT. LOCATION - DAY" / "EXT. LOCATION - NIGHT".
- Character cues are ALL CAPS on their own line directly above dialogue. Use
  extensions in parentheses when needed: (V.O.), (O.S.), (CONT'D).
- Parentheticals are short, lowercase, used sparingly - only when the manner
  of delivery isn't clear from the dialogue or action itself.
- Action/description blocks are written in present tense and kept to 4 lines
  or fewer per paragraph; break longer beats into multiple short paragraphs.
- Transitions are ALL CAPS and end in "TO:" (e.g. CUT TO:, DISSOLVE TO:) or
  are FADE IN: / FADE OUT., used sparingly.
- Show, don't tell: action lines describe only what the camera can see."""

T2I_PROMPT_GUIDELINES = """T2I PROMPT GUIDELINES (target model: Nano Banana Pro):
- Formula: [Subject] + [Action] + [Location/context] + [Composition] + [Style].
- Write full narrative sentences, never a bare keyword list.
- Use positive framing only - describe what IS in frame, never "no X".
- Open with a strong verb: "Create a...", "Generate an image of...".
- Use concrete materials/textures, not generic nouns (e.g. "navy blue tweed
  suit jacket", not "a suit").
- Specify camera, lens, and lighting explicitly (e.g. "shot on a Fujifilm
  camera, shallow depth of field f/1.8, three-point softbox lighting").
- Any on-screen text goes in quotes with font/style specified.
- Consistency without reference images: reuse one fixed canonical physical-
  description clause for a character (face, hair, build, signature
  color/prop) verbatim across all of that character's prompt boxes."""


@dataclass(frozen=True)
class SkillProfile:
    name: str
    display_name: str
    output_format: OutputFormat
    persona: str
    outline_template: list[str]
    review_criteria: list[str]

    def system_prompt(self) -> str:
        return f"{self.persona}\n\n{MASTER_SCENE_RULES}"

    def review_system_prompt(self) -> str:
        checklist = "\n".join(f"- {c}" for c in self.review_criteria)
        return (
            f"You are a professional script doctor reviewing a {self.display_name} "
            f"script written by an AI writer.\n\nEvaluate strictly against this "
            f"checklist:\n{checklist}\n\nBe specific and actionable in your critique. "
            "A script scoring 8.0 or higher out of 10 passes."
        )

    def outline_prompt(self) -> str:
        beats = "\n".join(f"{i + 1}. {beat}" for i, beat in enumerate(self.outline_template))
        return f"Draft a beat sheet / treatment for a {self.display_name} following this structure:\n{beats}"
```

- [ ] **Step 4: Implement `src/skills/short_film.py`**

```python
# src/skills/short_film.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

SHORT_FILM = SkillProfile(
    name="short_film",
    display_name="Short Film",
    output_format=OutputFormat.FOUNTAIN,
    persona=(
        "You are an award-winning short film screenwriter. Write in Master Scene "
        "Format: show, don't tell; every line of dialogue carries subtext; action "
        "lines stay lean and visual."
    ),
    outline_template=[
        "Opening image / hook",
        "Inciting incident",
        "Escalating complication",
        "Turning point",
        "Climax",
        "Resolution / final image",
    ],
    review_criteria=[
        "Scene headings are ALL CAPS and correctly formatted (INT./EXT. LOCATION - TIME)",
        "Action blocks are 4 lines or fewer per paragraph",
        "Dialogue sounds spoken, not expository",
        "Parentheticals are used sparingly and only when meaning isn't clear from context",
        "Subtext: characters rarely state their goal outright",
        "Visual storytelling: the camera could film every action line as written",
    ],
)
```

- [ ] **Step 5: Implement `src/skills/__init__.py` (registry, short_film only for now)**

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_skills.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add src/skills/base.py src/skills/short_film.py src/skills/__init__.py tests/test_skills.py
git commit -m "feat: add skill profile framework and short_film skill"
```

---

### Task 10: Skills - remaining five (documentary, commercial, product_shot, learning_dev, infomedia)

**Files:**
- Create: `src/skills/documentary.py`, `src/skills/commercial.py`, `src/skills/product_shot.py`, `src/skills/learning_dev.py`, `src/skills/infomedia.py`
- Modify: `src/skills/__init__.py` (register all six)
- Modify: `tests/test_skills.py` (append registry-completeness tests)

**Interfaces:**
- Produces: `DOCUMENTARY`, `COMMERCIAL`, `PRODUCT_SHOT`, `LEARNING_DEV`, `INFOMEDIA` (each a `SkillProfile` with `output_format=OutputFormat.DUAL_COLUMN`), registered in `_REGISTRY` alongside `SHORT_FILM`.

- [ ] **Step 1: Append the failing tests to `tests/test_skills.py`**

```python
# --- append to tests/test_skills.py ---
import pytest

from src.skills import UnknownSkillError, available_skills

ALL_SKILLS = ["short_film", "documentary", "commercial", "product_shot", "learning_dev", "infomedia"]


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_skills.py -v`
Expected: FAIL - `test_available_skills_lists_all_six` and the `documentary`/`commercial`/`product_shot`/`learning_dev`/`infomedia` parametrized cases raise `UnknownSkillError`.

- [ ] **Step 3: Implement the five remaining skill files**

```python
# src/skills/documentary.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

DOCUMENTARY = SkillProfile(
    name="documentary",
    display_name="Documentary",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a documentary film writer-producer. You script in Dual-Column "
        "Audio/Visual format: the VISUAL column carries B-roll, archival footage "
        "cues, and interview framing; the AUDIO column carries voice-over (VO), "
        "interview subject dialogue, and ambient sound notes."
    ),
    outline_template=[
        "Cold open / hook",
        "Thesis / question the film explores",
        "Key interview subjects introduced",
        "Archival + B-roll evidence beats",
        "Turning point / complication in the narrative",
        "Resolution and closing VO",
    ],
    review_criteria=[
        "Every row has both a VISUAL and an AUDIO cue - no empty columns",
        "Archival footage is explicitly labeled as archival, not implied",
        "VO reads at a natural spoken pace, not written prose",
        "Interview questions (if included) are open-ended, not leading",
        "B-roll cues are specific and shootable, not abstract",
    ],
)
```

```python
# src/skills/commercial.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

COMMERCIAL = SkillProfile(
    name="commercial",
    display_name="Commercial",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are an award-winning commercial copywriter/director. You script "
        "high-tempo 15s/30s/60s spots in Dual-Column format: hook the viewer in "
        "the first 3 seconds, establish the problem, land an emotional beat, "
        "showcase the product, and close on a clear call to action (CTA)."
    ),
    outline_template=[
        "Hook (0-3s)",
        "Problem / tension",
        "Emotional turn / brand promise",
        "Product hero shot",
        "Call to action + brand lockup",
    ],
    review_criteria=[
        "The hook lands within the first 3 seconds of screen time",
        "Exactly one clear call to action appears near the end",
        "Product placement is specific (shot type, timing) not generic",
        "Runtime implied by the beats matches the target spot length",
        "Emotional beat is earned, not just stated in VO",
    ],
)
```

```python
# src/skills/product_shot.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

PRODUCT_SHOT = SkillProfile(
    name="product_shot",
    display_name="Product Shot",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a spec-ad director specializing in sensorially rich product "
        "films. You script in Dual-Column format with macro-lens camera "
        "movements, deliberate lighting setups, and ambient foley/sound design "
        "notes for every beat."
    ),
    outline_template=[
        "Establishing atmosphere shot",
        "Macro hero reveal of the product",
        "Texture / material detail beats",
        "Motion / interaction beat",
        "Final hero frame + logo",
    ],
    review_criteria=[
        "Every visual cue specifies lens/framing (e.g. macro, slow motion, close up)",
        "Lighting is explicitly described for each major beat",
        "At least one foley/ambient sound design note appears in the audio column",
        "Pacing favors extended, lingering shots over quick cuts",
        "The product is the visual subject of the majority of beats",
    ],
)
```

```python
# src/skills/learning_dev.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

LEARNING_DEV = SkillProfile(
    name="learning_dev",
    display_name="Learning & Development",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are an instructional designer scripting a corporate training "
        "video. You script in Dual-Column format: the VISUAL column carries "
        "on-screen text (OST) and presenter direction, the AUDIO column carries "
        "narration written to a stated learning objective, with interactive "
        "pause prompts inserted where a learner should reflect or act."
    ),
    outline_template=[
        "Learning objective stated up front",
        "Concept introduction",
        "Worked example / demonstration",
        "Interactive pause / knowledge check",
        "Summary and objective recap",
    ],
    review_criteria=[
        "The learning objective is stated explicitly near the start",
        "On-screen text (OST) beats are short enough to read in the shot duration",
        "At least one interactive pause prompt is present",
        "Narration avoids jargon the target learner wouldn't know",
        "The closing beat recaps the stated objective",
    ],
)
```

```python
# src/skills/infomedia.py
from __future__ import annotations

from src.skills.base import OutputFormat, SkillProfile

INFOMEDIA = SkillProfile(
    name="infomedia",
    display_name="Infomedia / Explainer",
    output_format=OutputFormat.DUAL_COLUMN,
    persona=(
        "You are a motion-graphics explainer-video writer. You script in "
        "Dual-Column format using the Hook-Retain-Payoff structure: kinetic "
        "typography and infographic transitions in the VISUAL column, "
        "brisk narration paced to match the graphics in the AUDIO column."
    ),
    outline_template=[
        "Hook - the question or problem",
        "Retain - build understanding with kinetic graphics",
        "Payoff - the resolution or key takeaway",
        "Infographic summary beat",
        "Closing call to action",
    ],
    review_criteria=[
        "Follows Hook-Retain-Payoff structure recognizably",
        "Each visual beat names a specific motion-graphics treatment (kinetic type, icon animation, chart build, etc.)",
        "Narration pace notes (words per beat) are plausible for the stated beat duration",
        "Infographic/data beats are simple enough to parse on a single screen",
        "Closing beat delivers one clear takeaway, not several",
    ],
)
```

- [ ] **Step 4: Update `src/skills/__init__.py` to register all six**

```python
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


_REGISTRY: dict[str, SkillProfile] = {
    skill.name: skill
    for skill in (SHORT_FILM, DOCUMENTARY, COMMERCIAL, PRODUCT_SHOT, LEARNING_DEV, INFOMEDIA)
}


def get_skill(name: str) -> SkillProfile:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise UnknownSkillError(f"Unknown skill '{name}'. Available: {sorted(_REGISTRY)}") from exc


def available_skills() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_skills.py -v`
Expected: PASS (17 tests)

- [ ] **Step 6: Commit**

```bash
git add src/skills/documentary.py src/skills/commercial.py src/skills/product_shot.py \
        src/skills/learning_dev.py src/skills/infomedia.py src/skills/__init__.py tests/test_skills.py
git commit -m "feat: add remaining five Dual-Column skill profiles"
```

---

### Task 11: Formatters - dual_column.py

**Files:**
- Create: `src/formatters/dual_column.py`
- Test: `tests/test_formatters.py`

**Interfaces:**
- Consumes: `extract_json_array` from `src.utils`.
- Produces: `TABLE_HEADER: str`; `class AVBeat(BaseModel) {timecode: str, visual: str, audio: str}`; `parse_av_beats(raw: str) -> list[AVBeat]`; `render_dual_column_table(beats: list[AVBeat]) -> str`. `AVBeat` is consumed by `formatters/fountain.py` and `graph/nodes/writer.py`/`finalize.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_formatters.py
from __future__ import annotations

from src.formatters.dual_column import AVBeat, TABLE_HEADER, parse_av_beats, render_dual_column_table


def test_parse_av_beats_valid_json():
    raw = (
        'Here is the script:\n'
        '[{"timecode": "0:00-0:03", "visual": "Logo reveal", "audio": "Upbeat sting"}]'
    )
    beats = parse_av_beats(raw)
    assert len(beats) == 1
    assert beats[0].timecode == "0:00-0:03"
    assert beats[0].visual == "Logo reveal"


def test_parse_av_beats_invalid_returns_empty():
    assert parse_av_beats("not json at all") == []


def test_render_dual_column_table_header_only_when_empty():
    assert render_dual_column_table([]) == TABLE_HEADER


def test_render_dual_column_table_includes_rows():
    beats = [AVBeat(timecode="0:00", visual="V1", audio="A1")]
    table = render_dual_column_table(beats)
    assert "| 0:00 | V1 | A1 |" in table
    assert table.startswith(TABLE_HEADER)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_formatters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.formatters.dual_column'`

- [ ] **Step 3: Implement `src/formatters/dual_column.py`**

```python
# src/formatters/dual_column.py
from __future__ import annotations

from pydantic import BaseModel

from src.utils import extract_json_array

TABLE_HEADER = "| TIMECODE / BEAT | VISUAL & CAMERA (VIDEO) | AUDIO, DIALOGUE & SFX |\n|---|---|---|\n"


class AVBeat(BaseModel):
    timecode: str
    visual: str
    audio: str


def parse_av_beats(raw: str) -> list[AVBeat]:
    return extract_json_array(raw, AVBeat)


def render_dual_column_table(beats: list[AVBeat]) -> str:
    if not beats:
        return TABLE_HEADER
    rows = "\n".join(f"| {b.timecode} | {b.visual} | {b.audio} |" for b in beats)
    return TABLE_HEADER + rows + "\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_formatters.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/formatters/dual_column.py tests/test_formatters.py
git commit -m "feat: add Dual-Column A/V beat model and table renderer"
```

---

### Task 12: Formatters - fountain.py

**Files:**
- Create: `src/formatters/fountain.py`
- Modify: `tests/test_formatters.py` (append)

**Interfaces:**
- Consumes: `AVBeat` from `src.formatters.dual_column`.
- Produces: `render_fountain(draft: str) -> str`; `validate_fountain(text: str) -> list[str]`; `render_dual_column_as_fountain(beats: list[AVBeat]) -> str`. Consumed by `graph/nodes/finalize.py`.

- [ ] **Step 1: Append the failing tests to `tests/test_formatters.py`**

```python
# --- append to tests/test_formatters.py ---
from src.formatters.fountain import render_dual_column_as_fountain, render_fountain, validate_fountain


def test_render_fountain_uppercases_lowercase_scene_heading():
    draft = "int. kitchen - day\n\nShe walks in and sits down.\n"
    result = render_fountain(draft)
    assert result.splitlines()[0] == "INT. KITCHEN - DAY"


def test_render_fountain_leaves_action_lines_unchanged():
    draft = "INT. KITCHEN - DAY\n\nShe walks in.\n"
    result = render_fountain(draft)
    assert "She walks in." in result


def test_validate_fountain_flags_long_action_block():
    text = "INT. KITCHEN - DAY\n\nOne.\nTwo.\nThree.\nFour.\nFive.\n"
    warnings = validate_fountain(text)
    assert len(warnings) == 1
    assert "5 lines" in warnings[0]


def test_validate_fountain_allows_short_action_block():
    text = "INT. KITCHEN - DAY\n\nShe enters.\nShe sits.\n"
    assert validate_fountain(text) == []


def test_validate_fountain_resets_after_character_cue():
    text = "INT. KITCHEN - DAY\n\nOne.\nTwo.\nThree.\nFour.\n\nJANE\nHello there.\n"
    assert validate_fountain(text) == []


def test_render_dual_column_as_fountain_produces_narrator_cues():
    beats = [AVBeat(timecode="0:00-0:03", visual="Logo reveal on black.", audio="Upbeat sting plays.")]
    result = render_dual_column_as_fountain(beats)
    assert "[[0:00-0:03]]" in result
    assert "Logo reveal on black." in result
    assert "NARRATOR" in result
    assert "Upbeat sting plays." in result
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_formatters.py -k fountain -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.formatters.fountain'`

- [ ] **Step 3: Implement `src/formatters/fountain.py`**

```python
# src/formatters/fountain.py
from __future__ import annotations

import re

from src.formatters.dual_column import AVBeat

_SCENE_HEADING_RE = re.compile(r"^(INT|EXT|INT\./EXT|I/E)[\./ ]", re.IGNORECASE)


def render_fountain(draft: str) -> str:
    """Uppercases scene heading lines; leaves everything else as authored."""
    rendered_lines: list[str] = []
    for line in draft.splitlines():
        stripped = line.strip()
        if _SCENE_HEADING_RE.match(stripped):
            rendered_lines.append(stripped.upper())
        else:
            rendered_lines.append(line)
    return "\n".join(rendered_lines)


def validate_fountain(text: str) -> list[str]:
    """Best-effort format linter. Never raises; returns human-readable warnings."""
    warnings: list[str] = []
    block: list[str] = []

    def flush_block() -> None:
        if len(block) > 4:
            preview = " ".join(block)[:60]
            warnings.append(f"Action block has {len(block)} lines (max 4): {preview!r}")
        block.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_block()
            continue
        is_heading = bool(_SCENE_HEADING_RE.match(stripped))
        is_character_cue = stripped.isupper() and len(stripped.split()) <= 6
        if is_heading or is_character_cue:
            flush_block()
            continue
        block.append(stripped)
    flush_block()
    return warnings


def render_dual_column_as_fountain(beats: list[AVBeat]) -> str:
    """Maps Dual-Column A/V beats into Fountain conventions: action lines for
    VISUAL, a NARRATOR character cue for AUDIO. SFX/OST/graphics notes the
    writer already wrote inline in visual/audio text pass through as-is."""
    lines: list[str] = []
    for beat in beats:
        lines.append(f"[[{beat.timecode}]]")
        lines.append(beat.visual.strip())
        lines.append("")
        lines.append("NARRATOR")
        lines.append(beat.audio.strip())
        lines.append("")
    return "\n".join(lines).rstrip("\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_formatters.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/formatters/fountain.py tests/test_formatters.py
git commit -m "feat: add Fountain renderer/validator and Dual-Column-to-Fountain mapping"
```

---

### Task 13: Formatters - bible.py

**Files:**
- Create: `src/formatters/bible.py`
- Modify: `tests/test_formatters.py` (append)

**Interfaces:**
- Consumes: `extract_json_array` from `src.utils`.
- Produces: `class CharacterBibleEntry(BaseModel) {name, role, appearance, personality, voice, backstory, headshot_prompt, contact_sheet_prompt, wardrobe_prompt: str}`; `class LocationBibleEntry(BaseModel) {name, description, mood, t2i_prompt: str}`; `parse_character_entries(raw: str) -> list[CharacterBibleEntry]`; `parse_location_entries(raw: str) -> list[LocationBibleEntry]`; `render_character_bible(entries) -> str`; `render_location_bible(entries) -> str`. Consumed by `graph/nodes/character_bible.py` and `location_bible.py`.

- [ ] **Step 1: Append the failing tests to `tests/test_formatters.py`**

```python
# --- append to tests/test_formatters.py ---
from src.formatters.bible import (
    CharacterBibleEntry,
    LocationBibleEntry,
    parse_character_entries,
    parse_location_entries,
    render_character_bible,
    render_location_bible,
)


def _character() -> CharacterBibleEntry:
    return CharacterBibleEntry(
        name="Jane Voss",
        role="Protagonist",
        appearance="Sharp business attire, tired eyes, close-cropped dark hair.",
        personality="Relentless, dryly funny under pressure.",
        voice="Clipped, impatient, precise.",
        backstory="Former detective turned voicemail-service owner.",
        headshot_prompt="Create a headshot of Jane Voss, a woman with tired eyes.",
        contact_sheet_prompt="Create a 3x2 contact sheet of Jane Voss in six expressions.",
        wardrobe_prompt="Create a full-body wardrobe shot of Jane Voss in sharp business attire.",
    )


def _location() -> LocationBibleEntry:
    return LocationBibleEntry(
        name="Voss Voicemail Office",
        description="A cramped, fluorescent-lit office stacked with old answering machines.",
        mood="Tense and claustrophobic.",
        t2i_prompt="Create a wide shot of a cramped, fluorescent-lit detective office at night.",
    )


def test_parse_character_entries_valid_json():
    raw = f'[{_character().model_dump_json()}]'
    entries = parse_character_entries(raw)
    assert len(entries) == 1
    assert entries[0].name == "Jane Voss"


def test_parse_character_entries_invalid_returns_empty():
    assert parse_character_entries("not json") == []


def test_render_character_bible_empty_list():
    assert render_character_bible([]) == "# Character Bible\n\nNo principal characters identified.\n"


def test_render_character_bible_includes_t2i_boxes():
    rendered = render_character_bible([_character()])
    assert "Jane Voss" in rendered
    assert "**Headshot Prompt:**" in rendered
    assert "```text" in rendered
    assert "Create a headshot of Jane Voss" in rendered
    assert "**Contact Sheet Prompt:**" in rendered
    assert "**Wardrobe & Accessories Prompt:**" in rendered


def test_parse_location_entries_valid_json():
    raw = f'[{_location().model_dump_json()}]'
    entries = parse_location_entries(raw)
    assert len(entries) == 1
    assert entries[0].name == "Voss Voicemail Office"


def test_render_location_bible_empty_list():
    assert render_location_bible([]) == "# Location Bible\n\nNo locations identified.\n"


def test_render_location_bible_includes_t2i_box():
    rendered = render_location_bible([_location()])
    assert "Voss Voicemail Office" in rendered
    assert "**Location T2I Prompt:**" in rendered
    assert "```text" in rendered
    assert "cramped, fluorescent-lit detective office" in rendered
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_formatters.py -k bible -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.formatters.bible'`

- [ ] **Step 3: Implement `src/formatters/bible.py`**

```python
# src/formatters/bible.py
from __future__ import annotations

from pydantic import BaseModel

from src.utils import extract_json_array


class CharacterBibleEntry(BaseModel):
    name: str
    role: str
    appearance: str
    personality: str
    voice: str
    backstory: str
    headshot_prompt: str
    contact_sheet_prompt: str
    wardrobe_prompt: str


class LocationBibleEntry(BaseModel):
    name: str
    description: str
    mood: str
    t2i_prompt: str


def parse_character_entries(raw: str) -> list[CharacterBibleEntry]:
    return extract_json_array(raw, CharacterBibleEntry)


def parse_location_entries(raw: str) -> list[LocationBibleEntry]:
    return extract_json_array(raw, LocationBibleEntry)


def _t2i_box(label: str, prompt: str) -> str:
    return f"**{label}:**\n```text\n{prompt}\n```\n"


def render_character_bible(entries: list[CharacterBibleEntry]) -> str:
    if not entries:
        return "# Character Bible\n\nNo principal characters identified.\n"
    sections = ["# Character Bible\n"]
    for entry in entries:
        sections.append(f"## {entry.name} — {entry.role}\n")
        sections.append(f"**Appearance:** {entry.appearance}\n")
        sections.append(f"**Personality:** {entry.personality}\n")
        sections.append(f"**Voice:** {entry.voice}\n")
        sections.append(f"**Backstory:** {entry.backstory}\n")
        sections.append(_t2i_box("Headshot Prompt", entry.headshot_prompt))
        sections.append(_t2i_box("Contact Sheet Prompt", entry.contact_sheet_prompt))
        sections.append(_t2i_box("Wardrobe & Accessories Prompt", entry.wardrobe_prompt))
    return "\n".join(sections)


def render_location_bible(entries: list[LocationBibleEntry]) -> str:
    if not entries:
        return "# Location Bible\n\nNo locations identified.\n"
    sections = ["# Location Bible\n"]
    for entry in entries:
        sections.append(f"## {entry.name}\n")
        sections.append(f"**Description:** {entry.description}\n")
        sections.append(f"**Mood:** {entry.mood}\n")
        sections.append(_t2i_box("Location T2I Prompt", entry.t2i_prompt))
    return "\n".join(sections)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_formatters.py -v`
Expected: PASS (18 tests)

- [ ] **Step 5: Commit**

```bash
git add src/formatters/bible.py tests/test_formatters.py
git commit -m "feat: add character/location bible models and T2I-prompt-box renderer"
```

---

### Task 14: Graph State (`src/graph/state.py`)

**Files:**
- Create: `src/graph/state.py`
- Test: `tests/test_graph_state.py`

**Interfaces:**
- Produces: `class InterviewTurn(TypedDict) {question: str, answer: str}`; `class ScreenplayState(TypedDict)` with fields `topic, skill, file_paths: list[str], parsed_context, rag_run_id, rag_indexed: bool, research_notes, interview_transcript: list[InterviewTurn], interview_turn_count: int, interview_complete: bool, pending_question, autonomous: bool, outline, draft, review_feedback, review_score: float, revision_count: int, max_revisions: int, character_bible, location_bible, bible_review_feedback, bible_review_score: float, bible_revision_count: int, max_bible_revisions: int, fountain_script, screenplay_markdown, status`; `NodeUpdate = dict[str, Any]`; `new_initial_state(*, topic, skill, file_paths, max_revisions, max_bible_revisions, autonomous=False) -> ScreenplayState`. Every later graph node imports `ScreenplayState`/`NodeUpdate`; every graph test imports `new_initial_state`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_state.py
from __future__ import annotations

from src.graph.state import new_initial_state


def test_new_initial_state_has_expected_defaults():
    state = new_initial_state(
        topic="A heist gone wrong",
        skill="short_film",
        file_paths=["notes.md"],
        max_revisions=2,
        max_bible_revisions=1,
    )
    assert state["topic"] == "A heist gone wrong"
    assert state["skill"] == "short_film"
    assert state["file_paths"] == ["notes.md"]
    assert state["parsed_context"] == ""
    assert state["interview_transcript"] == []
    assert state["interview_turn_count"] == 0
    assert state["interview_complete"] is False
    assert state["pending_question"] == ""
    assert state["autonomous"] is False
    assert state["review_score"] == 0.0
    assert state["revision_count"] == 0
    assert state["max_revisions"] == 2
    assert state["max_bible_revisions"] == 1
    assert state["status"] == "initialized"


def test_new_initial_state_autonomous_flag():
    state = new_initial_state(
        topic="x", skill="short_film", file_paths=[], max_revisions=2,
        max_bible_revisions=1, autonomous=True,
    )
    assert state["autonomous"] is True


def test_new_initial_state_is_mutable_dict_for_langgraph_updates():
    state = new_initial_state(
        topic="x", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    state["draft"] = "INT. ROOM - DAY\n\nShe waits.\n"
    assert state["draft"] == "INT. ROOM - DAY\n\nShe waits.\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.state'`

- [ ] **Step 3: Implement `src/graph/state.py`**

```python
# src/graph/state.py
from __future__ import annotations

from typing import Any, TypedDict


class InterviewTurn(TypedDict):
    question: str
    answer: str


class ScreenplayState(TypedDict):
    topic: str
    skill: str
    file_paths: list[str]
    parsed_context: str
    rag_run_id: str
    rag_indexed: bool
    research_notes: str
    interview_transcript: list[InterviewTurn]
    interview_turn_count: int
    interview_complete: bool
    pending_question: str
    autonomous: bool
    outline: str
    draft: str
    review_feedback: str
    review_score: float
    revision_count: int
    max_revisions: int
    character_bible: str
    location_bible: str
    bible_review_feedback: str
    bible_review_score: float
    bible_revision_count: int
    max_bible_revisions: int
    fountain_script: str
    screenplay_markdown: str
    status: str


NodeUpdate = dict[str, Any]
"""Partial-state update returned by LangGraph nodes. `Any` is intentional
here (and only here) - LangGraph's functional-update pattern returns a
dict with heterogeneous value types; every other signature in this
codebase uses concrete types."""


def new_initial_state(
    *,
    topic: str,
    skill: str,
    file_paths: list[str],
    max_revisions: int,
    max_bible_revisions: int,
    autonomous: bool = False,
) -> ScreenplayState:
    return ScreenplayState(
        topic=topic,
        skill=skill,
        file_paths=file_paths,
        parsed_context="",
        rag_run_id="",
        rag_indexed=False,
        research_notes="",
        interview_transcript=[],
        interview_turn_count=0,
        interview_complete=False,
        pending_question="",
        autonomous=autonomous,
        outline="",
        draft="",
        review_feedback="",
        review_score=0.0,
        revision_count=0,
        max_revisions=max_revisions,
        character_bible="",
        location_bible="",
        bible_review_feedback="",
        bible_review_score=0.0,
        bible_revision_count=0,
        max_bible_revisions=max_bible_revisions,
        fountain_script="",
        screenplay_markdown="",
        status="initialized",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_state.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/state.py tests/test_graph_state.py
git commit -m "feat: add ScreenplayState TypedDict and initial-state factory"
```

---

### Task 15: Graph Node - ingest_node

**Files:**
- Create: `src/graph/nodes/ingest.py`
- Test: `tests/test_graph_nodes.py` (create)

**Interfaces:**
- Consumes: `parse_context_files` (`src.parsers.base`), `chunk_parsed_context` (`src.parsers.chunker`), `EphemeralVectorStore`/`register_store` (`src.rag.store`), `Settings`/`load_settings`/`get_embeddings` (`src.config`), `ScreenplayState`/`NodeUpdate`/`new_initial_state` (`src.graph.state`).
- Produces: `class EmbeddingsFn(Protocol)`; `make_ingest_node(embeddings=None, settings=None) -> Callable[[ScreenplayState], NodeUpdate]`. This is the first node wired into `workflow.py` (Task 26).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_nodes.py
from __future__ import annotations

from src.config import Settings
from src.graph.nodes.ingest import make_ingest_node
from src.graph.state import new_initial_state
from src.rag.store import clear_store, get_store


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def _settings(**overrides: object) -> Settings:
    base = dict(
        ollama_model_name="m", ollama_base_url="https://ollama.com", ollama_api_key="k",
        ollama_embed_model="e", tavily_api_key=None,
        rag_chunk_size=50, rag_chunk_overlap=5, rag_top_k=5, rag_min_chars_to_index=20,
        max_interview_questions=5, review_pass_score=8.0, max_bible_revisions=1,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _state(**overrides: object):
    state = new_initial_state(
        topic="t", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_ingest_node_no_files_skips_parsing():
    node = make_ingest_node(settings=_settings())
    result = node(_state(file_paths=[]))
    assert result == {"parsed_context": "", "rag_indexed": False, "status": "ingested"}


def test_ingest_node_short_context_skips_rag(tmp_path):
    md_file = tmp_path / "short.md"
    md_file.write_text("Short note.", encoding="utf-8")
    node = make_ingest_node(settings=_settings(rag_min_chars_to_index=1000))
    result = node(_state(file_paths=[str(md_file)]))
    assert result["rag_indexed"] is False
    assert "Short note." in result["parsed_context"]


def test_ingest_node_long_context_indexes_into_rag(tmp_path):
    md_file = tmp_path / "long.md"
    md_file.write_text("word " * 50, encoding="utf-8")
    node = make_ingest_node(embeddings=_FakeEmbeddings(), settings=_settings(rag_min_chars_to_index=20))
    result = node(_state(file_paths=[str(md_file)]))
    try:
        assert result["rag_indexed"] is True
        assert result["rag_run_id"]
        assert get_store(result["rag_run_id"]) is not None
    finally:
        clear_store(result.get("rag_run_id", ""))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.ingest'`

- [ ] **Step 3: Implement `src/graph/nodes/ingest.py`**

```python
# src/graph/nodes/ingest.py
from __future__ import annotations

import uuid
from typing import Callable, Protocol

from src.config import Settings, get_embeddings, load_settings
from src.graph.state import NodeUpdate, ScreenplayState
from src.parsers.base import parse_context_files
from src.parsers.chunker import chunk_parsed_context
from src.rag.store import EphemeralVectorStore, register_store


class EmbeddingsFn(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def make_ingest_node(
    embeddings: EmbeddingsFn | None = None,
    settings: Settings | None = None,
) -> Callable[[ScreenplayState], NodeUpdate]:
    settings = settings or load_settings()

    def ingest_node(state: ScreenplayState) -> NodeUpdate:
        if not state["file_paths"]:
            return {"parsed_context": "", "rag_indexed": False, "status": "ingested"}

        parsed = parse_context_files(state["file_paths"])
        update: NodeUpdate = {"parsed_context": parsed.combined_text, "status": "ingested"}

        if len(parsed.combined_text) < settings.rag_min_chars_to_index:
            update["rag_indexed"] = False
            return update

        emb = embeddings or get_embeddings(settings)
        chunks = chunk_parsed_context(
            parsed, chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap
        )
        run_id = f"run-{uuid.uuid4().hex}"
        store = EphemeralVectorStore(embeddings=emb, collection_name=run_id)
        store.add_chunks(chunks)
        register_store(run_id, store)
        update["rag_run_id"] = run_id
        update["rag_indexed"] = True
        return update

    return ingest_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/ingest.py tests/test_graph_nodes.py
git commit -m "feat: add ingest_node with threshold-gated ephemeral RAG indexing"
```

---

### Task 16: Graph Node - interview_ask_node + interview_wait_node

**Files:**
- Create: `src/graph/nodes/interviewer.py`
- Test: `tests/test_interview_node.py`

**Interfaces:**
- Consumes: `extract_json_object` (`src.utils`), `get_llm` (`src.config`), `interrupt` (`langgraph.types`).
- Produces: `class InterviewDecision(BaseModel) {has_enough_info: bool, question: str}`; `make_interview_ask_node(llm=None) -> Callable[[ScreenplayState], NodeUpdate]`; `interview_wait_node(state: ScreenplayState) -> NodeUpdate` (module-level, not a factory - it has no LLM dependency to inject). Both wired into `workflow.py` (Task 26); routing in `edges.py` (Task 25) decides when each fires. This task also adds `FakeChatModel` to `tests/conftest.py` - the deterministic offline chat-model double every remaining LLM-calling node test (Tasks 16-24, 28) imports from `tests.conftest`.

- [ ] **Step 1: Add `FakeChatModel` to `tests/conftest.py`**

Append to the existing `tests/conftest.py` (keep the `_default_ollama_env` fixture from Task 2):

```python
# --- append to tests/conftest.py ---
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class FakeChatModel(BaseChatModel):
    """Deterministic offline chat model double. Returns `responses` in order;
    repeats the last one once exhausted. `_call_count` lets tests assert how
    many times `.invoke()` was called."""

    responses: list[str]
    _call_count: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        index = min(self._call_count, len(self.responses) - 1)
        content = self.responses[index]
        self._call_count += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_interview_node.py
from __future__ import annotations

from unittest.mock import patch

from tests.conftest import FakeChatModel
from src.graph.nodes.interviewer import interview_wait_node, make_interview_ask_node
from src.graph.state import new_initial_state


def _state(**overrides: object):
    state = new_initial_state(
        topic="A heist film", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_interview_ask_node_autonomous_skips_llm_call():
    llm = FakeChatModel(responses=["should never be used"])
    node = make_interview_ask_node(llm=llm)
    result = node(_state(autonomous=True))
    assert result == {"interview_complete": True, "pending_question": ""}
    assert llm._call_count == 0


def test_interview_ask_node_asks_question_when_more_info_needed():
    llm = FakeChatModel(responses=['{"has_enough_info": false, "question": "What genre?"}'])
    node = make_interview_ask_node(llm=llm)
    result = node(_state())
    assert result == {"pending_question": "What genre?"}


def test_interview_ask_node_completes_when_llm_has_enough_info():
    llm = FakeChatModel(responses=['{"has_enough_info": true, "question": ""}'])
    node = make_interview_ask_node(llm=llm)
    result = node(_state())
    assert result == {"interview_complete": True, "pending_question": ""}


def test_interview_ask_node_completes_on_unparseable_response():
    llm = FakeChatModel(responses=["not json"])
    node = make_interview_ask_node(llm=llm)
    result = node(_state())
    assert result == {"interview_complete": True, "pending_question": ""}


def test_interview_wait_node_records_turn_from_resume_value():
    state = _state()
    state["pending_question"] = "What genre?"
    with patch(
        "src.graph.nodes.interviewer.interrupt",
        return_value={"answer": "Noir thriller", "autonomous": False},
    ):
        result = interview_wait_node(state)
    assert result["interview_transcript"] == [{"question": "What genre?", "answer": "Noir thriller"}]
    assert result["interview_turn_count"] == 1
    assert result["autonomous"] is False
    assert result["pending_question"] == ""


def test_interview_wait_node_propagates_autonomous_from_resume():
    state = _state()
    state["pending_question"] = "What genre?"
    with patch(
        "src.graph.nodes.interviewer.interrupt",
        return_value={"answer": "/finish", "autonomous": True},
    ):
        result = interview_wait_node(state)
    assert result["autonomous"] is True
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_interview_node.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.interviewer'`

- [ ] **Step 4: Implement `src/graph/nodes/interviewer.py`**

```python
# src/graph/nodes/interviewer.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from src.config import get_llm
from src.graph.state import InterviewTurn, NodeUpdate, ScreenplayState
from src.utils import extract_json_object


class InterviewDecision(BaseModel):
    has_enough_info: bool
    question: str = ""


_SYSTEM_PROMPT = (
    "You are a screenwriting collaborator interviewing the user to sharpen "
    "creative direction before drafting. Ask exactly one focused question at "
    "a time about tone, character, structure, or constraints not yet covered "
    "by the topic, uploaded context, or prior answers. "
    'Respond with JSON only: {"has_enough_info": bool, "question": str}. '
    'Set has_enough_info to true (and question to "") once you have enough '
    "direction to write a strong first draft."
)


def make_interview_ask_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.5)

    def interview_ask_node(state: ScreenplayState) -> NodeUpdate:
        if state["autonomous"]:
            return {"interview_complete": True, "pending_question": ""}

        transcript_text = "\n".join(
            f"Q: {turn['question']}\nA: {turn['answer']}" for turn in state["interview_transcript"]
        ) or "(none yet)"
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\nSkill: {state['skill']}\n"
                    f"Context summary: {state['parsed_context'][:2000]}\n\n"
                    f"Interview so far:\n{transcript_text}"
                )
            ),
        ]
        response = llm.invoke(messages)
        decision = extract_json_object(str(response.content), InterviewDecision)
        if decision is None or decision.has_enough_info or not decision.question:
            return {"interview_complete": True, "pending_question": ""}
        return {"pending_question": decision.question}

    return interview_ask_node


def interview_wait_node(state: ScreenplayState) -> NodeUpdate:
    resumed = interrupt(state["pending_question"])
    turn: InterviewTurn = {"question": state["pending_question"], "answer": resumed["answer"]}
    return {
        "interview_transcript": state["interview_transcript"] + [turn],
        "interview_turn_count": state["interview_turn_count"] + 1,
        "autonomous": bool(resumed.get("autonomous", False)),
        "pending_question": "",
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_interview_node.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add src/graph/nodes/interviewer.py tests/test_interview_node.py tests/conftest.py
git commit -m "feat: add two-node interview loop (ask/wait) using LangGraph interrupt"
```

---

### Task 17: Graph Node - researcher_node

**Files:**
- Create: `src/graph/nodes/researcher.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `search`, `SearchResult` (`src.tools.search`).
- Produces: `make_researcher_node(enabled: bool, search_fn=search) -> Callable[[ScreenplayState], NodeUpdate]`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from src.graph.nodes.researcher import make_researcher_node
from src.tools.search import SearchResult


def test_researcher_node_disabled_returns_empty_notes():
    node = make_researcher_node(enabled=False, search_fn=lambda q: [SearchResult(title="x", url="u", snippet="s")])
    result = node(_state(topic="anything"))
    assert result["research_notes"] == ""


def test_researcher_node_enabled_formats_results():
    def fake_search(query: str) -> list[SearchResult]:
        assert query == "space tourism"
        return [SearchResult(title="Title", url="http://x", snippet="Snippet text")]

    node = make_researcher_node(enabled=True, search_fn=fake_search)
    result = node(_state(topic="space tourism"))
    assert "Title" in result["research_notes"]
    assert "Snippet text" in result["research_notes"]
    assert "http://x" in result["research_notes"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k researcher -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.researcher'`

- [ ] **Step 3: Implement `src/graph/nodes/researcher.py`**

```python
# src/graph/nodes/researcher.py
from __future__ import annotations

from typing import Callable

from src.graph.state import NodeUpdate, ScreenplayState
from src.tools.search import SearchResult, search


def make_researcher_node(
    enabled: bool, search_fn: Callable[[str], list[SearchResult]] = search
) -> Callable[[ScreenplayState], NodeUpdate]:
    def researcher_node(state: ScreenplayState) -> NodeUpdate:
        if not enabled:
            return {"research_notes": "", "status": "researched"}
        results = search_fn(state["topic"])
        notes = "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results)
        return {"research_notes": notes, "status": "researched"}

    return researcher_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/researcher.py tests/test_graph_nodes.py
git commit -m "feat: add researcher_node wrapping the search fallback"
```

---

### Task 18: Graph Node - outliner_node

**Files:**
- Create: `src/graph/nodes/outliner.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `get_llm` (`src.config`), `retrieve_context` (`src.rag.retriever`), `get_skill` (`src.skills`).
- Produces: `make_outliner_node(llm=None, retrieve_fn=retrieve_context, rag_top_k=5) -> Callable[[ScreenplayState], NodeUpdate]`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from tests.conftest import FakeChatModel
from src.graph.nodes.outliner import make_outliner_node


def test_outliner_node_returns_llm_output_as_outline():
    llm = FakeChatModel(responses=["1. Hook\n2. Climax\n3. Resolution"])
    node = make_outliner_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    result = node(_state(topic="A retired detective solves crimes via voicemail"))
    assert result["outline"] == "1. Hook\n2. Climax\n3. Resolution"
    assert result["status"] == "outlined"


def test_outliner_node_passes_retrieved_chunks_into_prompt():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=["outline text"])
    node = make_outliner_node(
        llm=llm, retrieve_fn=lambda run_id, query, k: ["Retrieved chunk about the detective's past."]
    )
    node(_state(topic="t", rag_run_id="run-1"))
    human_content = str(captured_messages[0][-1].content)
    assert "Retrieved chunk about the detective's past." in human_content
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k outliner -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.outliner'`

- [ ] **Step 3: Implement `src/graph/nodes/outliner.py`**

```python
# src/graph/nodes/outliner.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.graph.state import NodeUpdate, ScreenplayState
from src.rag.retriever import retrieve_context
from src.skills import get_skill


def make_outliner_node(
    llm: BaseChatModel | None = None,
    retrieve_fn: Callable[[str, str, int], list[str]] = retrieve_context,
    rag_top_k: int = 5,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.7)

    def outliner_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        transcript_text = "\n".join(
            f"Q: {t['question']}\nA: {t['answer']}" for t in state["interview_transcript"]
        )
        grounding_chunks = retrieve_fn(state["rag_run_id"], state["topic"], rag_top_k)
        grounding = "\n\n".join(grounding_chunks) or state["parsed_context"]
        messages = [
            SystemMessage(content=skill.system_prompt()),
            HumanMessage(
                content=(
                    f"{skill.outline_prompt()}\n\n"
                    f"Topic: {state['topic']}\n"
                    f"Interview notes:\n{transcript_text or '(none)'}\n\n"
                    f"Research notes:\n{state['research_notes'] or '(none)'}\n\n"
                    f"Source context:\n{grounding or '(none)'}"
                )
            ),
        ]
        response = llm.invoke(messages)
        return {"outline": str(response.content), "status": "outlined"}

    return outliner_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/outliner.py tests/test_graph_nodes.py
git commit -m "feat: add RAG-aware outliner_node"
```

---

### Task 19: Graph Node - writer_node

**Files:**
- Create: `src/graph/nodes/writer.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `get_llm` (`src.config`), `retrieve_context` (`src.rag.retriever`), `get_skill` (`src.skills`), `OutputFormat` (`src.skills.base`).
- Produces: `make_writer_node(llm=None, retrieve_fn=retrieve_context, rag_top_k=5) -> Callable[[ScreenplayState], NodeUpdate]`. On a revision pass (`state["review_feedback"]` non-empty) it also bumps `revision_count`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from src.graph.nodes.writer import make_writer_node
from src.skills.base import OutputFormat


def test_writer_node_first_draft_does_not_bump_revision_count():
    llm = FakeChatModel(responses=["INT. ROOM - DAY\n\nShe waits.\n"])
    node = make_writer_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    result = node(_state(outline="1. Hook", review_feedback=""))
    assert result["draft"] == "INT. ROOM - DAY\n\nShe waits.\n"
    assert "revision_count" not in result


def test_writer_node_revision_pass_bumps_revision_count():
    llm = FakeChatModel(responses=["INT. ROOM - DAY\n\nShe answers.\n"])
    node = make_writer_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    result = node(_state(outline="1. Hook", review_feedback="Trim the action lines.", revision_count=0))
    assert result["revision_count"] == 1


def test_writer_node_uses_dual_column_instruction_for_commercial_skill():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=['[{"timecode": "0:00", "visual": "v", "audio": "a"}]'])
    node = make_writer_node(llm=llm, retrieve_fn=lambda run_id, query, k: [])
    node(_state(skill="commercial", outline="1. Hook"))
    human_content = str(captured_messages[0][-1].content)
    assert "JSON array" in human_content
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k writer -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.writer'`

- [ ] **Step 3: Implement `src/graph/nodes/writer.py`**

```python
# src/graph/nodes/writer.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.graph.state import NodeUpdate, ScreenplayState
from src.rag.retriever import retrieve_context
from src.skills import get_skill
from src.skills.base import OutputFormat


def make_writer_node(
    llm: BaseChatModel | None = None,
    retrieve_fn: Callable[[str, str, int], list[str]] = retrieve_context,
    rag_top_k: int = 5,
) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.8)

    def writer_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        grounding_chunks = retrieve_fn(state["rag_run_id"], state["outline"], rag_top_k)
        grounding = "\n\n".join(grounding_chunks) or state["parsed_context"]

        if skill.output_format == OutputFormat.FOUNTAIN:
            format_instruction = "Write the full script in Master Scene (Fountain) Format."
        else:
            format_instruction = (
                'Write the full script as a JSON array only: [{"timecode": str, '
                '"visual": str, "audio": str}, ...] - one object per beat.'
            )

        is_revision = bool(state["review_feedback"])
        revision_instruction = (
            f"\n\nThis is a revision. Address this feedback:\n{state['review_feedback']}"
            if is_revision
            else ""
        )
        messages = [
            SystemMessage(content=skill.system_prompt()),
            HumanMessage(
                content=(
                    f"Outline:\n{state['outline']}\n\n"
                    f"Source context:\n{grounding or '(none)'}\n\n"
                    f"{format_instruction}{revision_instruction}"
                )
            ),
        ]
        response = llm.invoke(messages)
        update: NodeUpdate = {"draft": str(response.content), "status": "drafted"}
        if is_revision:
            update["revision_count"] = state["revision_count"] + 1
        return update

    return writer_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/writer.py tests/test_graph_nodes.py
git commit -m "feat: add RAG-aware writer_node with format-aware drafting"
```

---

### Task 20: Graph Node - reviewer_node (+ review_shared.py)

**Files:**
- Create: `src/graph/nodes/review_shared.py`, `src/graph/nodes/reviewer.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `extract_json_object` (`src.utils`), `get_llm` (`src.config`), `get_skill` (`src.skills`).
- Produces (`review_shared.py`, reused by `bible_reviewer.py` in Task 23): `REVIEW_RESPONSE_INSTRUCTION: str`; `class ReviewFeedback(BaseModel) {score: float, passed: bool, critique: str, actionable_revisions: list[str]}`; `FALLBACK_REVIEW_FEEDBACK: ReviewFeedback`; `format_critique(feedback: ReviewFeedback) -> str`.
- Produces (`reviewer.py`): `make_reviewer_node(llm=None) -> Callable[[ScreenplayState], NodeUpdate]`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from src.graph.nodes.reviewer import make_reviewer_node


def test_reviewer_node_parses_valid_json():
    llm = FakeChatModel(responses=[
        '{"score": 9.2, "passed": true, "critique": "Strong draft.", "actionable_revisions": []}'
    ])
    node = make_reviewer_node(llm=llm)
    result = node(_state(draft="INT. ROOM - DAY\n\nShe waits."))
    assert result["review_score"] == 9.2
    assert "Strong draft." in result["review_feedback"]


def test_reviewer_node_includes_actionable_revisions_in_feedback():
    llm = FakeChatModel(responses=[
        '{"score": 5.0, "passed": false, "critique": "Needs work.", '
        '"actionable_revisions": ["Trim action lines", "Fix slugline case"]}'
    ])
    node = make_reviewer_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert result["review_score"] == 5.0
    assert "Trim action lines" in result["review_feedback"]
    assert "Fix slugline case" in result["review_feedback"]


def test_reviewer_node_falls_back_gracefully_on_unparseable_response():
    llm = FakeChatModel(responses=["I refuse to output JSON today."])
    node = make_reviewer_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert result["review_score"] == 0.0
    assert "could not be parsed" in result["review_feedback"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k reviewer -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.reviewer'`

- [ ] **Step 3: Implement `src/graph/nodes/review_shared.py`**

```python
# src/graph/nodes/review_shared.py
from __future__ import annotations

from pydantic import BaseModel, Field

REVIEW_RESPONSE_INSTRUCTION = (
    'Respond with JSON only: {"score": <0-10 float>, "passed": <bool>, '
    '"critique": <string>, "actionable_revisions": [<string>, ...]}'
)


class ReviewFeedback(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    passed: bool
    critique: str
    actionable_revisions: list[str] = Field(default_factory=list)


FALLBACK_REVIEW_FEEDBACK = ReviewFeedback(
    score=0.0,
    passed=False,
    critique="Reviewer response could not be parsed as valid JSON.",
    actionable_revisions=["Retry: ensure the reviewer responds with JSON only."],
)


def format_critique(feedback: ReviewFeedback) -> str:
    critique_text = feedback.critique
    if feedback.actionable_revisions:
        bullets = "\n".join(f"- {r}" for r in feedback.actionable_revisions)
        critique_text = f"{critique_text}\n\nActionable revisions:\n{bullets}"
    return critique_text
```

- [ ] **Step 4: Implement `src/graph/nodes/reviewer.py`**

```python
# src/graph/nodes/reviewer.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.graph.nodes.review_shared import (
    FALLBACK_REVIEW_FEEDBACK,
    REVIEW_RESPONSE_INSTRUCTION,
    ReviewFeedback,
    format_critique,
)
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills import get_skill
from src.utils import extract_json_object


def make_reviewer_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.0)

    def reviewer_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        messages = [
            SystemMessage(content=skill.review_system_prompt()),
            HumanMessage(
                content=f"Script to review:\n\n{state['draft']}\n\n{REVIEW_RESPONSE_INSTRUCTION}"
            ),
        ]
        response = llm.invoke(messages)
        feedback = extract_json_object(str(response.content), ReviewFeedback) or FALLBACK_REVIEW_FEEDBACK
        return {
            "review_score": feedback.score,
            "review_feedback": format_critique(feedback),
            "status": "reviewed",
        }

    return reviewer_node
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (13 tests)

- [ ] **Step 6: Commit**

```bash
git add src/graph/nodes/review_shared.py src/graph/nodes/reviewer.py tests/test_graph_nodes.py
git commit -m "feat: add reviewer_node with shared review-feedback parsing"
```

---

### Task 21: Graph Node - character_bible_node

**Files:**
- Create: `src/graph/nodes/character_bible.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `parse_character_entries`, `render_character_bible` (`src.formatters.bible`), `T2I_PROMPT_GUIDELINES` (`src.skills.base`), `get_llm` (`src.config`).
- Produces: `make_character_bible_node(llm=None) -> Callable[[ScreenplayState], NodeUpdate]`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from src.graph.nodes.character_bible import make_character_bible_node

_CHARACTER_JSON = (
    '[{"name": "Jane", "role": "Protagonist", "appearance": "Sharp business attire.", '
    '"personality": "Relentless.", "voice": "Clipped.", "backstory": "Ex-detective.", '
    '"headshot_prompt": "Create a headshot of Jane.", '
    '"contact_sheet_prompt": "Create a contact sheet of Jane.", '
    '"wardrobe_prompt": "Create a wardrobe shot of Jane."}]'
)


def test_character_bible_node_renders_entries_from_valid_json():
    llm = FakeChatModel(responses=[_CHARACTER_JSON])
    node = make_character_bible_node(llm=llm)
    result = node(_state(draft="INT. OFFICE - DAY\n\nJANE stares at the phone."))
    assert "Jane" in result["character_bible"]
    assert "Headshot Prompt" in result["character_bible"]


def test_character_bible_node_empty_on_unparseable_response():
    llm = FakeChatModel(responses=["not json"])
    node = make_character_bible_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert result["character_bible"] == "# Character Bible\n\nNo principal characters identified.\n"


def test_character_bible_node_includes_prior_feedback_when_revising():
    captured_messages = []

    class _CapturingLLM(FakeChatModel):
        def invoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.append(messages)
            return super().invoke(messages, **kwargs)

    llm = _CapturingLLM(responses=[_CHARACTER_JSON])
    node = make_character_bible_node(llm=llm)
    node(_state(draft="draft text", bible_review_feedback="Add more wardrobe detail."))
    human_content = str(captured_messages[0][-1].content)
    assert "Add more wardrobe detail." in human_content
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k character_bible -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.character_bible'`

- [ ] **Step 3: Implement `src/graph/nodes/character_bible.py`**

```python
# src/graph/nodes/character_bible.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.bible import parse_character_entries, render_character_bible
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills.base import T2I_PROMPT_GUIDELINES

_SYSTEM_PROMPT = (
    "You identify the principal cast of a script - characters significant enough "
    "to warrant a full production bible entry, not every speaking role - and write "
    "one entry per principal character.\n\n" + T2I_PROMPT_GUIDELINES
)

_RESPONSE_INSTRUCTION = (
    'Respond with a JSON array only: [{"name": str, "role": str, "appearance": str, '
    '"personality": str, "voice": str, "backstory": str, "headshot_prompt": str, '
    '"contact_sheet_prompt": str, "wardrobe_prompt": str}, ...]'
)


def make_character_bible_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def character_bible_node(state: ScreenplayState) -> NodeUpdate:
        feedback_note = (
            f"\n\nAddress this reviewer feedback:\n{state['bible_review_feedback']}"
            if state["bible_review_feedback"]
            else ""
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Screenplay draft:\n\n{state['draft']}\n\n{_RESPONSE_INSTRUCTION}{feedback_note}"
            ),
        ]
        response = llm.invoke(messages)
        entries = parse_character_entries(str(response.content))
        return {"character_bible": render_character_bible(entries)}

    return character_bible_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/character_bible.py tests/test_graph_nodes.py
git commit -m "feat: add character_bible_node with curated principal-cast T2I prompts"
```

---

### Task 22: Graph Node - location_bible_node

**Files:**
- Create: `src/graph/nodes/location_bible.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `parse_location_entries`, `render_location_bible` (`src.formatters.bible`), `T2I_PROMPT_GUIDELINES` (`src.skills.base`), `get_llm` (`src.config`).
- Produces: `make_location_bible_node(llm=None) -> Callable[[ScreenplayState], NodeUpdate]`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from src.graph.nodes.location_bible import make_location_bible_node

_LOCATION_JSON = (
    '[{"name": "Office", "description": "A cramped detective office.", '
    '"mood": "Tense.", "t2i_prompt": "Create a wide shot of a cramped office."}]'
)


def test_location_bible_node_renders_entries_from_valid_json():
    llm = FakeChatModel(responses=[_LOCATION_JSON])
    node = make_location_bible_node(llm=llm)
    result = node(_state(draft="INT. OFFICE - DAY\n\nJANE stares at the phone."))
    assert "Office" in result["location_bible"]
    assert "Location T2I Prompt" in result["location_bible"]


def test_location_bible_node_empty_on_unparseable_response():
    llm = FakeChatModel(responses=["not json"])
    node = make_location_bible_node(llm=llm)
    result = node(_state(draft="draft text"))
    assert result["location_bible"] == "# Location Bible\n\nNo locations identified.\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k location_bible -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.location_bible'`

- [ ] **Step 3: Implement `src/graph/nodes/location_bible.py`**

```python
# src/graph/nodes/location_bible.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.bible import parse_location_entries, render_location_bible
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills.base import T2I_PROMPT_GUIDELINES

_SYSTEM_PROMPT = (
    "You identify every key location in a script and write one production "
    "bible entry per location.\n\n" + T2I_PROMPT_GUIDELINES
)

_RESPONSE_INSTRUCTION = (
    'Respond with a JSON array only: [{"name": str, "description": str, "mood": str, '
    '"t2i_prompt": str}, ...]'
)


def make_location_bible_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def location_bible_node(state: ScreenplayState) -> NodeUpdate:
        feedback_note = (
            f"\n\nAddress this reviewer feedback:\n{state['bible_review_feedback']}"
            if state["bible_review_feedback"]
            else ""
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Screenplay draft:\n\n{state['draft']}\n\n{_RESPONSE_INSTRUCTION}{feedback_note}"
            ),
        ]
        response = llm.invoke(messages)
        entries = parse_location_entries(str(response.content))
        return {"location_bible": render_location_bible(entries)}

    return location_bible_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (18 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/location_bible.py tests/test_graph_nodes.py
git commit -m "feat: add location_bible_node with per-location T2I prompts"
```

---

### Task 23: Graph Node - bible_reviewer_node

**Files:**
- Create: `src/graph/nodes/bible_reviewer.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `ReviewFeedback`, `FALLBACK_REVIEW_FEEDBACK`, `REVIEW_RESPONSE_INSTRUCTION`, `format_critique` (`src.graph.nodes.review_shared`), `extract_json_object` (`src.utils`), `get_llm` (`src.config`).
- Produces: `make_bible_reviewer_node(llm=None) -> Callable[[ScreenplayState], NodeUpdate]`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from src.graph.nodes.bible_reviewer import make_bible_reviewer_node


def test_bible_reviewer_node_parses_valid_json_and_bumps_revision_count():
    llm = FakeChatModel(responses=[
        '{"score": 9.0, "passed": true, "critique": "Consistent.", "actionable_revisions": []}'
    ])
    node = make_bible_reviewer_node(llm=llm)
    result = node(_state(
        draft="draft text", character_bible="# Character Bible\n", location_bible="# Location Bible\n",
        bible_revision_count=0,
    ))
    assert result["bible_review_score"] == 9.0
    assert result["bible_revision_count"] == 1


def test_bible_reviewer_node_falls_back_gracefully_on_unparseable_response():
    llm = FakeChatModel(responses=["not json"])
    node = make_bible_reviewer_node(llm=llm)
    result = node(_state(draft="draft text", character_bible="", location_bible="", bible_revision_count=0))
    assert result["bible_review_score"] == 0.0
    assert result["bible_revision_count"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k bible_reviewer -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.bible_reviewer'`

- [ ] **Step 3: Implement `src/graph/nodes/bible_reviewer.py`**

```python
# src/graph/nodes/bible_reviewer.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.graph.nodes.review_shared import (
    FALLBACK_REVIEW_FEEDBACK,
    REVIEW_RESPONSE_INSTRUCTION,
    ReviewFeedback,
    format_critique,
)
from src.graph.state import NodeUpdate, ScreenplayState
from src.utils import extract_json_object

_SYSTEM_PROMPT = (
    "You are a production bible editor. Review the character and location bibles "
    "together against the screenplay draft they were derived from. Check: every "
    "principal character from the draft is represented, descriptions are consistent "
    "with how the character/location reads in the script, and every entry has all "
    "required fields filled in with concrete (not generic) detail."
)


def make_bible_reviewer_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.0)

    def bible_reviewer_node(state: ScreenplayState) -> NodeUpdate:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Screenplay draft:\n\n{state['draft']}\n\n"
                    f"Character Bible:\n\n{state['character_bible']}\n\n"
                    f"Location Bible:\n\n{state['location_bible']}\n\n{REVIEW_RESPONSE_INSTRUCTION}"
                )
            ),
        ]
        response = llm.invoke(messages)
        feedback = extract_json_object(str(response.content), ReviewFeedback) or FALLBACK_REVIEW_FEEDBACK
        return {
            "bible_review_score": feedback.score,
            "bible_review_feedback": format_critique(feedback),
            "bible_revision_count": state["bible_revision_count"] + 1,
        }

    return bible_reviewer_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (20 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/bible_reviewer.py tests/test_graph_nodes.py
git commit -m "feat: add combined bible_reviewer_node"
```

---

### Task 24: Graph Node - finalize_node

**Files:**
- Create: `src/graph/nodes/finalize.py`
- Modify: `tests/test_graph_nodes.py` (append)

**Interfaces:**
- Consumes: `parse_av_beats`, `render_dual_column_table` (`src.formatters.dual_column`), `render_dual_column_as_fountain`, `render_fountain` (`src.formatters.fountain`), `get_skill` (`src.skills`), `OutputFormat`, `T2I_PROMPT_GUIDELINES` (`src.skills.base`), `get_llm` (`src.config`).
- Produces: `make_finalize_node(llm=None) -> Callable[[ScreenplayState], NodeUpdate]`, returning `{"fountain_script": str, "screenplay_markdown": str, "status": "finalized"}`.

- [ ] **Step 1: Append the failing tests to `tests/test_graph_nodes.py`**

```python
# --- append to tests/test_graph_nodes.py ---
from src.graph.nodes.finalize import make_finalize_node


def test_finalize_node_short_film_produces_fountain_and_markdown():
    llm = FakeChatModel(responses=["Create a moody poster of a detective by a window."])
    node = make_finalize_node(llm=llm)
    result = node(_state(skill="short_film", draft="int. room - day\n\nShe waits.\n"))
    assert result["status"] == "finalized"
    assert result["fountain_script"].splitlines()[0] == "INT. ROOM - DAY"
    assert "Key Art T2I Prompt" in result["screenplay_markdown"]
    assert "She waits." in result["screenplay_markdown"]


def test_finalize_node_dual_column_skill_produces_table_and_narrator_fountain():
    draft = '[{"timecode": "0:00", "visual": "Logo reveal", "audio": "Sting plays"}]'
    llm = FakeChatModel(responses=["Create a minimalist poster with a bold logo."])
    node = make_finalize_node(llm=llm)
    result = node(_state(skill="commercial", draft=draft))
    assert "| 0:00 | Logo reveal | Sting plays |" in result["screenplay_markdown"]
    assert "NARRATOR" in result["fountain_script"]
    assert "Sting plays" in result["fountain_script"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_nodes.py -k finalize -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.nodes.finalize'`

- [ ] **Step 3: Implement `src/graph/nodes/finalize.py`**

```python
# src/graph/nodes/finalize.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.formatters.dual_column import parse_av_beats, render_dual_column_table
from src.formatters.fountain import render_dual_column_as_fountain, render_fountain
from src.graph.state import NodeUpdate, ScreenplayState
from src.skills import get_skill
from src.skills.base import OutputFormat, T2I_PROMPT_GUIDELINES

_KEY_ART_SYSTEM_PROMPT = (
    "You write a single T2I key-art prompt for a screenplay's poster/key visual, "
    "capturing its tone and central image in one image.\n\n" + T2I_PROMPT_GUIDELINES
)


def make_finalize_node(llm: BaseChatModel | None = None) -> Callable[[ScreenplayState], NodeUpdate]:
    llm = llm or get_llm(temperature=0.6)

    def finalize_node(state: ScreenplayState) -> NodeUpdate:
        skill = get_skill(state["skill"])
        key_art_response = llm.invoke(
            [
                SystemMessage(content=_KEY_ART_SYSTEM_PROMPT),
                HumanMessage(content=f"Screenplay draft:\n\n{state['draft']}"),
            ]
        )
        key_art_prompt = str(key_art_response.content)

        if skill.output_format == OutputFormat.FOUNTAIN:
            fountain_script = render_fountain(state["draft"])
            body = state["draft"]
        else:
            beats = parse_av_beats(state["draft"])
            fountain_script = render_dual_column_as_fountain(beats)
            body = render_dual_column_table(beats)

        key_art_box = f"**Key Art T2I Prompt:**\n```text\n{key_art_prompt}\n```\n"
        screenplay_markdown = f"{key_art_box}\n# Screenplay\n\n{body}"

        return {
            "fountain_script": fountain_script,
            "screenplay_markdown": screenplay_markdown,
            "status": "finalized",
        }

    return finalize_node
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_nodes.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/finalize.py tests/test_graph_nodes.py
git commit -m "feat: add finalize_node rendering Fountain, Markdown, and key-art T2I prompt"
```

---

### Task 25: Graph Edges (`src/graph/edges.py`)

**Files:**
- Create: `src/graph/edges.py`
- Test: `tests/test_graph_edges.py`

**Interfaces:**
- Consumes: `Settings`, `load_settings` (`src.config`).
- Produces: `route_after_interview_ask(state, settings=None) -> str` (`"researcher" | "interview_wait"`); `route_after_interview_wait(state) -> str` (`"researcher" | "interview_ask"`); `route_after_reviewer(state, settings=None) -> str` (`"character_bible" | "writer"`); `route_after_bible_reviewer(state, settings=None) -> str` (`"finalize" | "character_bible"`). Wired into `workflow.py` (Task 26) via `add_conditional_edges`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_edges.py
from __future__ import annotations

from src.config import Settings
from src.graph.edges import (
    route_after_bible_reviewer,
    route_after_interview_ask,
    route_after_interview_wait,
    route_after_reviewer,
)
from src.graph.state import new_initial_state


def _settings(**overrides: object) -> Settings:
    base = dict(
        ollama_model_name="m", ollama_base_url="https://ollama.com", ollama_api_key="k",
        ollama_embed_model="e", tavily_api_key=None,
        rag_chunk_size=1000, rag_chunk_overlap=150, rag_top_k=5, rag_min_chars_to_index=4000,
        max_interview_questions=3, review_pass_score=8.0, max_bible_revisions=1,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _state(**overrides: object):
    state = new_initial_state(
        topic="t", skill="short_film", file_paths=[], max_revisions=2, max_bible_revisions=1
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def test_route_after_interview_ask_continues_when_more_questions_needed():
    state = _state(interview_complete=False, autonomous=False, interview_turn_count=1)
    assert route_after_interview_ask(state, _settings()) == "interview_wait"


def test_route_after_interview_ask_proceeds_when_complete():
    assert route_after_interview_ask(_state(interview_complete=True), _settings()) == "researcher"


def test_route_after_interview_ask_proceeds_when_autonomous():
    state = _state(interview_complete=False, autonomous=True)
    assert route_after_interview_ask(state, _settings()) == "researcher"


def test_route_after_interview_ask_proceeds_at_turn_cap():
    state = _state(interview_complete=False, autonomous=False, interview_turn_count=3)
    assert route_after_interview_ask(state, _settings(max_interview_questions=3)) == "researcher"


def test_route_after_interview_wait_loops_back_by_default():
    assert route_after_interview_wait(_state(autonomous=False)) == "interview_ask"


def test_route_after_interview_wait_exits_when_autonomous():
    assert route_after_interview_wait(_state(autonomous=True)) == "researcher"


def test_route_after_reviewer_passes_on_high_score():
    state = _state(review_score=8.5, revision_count=0, max_revisions=2)
    assert route_after_reviewer(state, _settings()) == "character_bible"


def test_route_after_reviewer_revises_on_low_score_under_cap():
    state = _state(review_score=4.0, revision_count=0, max_revisions=2)
    assert route_after_reviewer(state, _settings()) == "writer"


def test_route_after_reviewer_halts_at_revision_cap():
    state = _state(review_score=4.0, revision_count=2, max_revisions=2)
    assert route_after_reviewer(state, _settings()) == "character_bible"


def test_route_after_bible_reviewer_passes_on_high_score():
    state = _state(bible_review_score=9.0, bible_revision_count=0, max_bible_revisions=1)
    assert route_after_bible_reviewer(state, _settings()) == "finalize"


def test_route_after_bible_reviewer_revises_under_cap():
    state = _state(bible_review_score=3.0, bible_revision_count=0, max_bible_revisions=1)
    assert route_after_bible_reviewer(state, _settings()) == "character_bible"


def test_route_after_bible_reviewer_halts_at_cap():
    state = _state(bible_review_score=3.0, bible_revision_count=1, max_bible_revisions=1)
    assert route_after_bible_reviewer(state, _settings()) == "finalize"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_graph_edges.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.edges'`

- [ ] **Step 3: Implement `src/graph/edges.py`**

```python
# src/graph/edges.py
from __future__ import annotations

from src.config import Settings, load_settings
from src.graph.state import ScreenplayState


def route_after_interview_ask(state: ScreenplayState, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if state["interview_complete"]:
        return "researcher"
    if state["autonomous"]:
        return "researcher"
    if state["interview_turn_count"] >= settings.max_interview_questions:
        return "researcher"
    return "interview_wait"


def route_after_interview_wait(state: ScreenplayState) -> str:
    if state["autonomous"]:
        return "researcher"
    return "interview_ask"


def route_after_reviewer(state: ScreenplayState, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if state["review_score"] >= settings.review_pass_score or state["revision_count"] >= state["max_revisions"]:
        return "character_bible"
    return "writer"


def route_after_bible_reviewer(state: ScreenplayState, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    if (
        state["bible_review_score"] >= settings.review_pass_score
        or state["bible_revision_count"] >= state["max_bible_revisions"]
    ):
        return "finalize"
    return "character_bible"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_graph_edges.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/graph/edges.py tests/test_graph_edges.py
git commit -m "feat: add conditional routing for interview, revise, and bible-revise loops"
```

---

### Task 26: Graph Workflow (`src/graph/workflow.py`)

**Files:**
- Create: `src/graph/workflow.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: every `make_*_node`/`interview_wait_node` factory from `src.graph.nodes.*`, every `route_after_*` from `src.graph.edges`, `ScreenplayState` from `src.graph.state`, `search`/`SearchResult` from `src.tools.search`, `EmbeddingsFn` from `src.graph.nodes.ingest`.
- Produces: `build_workflow(enable_search=False, llm=None, embeddings=None, search_fn=search) -> CompiledStateGraph`. This is what `main.py` (Task 27) and the e2e tests (Task 28) call to get a runnable graph, with a shared `llm`/`embeddings` injected into every node for deterministic offline testing.

Note vs. the spec's diagram: `character_bible`/`location_bible` run **sequentially**, not fanned out in parallel - a single-process CLI gets no benefit from LangGraph's parallel-branch machinery here, and sequential edges are simpler to reason about and test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow.py
from __future__ import annotations

from src.graph.workflow import build_workflow


def test_build_workflow_compiles_with_all_expected_nodes():
    app = build_workflow(enable_search=False)
    node_names = set(app.get_graph().nodes.keys())
    for expected in [
        "ingest", "interview_ask", "interview_wait", "researcher", "outliner",
        "writer", "reviewer", "character_bible", "location_bible",
        "bible_reviewer", "finalize",
    ]:
        assert expected in node_names
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_workflow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph.workflow'`

- [ ] **Step 3: Implement `src/graph/workflow.py`**

```python
# src/graph/workflow.py
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.edges import (
    route_after_bible_reviewer,
    route_after_interview_ask,
    route_after_interview_wait,
    route_after_reviewer,
)
from src.graph.nodes.bible_reviewer import make_bible_reviewer_node
from src.graph.nodes.character_bible import make_character_bible_node
from src.graph.nodes.finalize import make_finalize_node
from src.graph.nodes.ingest import EmbeddingsFn, make_ingest_node
from src.graph.nodes.interviewer import interview_wait_node, make_interview_ask_node
from src.graph.nodes.location_bible import make_location_bible_node
from src.graph.nodes.outliner import make_outliner_node
from src.graph.nodes.researcher import make_researcher_node
from src.graph.nodes.reviewer import make_reviewer_node
from src.graph.nodes.writer import make_writer_node
from src.graph.state import ScreenplayState
from src.tools.search import SearchResult, search


def build_workflow(
    enable_search: bool = False,
    llm: BaseChatModel | None = None,
    embeddings: EmbeddingsFn | None = None,
    search_fn: Callable[[str], list[SearchResult]] = search,
) -> CompiledStateGraph:
    graph = StateGraph(ScreenplayState)

    graph.add_node("ingest", make_ingest_node(embeddings=embeddings))
    graph.add_node("interview_ask", make_interview_ask_node(llm=llm))
    graph.add_node("interview_wait", interview_wait_node)
    graph.add_node("researcher", make_researcher_node(enabled=enable_search, search_fn=search_fn))
    graph.add_node("outliner", make_outliner_node(llm=llm))
    graph.add_node("writer", make_writer_node(llm=llm))
    graph.add_node("reviewer", make_reviewer_node(llm=llm))
    graph.add_node("character_bible", make_character_bible_node(llm=llm))
    graph.add_node("location_bible", make_location_bible_node(llm=llm))
    graph.add_node("bible_reviewer", make_bible_reviewer_node(llm=llm))
    graph.add_node("finalize", make_finalize_node(llm=llm))

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "interview_ask")
    graph.add_conditional_edges(
        "interview_ask",
        route_after_interview_ask,
        {"interview_wait": "interview_wait", "researcher": "researcher"},
    )
    graph.add_conditional_edges(
        "interview_wait",
        route_after_interview_wait,
        {"interview_ask": "interview_ask", "researcher": "researcher"},
    )
    graph.add_edge("researcher", "outliner")
    graph.add_edge("outliner", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer", route_after_reviewer, {"writer": "writer", "character_bible": "character_bible"}
    )
    graph.add_edge("character_bible", "location_bible")
    graph.add_edge("location_bible", "bible_reviewer")
    graph.add_conditional_edges(
        "bible_reviewer",
        route_after_bible_reviewer,
        {"character_bible": "character_bible", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=InMemorySaver())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_workflow.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/graph/workflow.py tests/test_workflow.py
git commit -m "feat: assemble the compiled StateGraph workflow"
```

---

### Task 27: CLI (`main.py`)

**Files:**
- Create: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `build_workflow` (`src.graph.workflow`), `new_initial_state`, `ScreenplayState` (`src.graph.state`), `load_settings` (`src.config`), `Command` (`langgraph.types`).
- Produces: `SUPPORTED_CONTEXT_EXTENSIONS: set[str]`; `STATUS_BADGES: dict[str, str]`; `parse_args(argv=None) -> argparse.Namespace`; `is_finish_command(text: str) -> bool`; `resolve_output_stem(output: str | None) -> Path | None`; `write_outputs(stem: Path, result: ScreenplayState) -> list[Path]`; `print_outputs(result: ScreenplayState) -> None`; `run_interactive(app, initial_state, config) -> ScreenplayState`; `run(argv=None) -> ScreenplayState`.

Streaming badges use LangGraph's documented interrupt-in-stream behavior: with `stream_mode="updates"`, a superstep that hits an `interrupt()` yields a chunk keyed `"__interrupt__"` (the same sentinel `.invoke()` surfaces) instead of a node-name key - `run_interactive` checks for that key on every yielded chunk.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_main.py
from __future__ import annotations

from pathlib import Path

import pytest

from main import (
    is_finish_command,
    parse_args,
    print_outputs,
    resolve_output_stem,
    run_interactive,
    write_outputs,
)


class _FakeInterrupt:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeState:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values


class _FakeApp:
    def __init__(self, stream_batches: list[list[dict[str, object]]], final_values: dict[str, object]) -> None:
        self._batches = list(stream_batches)
        self._final_values = final_values

    def stream(self, current_input, config, stream_mode="updates"):
        batch = self._batches.pop(0)
        yield from batch

    def get_state(self, config):
        return _FakeState(self._final_values)


@pytest.mark.parametrize("text", ["/finish", "/auto", "/autonomous", " /FINISH ", "/Auto"])
def test_is_finish_command_matches_known_phrases(text):
    assert is_finish_command(text) is True


@pytest.mark.parametrize("text", ["go ahead", "no", "", "finish"])
def test_is_finish_command_rejects_other_text(text):
    assert is_finish_command(text) is False


def test_parse_args_requires_topic():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_defaults():
    args = parse_args(["--topic", "A story"])
    assert args.skill == "short_film"
    assert args.files == []
    assert args.max_revisions == 2
    assert args.autonomous is False


def test_parse_args_rejects_unsupported_file_extension(tmp_path):
    bad_file = tmp_path / "image.png"
    bad_file.write_text("not really an image")
    with pytest.raises(SystemExit):
        parse_args(["--topic", "A story", "--files", str(bad_file)])


def test_parse_args_accepts_supported_extensions(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# notes")
    args = parse_args(["--topic", "A story", "--files", str(md_file)])
    assert args.files == [str(md_file)]


def test_resolve_output_stem_none_returns_none():
    assert resolve_output_stem(None) is None


def test_resolve_output_stem_strips_extension():
    assert resolve_output_stem("scripts/my_script.fountain") == Path("scripts/my_script")


def test_resolve_output_stem_no_extension_unchanged():
    assert resolve_output_stem("scripts/my_script") == Path("scripts/my_script")


def test_write_outputs_creates_four_files(tmp_path):
    stem = tmp_path / "my_script"
    result = {
        "fountain_script": "FOUNTAIN TEXT",
        "screenplay_markdown": "MARKDOWN TEXT",
        "character_bible": "CHAR BIBLE",
        "location_bible": "LOC BIBLE",
    }
    written = write_outputs(stem, result)
    assert (tmp_path / "my_script.fountain").read_text() == "FOUNTAIN TEXT"
    assert (tmp_path / "my_script.md").read_text() == "MARKDOWN TEXT"
    assert (tmp_path / "my_script.characters.md").read_text() == "CHAR BIBLE"
    assert (tmp_path / "my_script.locations.md").read_text() == "LOC BIBLE"
    assert len(written) == 4


def test_print_outputs_includes_all_sections(capsys):
    result = {
        "screenplay_markdown": "MARKDOWN TEXT",
        "fountain_script": "FOUNTAIN TEXT",
        "character_bible": "CHAR BIBLE",
        "location_bible": "LOC BIBLE",
    }
    print_outputs(result)
    out = capsys.readouterr().out
    assert "MARKDOWN TEXT" in out
    assert "FOUNTAIN TEXT" in out
    assert "CHAR BIBLE" in out
    assert "LOC BIBLE" in out


def test_run_interactive_handles_one_interrupt_then_completes(monkeypatch, capsys):
    batches = [
        [{"ingest": {}}, {"__interrupt__": (_FakeInterrupt("What tone?"),)}],
        [{"outliner": {}}, {"writer": {}}, {"reviewer": {}}, {"finalize": {}}],
    ]
    app = _FakeApp(batches, final_values={"status": "finalized"})
    monkeypatch.setattr("builtins.input", lambda prompt="": "Melancholy")

    result = run_interactive(app, initial_state={"topic": "t"}, config={"configurable": {"thread_id": "x"}})

    assert result == {"status": "finalized"}
    assert "What tone?" in capsys.readouterr().out


def test_run_interactive_completes_immediately_when_no_interrupt(capsys):
    batches = [[{"ingest": {}}, {"outliner": {}}, {"writer": {}}, {"reviewer": {}}, {"finalize": {}}]]
    app = _FakeApp(batches, final_values={"status": "finalized"})

    result = run_interactive(app, initial_state={"topic": "t"}, config={"configurable": {"thread_id": "x"}})

    assert result == {"status": "finalized"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Implement `main.py`**

```python
# main.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langgraph.types import Command

from src.config import load_settings
from src.graph.state import ScreenplayState, new_initial_state
from src.graph.workflow import build_workflow

SUPPORTED_CONTEXT_EXTENSIONS = {".md", ".txt", ".pdf"}

STATUS_BADGES = {
    "ingest": "[INGESTING]",
    "interview_ask": "[INTERVIEWING]",
    "interview_wait": "[INTERVIEWING]",
    "researcher": "[RESEARCHING]",
    "outliner": "[OUTLINING]",
    "writer": "[WRITING DRAFT]",
    "reviewer": "[REVIEWING]",
    "character_bible": "[BUILDING CHARACTER BIBLE]",
    "location_bible": "[BUILDING LOCATION BIBLE]",
    "bible_reviewer": "[REVIEWING BIBLES]",
    "finalize": "[FINALIZING]",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screenwriter Agent CLI")
    parser.add_argument("--topic", "-t", required=True, help="Premise or prompt")
    parser.add_argument(
        "--skill", "-s", default="short_film",
        choices=["short_film", "documentary", "commercial", "product_shot", "learning_dev", "infomedia"],
    )
    parser.add_argument("--files", "-f", nargs="*", default=[], help="Context files (.md, .txt, .pdf)")
    parser.add_argument("--max-revisions", "-r", type=int, default=2)
    parser.add_argument("--output", "-o", default=None, help="Output file stem (no extension)")
    parser.add_argument("--enable-search", action="store_true")
    parser.add_argument("--autonomous", action="store_true", help="Skip the interview entirely")
    args = parser.parse_args(argv)

    for file_path in args.files:
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_CONTEXT_EXTENSIONS:
            parser.error(
                f"Unsupported file type '{ext}' for {file_path}. "
                f"Supported: {', '.join(sorted(SUPPORTED_CONTEXT_EXTENSIONS))}"
            )
    return args


def is_finish_command(text: str) -> bool:
    return text.strip().lower() in {"/finish", "/auto", "/autonomous"}


def resolve_output_stem(output: str | None) -> Path | None:
    if output is None:
        return None
    path = Path(output)
    return path.with_suffix("") if path.suffix else path


def write_outputs(stem: Path, result: ScreenplayState) -> list[Path]:
    targets = {
        stem.with_suffix(".fountain"): result["fountain_script"],
        stem.with_suffix(".md"): result["screenplay_markdown"],
        Path(f"{stem}.characters.md"): result["character_bible"],
        Path(f"{stem}.locations.md"): result["location_bible"],
    }
    written: list[Path] = []
    for path, content in targets.items():
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def print_outputs(result: ScreenplayState) -> None:
    print("\n" + "=" * 20 + " SCREENPLAY (Markdown) " + "=" * 20)
    print(result["screenplay_markdown"])
    print("\n" + "=" * 20 + " SCREENPLAY (Fountain) " + "=" * 20)
    print(result["fountain_script"])
    print("\n" + "=" * 20 + " CHARACTER BIBLE " + "=" * 20)
    print(result["character_bible"])
    print("\n" + "=" * 20 + " LOCATION BIBLE " + "=" * 20)
    print(result["location_bible"])


def run_interactive(app: Any, initial_state: Any, config: dict[str, Any]) -> ScreenplayState:
    current_input: Any = initial_state
    while True:
        interrupted = False
        for chunk in app.stream(current_input, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                question = chunk["__interrupt__"][0].value
                print(f"[INTERVIEWING] {question}")
                answer = input("> ")
                current_input = Command(
                    resume={"answer": answer, "autonomous": is_finish_command(answer)}
                )
                interrupted = True
                break
            for node_name in chunk:
                print(STATUS_BADGES.get(node_name, f"[{node_name.upper()}]"))
        if not interrupted:
            break
    return app.get_state(config).values  # type: ignore[no-any-return]


def run(argv: list[str] | None = None) -> ScreenplayState:
    args = parse_args(argv)
    load_settings()  # fail fast on invalid numeric env overrides
    app = build_workflow(enable_search=args.enable_search)

    initial_state = new_initial_state(
        topic=args.topic,
        skill=args.skill,
        file_paths=args.files,
        max_revisions=args.max_revisions,
        max_bible_revisions=1,
        autonomous=args.autonomous,
    )
    config = {"configurable": {"thread_id": f"cli-{id(args)}"}}
    result = run_interactive(app, initial_state, config)

    stem = resolve_output_stem(args.output)
    if stem is None:
        print_outputs(result)
    else:
        for path in write_outputs(stem, result):
            print(f"Wrote {path}")
    return result


if __name__ == "__main__":
    run()
    sys.exit(0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add CLI entrypoint with streaming badges and interview interrupt handling"
```

---

### Task 28: E2E Workflow Tests

**Files:**
- Create: `tests/test_workflow_e2e.py`

**Interfaces:**
- Consumes: `build_workflow` (`src.graph.workflow`), `new_initial_state` (`src.graph.state`), `FakeChatModel` (`tests.conftest`), `SearchResult` (`src.tools.search`), `Command` (`langgraph.types`).
- Produces: nothing consumed by later tasks - this is the full-pipeline correctness check the spec's testing section calls for ("Run the compiled graph with a mock LLM providing predetermined responses to verify full pipeline execution").

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_workflow_e2e.py
from __future__ import annotations

from langgraph.types import Command

from tests.conftest import FakeChatModel
from src.graph.state import new_initial_state
from src.graph.workflow import build_workflow
from src.tools.search import SearchResult


def test_full_workflow_autonomous_short_film_produces_all_artifacts():
    responses = [
        "1. Hook\n2. Inciting incident\n3. Climax\n4. Resolution",
        "INT. OFFICE - DAY\n\nJANE stares at the phone.\n\nJANE\nPick up.\n",
        '{"score": 9.0, "passed": true, "critique": "Solid draft.", "actionable_revisions": []}',
        '[{"name": "Jane", "role": "Protagonist", "appearance": "Sharp business attire, tired eyes.", '
        '"personality": "Relentless.", "voice": "Clipped, impatient.", "backstory": "Ex-detective.", '
        '"headshot_prompt": "Create a headshot of Jane, a woman with tired eyes in sharp business attire.", '
        '"contact_sheet_prompt": "Create a 3x2 contact sheet of Jane in six expressions.", '
        '"wardrobe_prompt": "Create a full-body wardrobe shot of Jane in sharp business attire."}]',
        '[{"name": "Office", "description": "A cramped, fluorescent-lit detective office.", '
        '"mood": "Tense and claustrophobic.", '
        '"t2i_prompt": "Create a wide shot of a cramped, fluorescent-lit detective office at night."}]',
        '{"score": 9.0, "passed": true, "critique": "Consistent with the draft.", "actionable_revisions": []}',
        "Create a moody cinematic poster of a detective silhouetted against office blinds.",
    ]
    llm = FakeChatModel(responses=responses)

    def fake_search(query: str) -> list[SearchResult]:
        return [SearchResult(title="Research hit", url="http://example.com", snippet="Useful context")]

    app = build_workflow(enable_search=True, llm=llm, search_fn=fake_search)

    initial_state = new_initial_state(
        topic="A retired detective solves crimes via voicemail",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
        autonomous=True,
    )
    config = {"configurable": {"thread_id": "test-thread-1"}}
    result = app.invoke(initial_state, config)

    assert "__interrupt__" not in result
    assert result["status"] == "finalized"
    assert "INT. OFFICE - DAY" in result["fountain_script"]
    assert "Key Art T2I Prompt" in result["screenplay_markdown"]
    assert "# Character Bible" in result["character_bible"]
    assert "Jane" in result["character_bible"]
    assert "Headshot Prompt" in result["character_bible"]
    assert "# Location Bible" in result["location_bible"]
    assert "Office" in result["location_bible"]


def test_workflow_interview_pauses_for_human_input_then_resumes():
    responses = [
        '{"has_enough_info": false, "question": "What tone should this have?"}',
        '{"has_enough_info": true, "question": ""}',
        "1. Hook\n2. Climax",
        "INT. ROOM - DAY\n\nA phone rings.\n",
        '{"score": 9.0, "passed": true, "critique": "Good.", "actionable_revisions": []}',
        "[]",
        "[]",
        '{"score": 9.0, "passed": true, "critique": "Fine.", "actionable_revisions": []}',
        "Create a minimalist poster of a ringing phone in a dark room.",
    ]
    llm = FakeChatModel(responses=responses)
    app = build_workflow(enable_search=False, llm=llm, search_fn=lambda q: [])

    initial_state = new_initial_state(
        topic="A phone that rings once a year",
        skill="short_film",
        file_paths=[],
        max_revisions=2,
        max_bible_revisions=1,
        autonomous=False,
    )
    config = {"configurable": {"thread_id": "test-thread-2"}}

    paused = app.invoke(initial_state, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value == "What tone should this have?"

    resumed = app.invoke(
        Command(resume={"answer": "Melancholy and quiet.", "autonomous": False}), config
    )

    assert "__interrupt__" not in resumed
    assert resumed["status"] == "finalized"
    assert resumed["interview_transcript"] == [
        {"question": "What tone should this have?", "answer": "Melancholy and quiet."}
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_workflow_e2e.py -v`
Expected: FAIL - either `ModuleNotFoundError` if any prior task's module is missing, or an assertion error if the graph doesn't wire together as intended. (By this point in the plan every module it imports already exists from Tasks 1-26, so this file is the first genuinely new failure surface - proceed straight to running it against the already-implemented graph.)

- [ ] **Step 3: Run the tests again after confirming all imports resolve**

Run: `pytest tests/test_workflow_e2e.py -v`
Expected: PASS (2 tests). If either test fails, apply the systematic-debugging skill: the likely culprits are a mismatched `FakeChatModel` response ordering (recount which node calls `.invoke()` in which order) or a routing edge in `src/graph/edges.py` disagreeing with this test's `max_interview_questions`/`review_pass_score` defaults (`load_settings()` defaults: cap 5, pass score 8.0 - both scripted responses here score 9.0 and finish the interview on turn 2, well within those defaults).

- [ ] **Step 4: Commit**

```bash
git add tests/test_workflow_e2e.py
git commit -m "test: add full-pipeline e2e tests covering autonomous run and interview interrupt/resume"
```

---

### Task 29: README + Final Verification

**Files:**
- Create: `README.md`
- Modify: none (verification only)

**Interfaces:**
- Produces: project documentation. No code interfaces - this is the closing task.

- [ ] **Step 1: Write `README.md`**

```markdown
# Screenwriter Agent CLI

A LangGraph-orchestrated CLI that turns a topic plus optional `.md`/`.txt`/`.pdf`
context into a reviewed screenplay, a character bible, and a location bible -
each production bible entry includes copy-paste [Nano Banana Pro](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-nano-banana)
T2I prompts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set OLLAMA_API_KEY to your Ollama cloud API key
```

`OLLAMA_API_KEY` is required - this project calls Ollama's cloud API
(`https://ollama.com`) exclusively; no local Ollama daemon is used. Get a key
from your Ollama account, then `pip install`'s `nomic-embed-text` embedding
model and the configured chat model (default `deepseek-v4.1-flash:cloud`)
must both be available to that key.

`TAVILY_API_KEY` is optional - without it, `--enable-search` falls back to
keyless DuckDuckGo search automatically.

## Usage

```bash
# Minimal run, autonomous (skips the interview), prints all four artifacts to stdout
python main.py --topic "A retired detective solves crimes via voicemail" --skill short_film --autonomous

# Interactive interview, writes four files: my_script.{fountain,md,characters.md,locations.md}
python main.py --topic "A retired detective solves crimes via voicemail" \
  --skill short_film --output my_script

# With context files and research enabled
python main.py --topic "Product launch spot" --skill commercial \
  --files brand_brief.pdf notes.md --enable-search --output launch_spot

# Documentary, more revision headroom
python main.py --topic "The last lighthouse keeper" --skill documentary \
  --max-revisions 3 --output lighthouse

# Product shot spec ad
python main.py --topic "A watch that survives anything" --skill product_shot --output watch_ad

# Corporate L&D training module
python main.py --topic "New-hire security awareness training" --skill learning_dev --output security_training

# Explainer / infomedia video
python main.py --topic "How compound interest works" --skill infomedia --output compound_interest
```

During the interview phase, answer each question directly, or type `/finish`
(or `/auto`) at any prompt to skip the rest of the interview and let the
writer proceed autonomously with what it already knows.

## Output files

Given `--output <stem>`, every run writes exactly four files:

| File | Contents |
|---|---|
| `<stem>.fountain` | Strict, importable Fountain syntax - works for all six skills (Dual-Column skills map into Fountain conventions: `NARRATOR`/`VO` cues, bracketed SFX/OST notes). |
| `<stem>.md` | Human-readable screenplay: Master Scene prose (`short_film`) or the Dual-Column A/V table (the other five skills), plus the key-art T2I prompt. |
| `<stem>.characters.md` | Character bible - one section per principal character, with Headshot / Contact Sheet / Wardrobe & Accessories T2I prompt boxes. |
| `<stem>.locations.md` | Location bible - one section per location, with a T2I prompt box each. |

Without `--output`, all four are printed to stdout instead.

## Skills

`short_film`, `documentary`, `commercial`, `product_shot`, `learning_dev`, `infomedia` - see `src/skills/` for each skill's persona, outline structure, and review checklist.

## Testing

```bash
pytest -v --cov=src
ruff check src tests main.py
```

All tests run 100% offline - no network calls are made to Ollama, Tavily,
DuckDuckGo, or Chroma during `pytest`.
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest -v --cov=src`
Expected: all tests across every module PASS, 0 failures. If anything fails, apply the systematic-debugging skill before proceeding - do not skip or delete a failing test to make the suite green.

- [ ] **Step 3: Run the linter**

Run: `ruff check src tests main.py`
Expected: no findings. Run `ruff check --fix src tests main.py` for any auto-fixable issues (import ordering, unused imports), then re-run `ruff check src tests main.py` to confirm a clean pass.

- [ ] **Step 4: Run the format check**

Run: `ruff format --check src tests main.py`
Expected: no reformatting needed. If it reports files needing changes, run `ruff format src tests main.py` and re-run the full test suite (Step 2) to confirm formatting changes didn't break anything.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, usage, and output-file reference"
```

---

## Self-Review Notes

- **Spec coverage:** every numbered section of `docs/superpowers/specs/2026-09-19-screenwriter-cli-design.md` maps to at least one task - §2/§4 config → Task 2; §5 parsers/RAG → Tasks 4-7; §6 skills → Tasks 9-10; §8 nodes/routing → Tasks 15-26; §9 output artifacts → Tasks 24, 27; §10 CLI → Task 27; §11 testing → Tasks throughout plus Task 28; §12 addenda (two-node interviewer, `InMemorySaver`, `AVBeat`, shared JSON utils) → Tasks 3, 11-12, 16, 26.
- **Placeholder scan:** no `TBD`/`TODO` remains; the one `NotImplementedError` (Task 4, Step 5) is a deliberate, immediately-superseded stub documented as such so `parse_context_files`'s lazy import resolves before Task 5 lands, not an unfinished design decision.
- **Type consistency:** `NodeUpdate = dict[str, Any]` (Task 14) is used identically as every node factory's return type from Task 15 onward; `ScreenplayState` field names introduced in Task 14 (`rag_run_id`, `pending_question`, `bible_revision_count`, etc.) are the exact keys every node reads/writes in Tasks 15-24 and every test constructs via `new_initial_state`/`state.update(...)`; `AVBeat` (Task 11) is the same model `fountain.py` (Task 12) and `finalize.py` (Task 24) both import, never redefined.

