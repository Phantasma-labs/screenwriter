# Project Guide: Screenwriter Agent CLI

A LangGraph-orchestrated CLI that turns a topic plus optional `.md`/`.txt`/`.pdf`
context files into a reviewed screenplay, a character bible, and a location
bible, each with copy-paste Nano Banana Pro T2I prompts. See `README.md` for
end-user setup/usage; this file is for anyone changing the code.

## Development Commands
* Setup environment: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
  (Windows: `.venv\Scripts\activate`)
* Configure: `cp .env.example .env` and set `OLLAMA_API_KEY` (Ollama cloud API key; `TAVILY_API_KEY` is optional)
* Run the CLI: `python main.py --topic "A retired detective solves crimes via voicemail" --skill short_film --autonomous`
* Run with file context: `python main.py --topic "Product launch spot" --skill commercial --files brand_brief.pdf notes.md`
* Run full test suite: `pytest`
* Run a single test file: `pytest tests/<file_name>.py`
* Run tests with verbose output & coverage: `pytest -v --cov=src`
* Lint and format check: `ruff check src tests main.py && ruff format --check src tests main.py`
* Fix lint/format errors: `ruff check --fix src tests main.py && ruff format src tests main.py`

## Tech Stack & Architecture
* Core language & runtime: Python 3.10+ (strict typing, `from __future__ import annotations`, `TypedDict`, Pydantic v2)
* Agent orchestration: LangGraph (`StateGraph`, `START`/`END`, conditional revision edges, `InMemorySaver` checkpointer for interrupt/resume) + `langchain-core`
* LLM transport: `src/config.py` is the single point of contact with the model provider. Chat/generation (`ChatOllama`) talks to Ollama's **cloud** API (`https://ollama.com`) only, and every call carries the `OLLAMA_API_KEY` bearer token from `.env`. Embeddings (`OllamaEmbeddings`) instead talk to a **local** Ollama daemon (`OLLAMA_EMBED_BASE_URL`, default `http://localhost:11434`) and need no API key - some cloud accounts lack `/api/embed` access, and a local daemon sidesteps that. There is still no OpenAI/other-provider switch anywhere in the project; both clients remain Ollama-only, just pointed at two different endpoints.
* Search & context: primary Tavily (`tavily-python`), automatic keyless fallback to DuckDuckGo (`ddgs`) when no `TAVILY_API_KEY` is set or Tavily fails
* File parsers: `pypdf` (PDF extraction) and a UTF-8 text parser (`.md`, `.txt`) - `--files` accepts only `.md`/`.txt`/`.pdf`; any other extension (including images) is a hard `argparse` error at the CLI boundary. There is no image ingestion anywhere in this project.
* RAG: ephemeral, in-process ChromaDB (`chromadb.EphemeralClient`, ad hoc collections, no persistence) via `src/rag/store.py` and `src/rag/retriever.py` - not `langchain-chroma`. Ingested context above `RAG_MIN_CHARS_TO_INDEX` is chunked and indexed per-run; below that threshold it's passed to the LLM directly instead. If indexing fails for any reason (e.g. the local embeddings daemon isn't running), `ingest` falls back to passing context directly to the model with an inline `[INGEST WARNINGS]` note instead of crashing.
* Configuration: `python-dotenv`, strictly centralized in `src/config.py` (`Settings` dataclass + `load_settings()`); no `os.getenv` scattered in business logic
* CLI & streaming: stdlib `argparse` + LangGraph `app.stream(..., stream_mode="updates")` event iteration in `main.py`, with `langgraph.types.interrupt`/`Command(resume=...)` driving the interview pause/resume loop

## Pipeline (LangGraph nodes)
`src/graph/workflow.py` wires 14 nodes (12 pipeline stages; the interview
and overview-discussion stages are each split across two nodes for their
interrupt boundaries):

`ingest -> interview_ask -> [interview_wait <-> interview_ask]* -> researcher -> outliner -> writer -> reviewer -> [writer <-> reviewer]* -> overview -> [overview_discussion_wait <-> overview_revise]* -> character_bible -> location_bible -> bible_reviewer -> [character_bible <-> bible_reviewer]* -> finalize`

* `ingest`: parses `--files`, surfaces parser warnings inline in `parsed_context`, and conditionally indexes into the ephemeral RAG store.
* `interview_ask` / `interview_wait`: asks up to `MAX_INTERVIEW_QUESTIONS` questions; `interview_wait` is the node that actually calls `interrupt()` and blocks for CLI input, so `--autonomous` (or a `/finish` reply) can short-circuit the loop from `interview_ask` without ever hitting the interrupt boundary.
* `researcher`: optional Tavily/DuckDuckGo web research when `--enable-search` is passed.
* `outliner` / `writer` / `reviewer`: actor-critic loop - `writer` only increments `revision_count` when it's redrafting against `review_feedback`; `reviewer` never touches the counter. Loops until `review_score >= REVIEW_PASS_SCORE` or `revision_count >= max_revisions`.
* `overview`: single one-shot LLM call once the draft passes review, producing a pre-result story description plus production tech specs (duration, frame format, aspect ratio, camera, lenses). Non-autonomous runs then pause at `overview_discussion_wait` for free-form human feedback; `overview_revise` regenerates the overview against that feedback and loops back to `overview_discussion_wait` until the user accepts (no revision cap - the human decides when to stop), then the pipeline proceeds to `character_bible`. Autonomous runs skip straight from `overview` to `character_bible`. Whatever `state["overview"]` holds when the loop exits is written to `Overview.md`.
* `character_bible` / `location_bible` / `bible_reviewer`: a second, symmetric actor-critic loop over the production bibles, gated by `bible_review_score`/`MAX_BIBLE_REVISIONS` the same way.
* `finalize`: renders the Fountain script (`Script.md`) and screenplay Markdown (`Screenplay.md`) (falling back to the raw draft with a note if a Dual-Column skill's A/V beats fail to parse), runs `validate_fountain` and appends any formatting warnings, and writes the key-art T2I prompt.

## Skills
Six `SkillProfile`s in `src/skills/`: `short_film` (Master Scene Fountain
output) and `documentary`, `commercial`, `product_shot`, `learning_dev`,
`infomedia` (Dual-Column beat-table output: Timecode / First Frame Image /
Last Frame Image / I2V Prompt / Description / Narration / Technical per
`AVBeat` in `src/formatters/dual_column.py` - First Frame Image is the
literal T2I-promptable shot the beat opens on; Last Frame Image is an
optional second T2I prompt for the shot's ending frame, written only when
the shot's visual state changes meaningfully within its duration; I2V Prompt
is the image-to-video motion direction, written FFLF-style when a Last Frame
Image is present or FF-style when it isn't; Description is broader
scene/action context, Narration is spoken/V.O. audio, Technical is
camera/lens/transition notes; Screenplay.md renders a compact table plus a
per-beat card with the full T2I/I2V prompts, while Script.md maps only the
First Frame Image into Fountain conventions). Each skill defines its own
system prompt (including realistic per-format shot-pacing guidance),
outline structure, and review checklist.

## Output Files
Given `--output <dir>`, a run writes exactly five fixed-named files into that
directory (created if missing): `Overview.md` (story description + tech
specs), `Script.md` (the Fountain-formatted script), `Screenplay.md` (key-art
prompt + body + formatting warnings), `CharacterBible.md`, `LocationBible.md`.
Without `--output`, all five print to stdout. File names are always these
five literals - `--output` only changes which directory they land in, so a
second run with a different `--output` value doesn't overwrite the first.

The webapp (`app.py`) mirrors these as tabs (Overview, Script, Screenplay,
Character Bible, Location Bible, plus a References tab listing
`search_results` links from the `researcher` node) and offers one "Download
MDs" button on the Overview tab that zips all five files together
(`src/webapp/archive.py`) rather than per-file download buttons.

## Code Style & Guardrails
* Workflow: TDD - write the failing test first, then the implementation. Confirm baseline `pytest` status before making changes.
* Imports: every file starts with `from __future__ import annotations`. Group imports: standard library, third-party, internal (`ruff`'s `I` isort rule enforces this).
* Formatting: Ruff, line length 100 (`[tool.ruff]` in `pyproject.toml`, `select = ["E", "F", "I", "UP"]`).
* Types: strict annotations on all function signatures and return types. `TypedDict` for the LangGraph global state (`ScreenplayState`); Pydantic v2 `BaseModel` for structured LLM outputs (e.g. `ReviewFeedback`). No bare `Any` or unparameterized `dict`/`list` outside `NodeUpdate` (intentionally `dict[str, Any]` - see its docstring in `src/graph/state.py`).
* State management: LangGraph node functions return partial-update dicts (`NodeUpdate`), never mutate state in place. A node only ever writes the counter it owns (e.g. `writer` owns `revision_count`, `character_bible` owns `bible_revision_count`) and omits the key entirely on a non-revision pass rather than writing it as `0`. Revision loops always have a `max_*` cap checked in `src/graph/edges.py`.
* Error handling: file ingestion is defensive - unreadable/corrupt files produce a `warnings` entry on `ParsedContext` rather than raising, and `ingest_node` surfaces those warnings into `parsed_context` so a failed parse is visible to the user, not silently dropped. Search failures downgrade Tavily -> DuckDuckGo -> an empty result list with a warning, never a crash.
* Testing: all tests run 100% offline - no live calls to Ollama, Tavily, DuckDuckGo, or Chroma during `pytest`. LLM calls are doubled with the `FakeChatModel` in `tests/conftest.py`; embeddings with light local fakes.
