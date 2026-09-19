# Screenwriter Agent CLI — Design Spec

Date: 2026-09-19
Status: Approved pending final user sign-off
Project root: `L:\AIStudio\ScreenWriter`

## 1. Overview

A Python 3.10+ CLI that takes a topic/premise plus optional text context
(`.md`, `.txt`, `.pdf`), interviews the user for missing creative direction,
runs an iterative Writer → Reviewer critique loop over LangGraph, and
produces:

1. A Fountain-compatible screenplay file.
2. A human-readable screenplay + key-art T2I prompt file.
3. A character bible file (principal cast, each with copy-paste T2I prompts).
4. A location bible file (each location with a copy-paste T2I prompt).

Six domain "skills" (short_film, documentary, commercial, product_shot,
learning_dev, infomedia) select the writing persona, outline structure,
and review checklist. Long/multi-document context is handled via an
ephemeral, in-memory RAG layer so users can drop in book-length source
material. All T2I prompts target **Nano Banana Pro** (Gemini image
generation) and are meant to be copy-pasted by the user — this project
never calls an image-generation API itself.

Everything runs against **Ollama's cloud API** only — no local Ollama
install/daemon is required, no OpenAI, no image-file ingestion.

## 2. Tech stack

| Layer | Specification |
|---|---|
| Language | Python 3.10+, `from __future__ import annotations`, `TypedDict`, Pydantic v2, strict type annotations everywhere |
| Orchestration | LangGraph (`StateGraph`, `START`/`END`, conditional edges, `interrupt()` + `MemorySaver` checkpointer for the interview loop) + `langchain-core` |
| LLM | `langchain-ollama` `ChatOllama` only, pointed at Ollama's cloud API (`OLLAMA_BASE_URL` + `OLLAMA_API_KEY`) — no dual-provider switch, no local daemon required |
| Embeddings | `langchain-ollama` `OllamaEmbeddings`, default model `nomic-embed-text`, same cloud endpoint/auth |
| Vector store | `chromadb` `EphemeralClient()` via `langchain-chroma`, one in-memory collection per run, nothing persisted to disk |
| Search | `tavily-python` primary, keyless `duckduckgo-search` (`ddgs`) fallback |
| Parsers | `pypdf` for `.pdf`, stdlib for `.md`/`.txt` — no image parsing |
| Config | `python-dotenv`, single source of truth in `src/config.py` |
| Testing | `pytest`, 100% offline — every `ChatOllama`, `OllamaEmbeddings`, Chroma, Tavily, and DDGS call is mocked |
| CLI | stdlib `argparse` + LangGraph `graph.stream()` for live status badges |

New dependencies vs. the original spec: `chromadb`, `langchain-chroma`.
Removed: `langchain-openai`, any image-handling library.

## 3. File tree

```text
screenwriter/
├── .env.example
├── pyproject.toml              # metadata + ruff/pytest config only, no build backend
├── requirements.txt
├── README.md
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py                 # Ollama-only LLM + embeddings factory, RAG/interview/T2I settings
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── text_parser.py        # .md, .txt
│   │   ├── pdf_parser.py         # .pdf via pypdf
│   │   └── chunker.py            # fixed-size + overlap chunking for RAG
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── store.py              # Chroma ephemeral client wrapper
│   │   └── retriever.py          # retrieve_context(query, k) -> list[str]
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── base.py               # SkillProfile interface + shared T2I prompt guidelines + Master Scene rules
│   │   ├── short_film.py
│   │   ├── documentary.py
│   │   ├── commercial.py
│   │   ├── product_shot.py
│   │   ├── learning_dev.py
│   │   └── infomedia.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search.py             # Tavily -> DDGS fallback (unchanged)
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py               # ScreenplayState (expanded, no image_data)
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py          # parse + chunk + index into ephemeral vector store
│   │   │   ├── interviewer.py     # human-in-the-loop clarifying Q&A (interrupt())
│   │   │   ├── researcher.py      # web search
│   │   │   ├── outliner.py        # RAG-aware beat sheet
│   │   │   ├── writer.py          # RAG-aware drafting
│   │   │   ├── reviewer.py        # screenplay critique loop
│   │   │   ├── character_bible.py # curated principal cast + T2I prompts
│   │   │   ├── location_bible.py  # locations + T2I prompts
│   │   │   ├── bible_reviewer.py  # combined lighter review pass for both bibles
│   │   │   └── finalize.py        # renders all 4 output artifacts, incl. key-art T2I prompt
│   │   ├── edges.py                # interview loop, revise loop, bible-revise loop routing
│   │   └── workflow.py             # StateGraph builder, compiled with MemorySaver
│   └── formatters/
│       ├── __init__.py
│       ├── fountain.py             # strict Fountain syntax renderer/validator, all 6 skills
│       ├── dual_column.py          # A/V Markdown table renderer (5 production skills)
│       └── bible.py                # character/location bible + T2I prompt-box renderer
└── tests/
    ├── __init__.py
    ├── conftest.py                 # fixtures, fake ChatOllama/OllamaEmbeddings/Chroma/search doubles
    ├── test_config.py
    ├── test_parsers.py
    ├── test_chunker.py
    ├── test_rag.py
    ├── test_skills.py
    ├── test_search_fallback.py
    ├── test_formatters.py
    ├── test_graph_nodes.py
    ├── test_interview_node.py
    └── test_workflow_e2e.py
```

## 4. Config (`src/config.py`)

- `get_llm() -> ChatOllama`: reads `OLLAMA_MODEL_NAME` (default
  `deepseek-v4.1-flash:cloud`), `OLLAMA_BASE_URL` (default
  `https://ollama.com`, Ollama's hosted cloud API — not a local
  daemon), and `OLLAMA_API_KEY` (required, sent as bearer-token auth;
  exact `ChatOllama`/`OllamaEmbeddings` client parameter for this —
  `headers`, `client_kwargs`, or similar — to be confirmed against the
  current `langchain-ollama` release during implementation). Raises a
  clear config error at startup if `OLLAMA_API_KEY` is unset.
- `get_embeddings() -> OllamaEmbeddings`: reads `OLLAMA_EMBED_MODEL`
  (default `nomic-embed-text`), same cloud base URL/API key.
- RAG settings: `RAG_CHUNK_SIZE` (1000), `RAG_CHUNK_OVERLAP` (150),
  `RAG_TOP_K` (5), `RAG_MIN_CHARS_TO_INDEX` (threshold below which
  `parsed_context` is used directly instead of retrieval — avoids
  pointless embedding calls for a one-page brief).
- Interview settings: `MAX_INTERVIEW_QUESTIONS` (default 5).
- Review settings: `REVIEW_PASS_SCORE` (8.0, existing), `max_revisions`
  (CLI flag, default 2), `MAX_BIBLE_REVISIONS` (default 1).
- Search settings: `TAVILY_API_KEY` (optional).
- No multimodal detection, no `OPENAI_*` vars, no image constants.

`.env.example`:
```
OLLAMA_MODEL_NAME=deepseek-v4.1-flash:cloud
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=
OLLAMA_EMBED_MODEL=nomic-embed-text
TAVILY_API_KEY=
```

## 5. Parsers & RAG

- `text_parser.py` / `pdf_parser.py`: unchanged from the original spec
  (structure-preserving markdown ingestion, page-referenced PDF text
  extraction). `parse_context_files(file_paths) -> ParsedContext`
  rejects any non-`.md`/`.txt`/`.pdf` extension with a clear error
  rather than silently ignoring it.
- `chunker.py`: splits `ParsedContext.full_text` into overlapping
  chunks, each tagged with `{source_file, chunk_index}` metadata.
- `rag/store.py`: wraps `chromadb.EphemeralClient()`; one collection
  per run (`collection_name = f"run-{uuid4()}"`), destroyed with the
  process — nothing touches disk.
- `rag/retriever.py`: `retrieve_context(query: str, k: int) -> list[str]`,
  used by `outliner_node` (per outline beat) and `writer_node` (per
  scene/section being drafted) whenever `file_paths` is non-empty and
  content exceeds `RAG_MIN_CHARS_TO_INDEX`. Below that threshold,
  nodes just use `parsed_context` wholesale.

## 6. Skills (`src/skills/`)

Same six `SkillProfile` implementations as originally specced
(short_film, documentary, commercial, product_shot, learning_dev,
infomedia), each still defining `system_prompt`, `output_format`
(`fountain` or `dual_column` — this now only controls which secondary
Markdown rendering is produced; **every** skill also always renders to
strict Fountain, see §8), `outline_template`, `review_criteria`. No
image-related fields.

`skills/base.py` additionally exports:
- Master Scene Format rules as a shared prompt fragment (sluglines
  `INT./EXT. LOCATION - TIME`, ALL-CAPS character cues, extensions
  `(V.O.)`/`(O.S.)`/`(CONT'D)`, sparing lowercase parentheticals,
  right-aligned ALL-CAPS transitions ending `TO:`, action blocks capped
  at 4 lines) — reused by every skill's `system_prompt` and by
  `formatters/fountain.py`'s validator.
- `T2I_PROMPT_GUIDELINES`, a shared prompt fragment derived from
  Google's Nano Banana Pro prompting guide, reused by
  `character_bible_node`, `location_bible_node`, and the key-art step
  in `finalize_node`:
  - Formula: `[Subject] + [Action] + [Location/context] + [Composition] + [Style]`.
  - Full narrative sentences, never bare keyword lists.
  - Positive framing only (describe what should be present, never
    "no X").
  - Open with a strong verb ("Create a...", "Generate an image of...").
  - Concrete materials/textures, not generic nouns ("navy blue tweed
    suit jacket", not "a suit").
  - Explicit camera/lens/lighting/color-grade descriptors (e.g. "shot
    on a Fujifilm camera, shallow depth of field f/1.8, three-point
    softbox lighting").
  - Any on-screen text goes in quotes with font/style specified.
  - **Consistency without reference images**: since this pipeline is
    text-only (no image attachments), character likeness consistency
    across a character's Headshot / Contact Sheet / Wardrobe prompts —
    and between the bible and any later scene art the user generates —
    is achieved by defining one canonical physical-description clause
    per character (fixed distinguishing features: face, hair, build,
    signature color/prop) and repeating it **verbatim** in all of that
    character's T2I prompt boxes.
  - Suggested aspect ratios baked into prompt templates: Headshot 4:5,
    Contact Sheet 1:1 ("3x2 grid of six expressions/angles"), Wardrobe
    & Accessories 3:4 full body, Location 16:9 or 21:9, screenplay key
    art 2:3.

## 7. Graph state (`src/graph/state.py`)

```python
class InterviewTurn(TypedDict):
    question: str
    answer: str

class ScreenplayState(TypedDict):
    topic: str
    skill: str
    file_paths: list[str]
    parsed_context: str
    rag_indexed: bool
    research_notes: str
    interview_transcript: list[InterviewTurn]
    interview_turn_count: int
    interview_complete: bool
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
```

(No `image_data` field — removed entirely.)

## 8. Graph nodes & routing

```
START -> ingest -> interviewer ⇄(loop) -> researcher -> outliner -> writer ⇄ reviewer
      -> character_bible + location_bible ⇄ bible_reviewer -> finalize -> END
```

- **ingest_node**: parses files, chunks + indexes into the ephemeral
  Chroma collection when applicable. Never crashes on a corrupt file —
  flags it in `parsed_context` and continues.
- **interviewer_node**: LLM call returns structured
  `{"question": str, "has_enough_info": bool}`. If not enough info and
  `interview_turn_count < MAX_INTERVIEW_QUESTIONS`, calls `interrupt(question)`
  to pause for a human answer, appends the turn, increments the
  counter, loops. Exits the loop (sets `interview_complete=True`) when
  the LLM signals enough info, the turn cap is hit, or the resumed
  value carries `autonomous=True` (the user typed `/finish` or
  `/auto` in the terminal).
- **researcher_node**: unchanged — Tavily → DDGS fallback, only runs
  when `--enable-search` is passed.
- **outliner_node** / **writer_node**: unchanged in role, now pull
  grounding chunks via `rag/retriever.py` instead of a flat context
  string when RAG is active.
- **reviewer_node**: unchanged mechanics (structured JSON score/critique,
  regex-extraction with a graceful fallback on parse failure).
- **character_bible_node**: one structured-output LLM call over the
  final draft; returns a curated list of principal cast (LLM judges
  significance, not "every speaking role"), each with prose bible
  fields (appearance, personality, voice, backstory snippet) plus
  three T2I prompt boxes (Headshot, Contact Sheet, Wardrobe &
  Accessories) per §6 guidelines.
- **location_bible_node**: same pattern, one T2I prompt box per key
  location.
- **bible_reviewer_node**: single combined review pass over both
  bibles (consistency with the script, completeness of required
  fields); routes back to both bible nodes together with feedback, or
  forward once `bible_review_score >= REVIEW_PASS_SCORE` or
  `bible_revision_count >= MAX_BIBLE_REVISIONS`.
- **finalize_node**: generates the one key-art T2I prompt box for the
  screenplay itself, then renders all four output artifacts (§9).

`edges.py` centralizes the three conditional loops (interview,
screenplay revise, bible revise), each with an explicit terminal-limit
exit per the existing "no infinite token consumption" guardrail in
`CLAUDE.md`.

## 9. Output artifacts

Given `--output <stem>` (or a path — the extension, if any, is
stripped to derive the stem; default stem `screenplay` printed to
stdout as 4 clearly delimited sections if `--output` is omitted),
`finalize_node` / `main.py` always write exactly four files:

1. **`<stem>.fountain`** — strict, importable Fountain syntax, for
   **all six skills**. For the five Dual-Column skills, content maps
   into Fountain conventions: action lines for VISUAL description,
   character cues (`NARRATOR`, `VO`, `PRESENTER`) for AUDIO/narration
   lines, bracketed action-line notes for SFX/OST/graphics cues
   (e.g. `[[SFX: whoosh]]`, `[[ON SCREEN TEXT: "50% OFF"]]`). No
   tables, no T2I prompts, no bibles — this file must parse cleanly in
   Fountain-compatible software (Highland, Fade In, etc.).
2. **`<stem>.md`** — human-readable screenplay: Master Scene prose
   (short_film) or the Dual-Column A/V Markdown table (the other five
   skills), plus the single key-art T2I prompt box at the top.
3. **`<stem>.characters.md`** — character bible: one section per
   principal character, prose fields + 3 fenced-code T2I prompt boxes
   each.
4. **`<stem>.locations.md`** — location bible: one section per
   location, prose fields + 1 fenced-code T2I prompt box each.

`formatters/fountain.py` owns file 1 and enforces the Master Scene
rules as a validator (warns, doesn't crash, on violations like a
5-line action block or a lowercase slugline). `formatters/dual_column.py`
owns the table portion of file 2 for the five production skills.
`formatters/bible.py` owns files 3 and 4 and the key-art box in file 2.

## 10. CLI (`main.py`)

Flags (supersedes original spec):
- `--topic` / `-t` (required)
- `--skill` / `-s` (choices: the 6 skills; default `short_film`)
- `--files` / `-f` (nargs `*`; `.md`/`.txt`/`.pdf` only — any other
  extension is a hard argparse-time error, not a silent skip)
- `--max-revisions` / `-r` (default 2)
- `--output` / `-o` (stem/path; default: print all 4 sections to stdout)
- `--enable-search` (unchanged)
- `--autonomous` (skip the interview entirely from the start — same
  effect as typing `/finish` on turn 1)

Interview loop mechanics: `main.py` builds the graph with a
`MemorySaver` checkpointer and a per-run `thread_id`. It calls
`graph.invoke(initial_state, config)`; whenever the result carries an
interrupt payload, it prints the interviewer's question, reads one
line via `input()`, and resumes with
`graph.invoke(Command(resume={"answer": answer, "autonomous": is_finish_command(answer)}), config)`.
`is_finish_command` matches literal `/finish` or `/auto` (case
-insensitive), nothing fuzzier — deterministic and easy to test.

Streaming status badges via `graph.stream()` cover every node:
`[INGESTING]`, `[INTERVIEWING]`, `[RESEARCHING]`, `[OUTLINING]`,
`[WRITING DRAFT]`, `[REVIEWING (Score: X/10)]`, `[REVISING]`,
`[BUILDING CHARACTER BIBLE]`, `[BUILDING LOCATION BIBLE]`,
`[REVIEWING BIBLES (Score: X/10)]`, `[FINALIZING]`.

## 11. Testing strategy

All existing guarantees from the original spec carry over (no live
network/model calls; mock `ChatOllama`, `OllamaEmbeddings`, Chroma,
Tavily, DDGS). New coverage:

- `test_chunker.py`: chunk boundaries/overlap correctness.
- `test_rag.py`: store add/query against a fake embeddings function
  (deterministic vectors), retriever top-k behavior, and the
  below-threshold bypass path.
- `test_interview_node.py`: drives `interviewer_node` directly with a
  scripted fake LLM and a stubbed `interrupt`/resume cycle — verifies
  the turn-count cap, the `has_enough_info` exit, and the
  `autonomous=True` short-circuit, all without a real LangGraph
  checkpointer's I/O.
- `test_graph_nodes.py`: extended with `character_bible_node`,
  `location_bible_node`, `bible_reviewer_node`, `finalize_node`
  (asserting all 4 output artifacts render, and that the Fountain
  output for a Dual-Column skill contains no raw table syntax).
- `test_workflow_e2e.py`: full compiled graph, scripted fake LLM
  responses for every node including a scripted interview exchange,
  asserting the four output files/sections are all produced with
  expected structure.

## 12. Implementation addenda (verified during plan-writing)

Discovered while confirming exact library APIs before writing the
implementation plan — these refine, not replace, the sections above:

- **Interviewer is two nodes, not one.** LangGraph re-executes a node's
  entire function body from the top on resume; only the specific
  `interrupt()` call site returns its cached value without re-pausing.
  A single node that calls the LLM to decide the question *and then*
  calls `interrupt()` would silently re-run that LLM call on resume,
  risking a mismatch between the question the user actually saw and
  the one recorded in the transcript. Split into `interview_ask_node`
  (LLM call, decides `has_enough_info`/`question`, writes
  `pending_question` to state, no `interrupt()`) and
  `interview_wait_node` (only calls `interrupt(state["pending_question"])`
  and records the answer — cheap and safe to re-execute). `pending_question: str`
  is added to `ScreenplayState`. Routing loops `interview_ask_node ⇄
  interview_wait_node` until exit (enough info / autonomous / turn cap),
  then proceeds to `researcher`.
- **Checkpointer class name**: `from langgraph.checkpoint.memory import InMemorySaver`
  (current LangGraph, not the older `MemorySaver` name).
- **Cloud auth mechanism confirmed**: the underlying `ollama` Python
  client takes `Client(host=..., headers={"Authorization": f"Bearer {key}"})`;
  `langchain-ollama`'s `ChatOllama`/`OllamaEmbeddings` forward extra
  client options via a `client_kwargs` field. The plan includes an
  explicit verification step (inspect the installed package's
  constructor signature) before relying on this, since it's the one
  piece not pinned by first-party docs snippets.
- **Structured draft representation**: `writer_node` produces
  `state["draft"]` as plain Fountain-ready prose for `short_film`
  (output_format `fountain`), but as a JSON array of `AVBeat {timecode,
  visual, audio}` objects (as text) for the five Dual-Column skills.
  Both `formatters/dual_column.py` (table) and `formatters/fountain.py`
  (action lines + `NARRATOR`/`VO` cues + bracketed SFX/OST notes) render
  from the same parsed `list[AVBeat]` — this is what makes "Fountain for
  all six skills" (§9) and the original Dual-Column table both
  achievable from one draft without duplicating writer logic.
- **Shared JSON-extraction utility**: `src/utils.py` centralizes the
  robust-parse-with-fallback logic (`extract_json_object`,
  `extract_json_array`, regex-scoped + Pydantic-validated) reused by
  `reviewer_node`, `bible_reviewer_node`, `interview_ask_node`,
  `character_bible_node`, `location_bible_node`, and
  `formatters/dual_column.py`'s `parse_av_beats`.

## 13. Explicit non-goals

- No image-file ingestion or vision-model support (removed entirely).
- No OpenAI / dual-provider support — Ollama only.
- No actual call to an image-generation API — T2I prompts are text for
  the user to copy into Nano Banana Pro manually.
- No persistence of the RAG vector store across runs — ephemeral,
  in-memory, per-invocation only.
