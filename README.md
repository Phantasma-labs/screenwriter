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

`OLLAMA_API_KEY` is required for chat/generation - this project calls Ollama's
cloud API (`https://ollama.com`) for the configured chat model (default
`deepseek-v4.1-flash:cloud`). Get a key from your Ollama account and make sure
that model is available to it.

Embeddings (used only for RAG indexing of uploaded `--files` context once it
exceeds `RAG_MIN_CHARS_TO_INDEX`) use a **local** Ollama daemon instead of the
cloud API - no API key needed, and it sidesteps cloud accounts that lack
`/api/embed` access. This is optional: install [Ollama](https://ollama.com/download)
locally, run `ollama pull nomic-embed-text`, and make sure the daemon is
running (`ollama serve`, or the desktop app) before a run with large file
context. If the local daemon isn't reachable, RAG indexing fails gracefully
and context is passed directly to the LLM instead, so this is a nice-to-have
for large-context runs, not a hard requirement. Override the daemon URL with
`OLLAMA_EMBED_BASE_URL` in `.env` if it isn't at the default
`http://localhost:11434`.

`TAVILY_API_KEY` is optional - without it, `--enable-search` falls back to
keyless DuckDuckGo search automatically.

## Usage

```bash
# Minimal run, autonomous (skips the interview and overview discussion), prints all four artifacts to stdout
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

Once the script passes review, a non-autonomous run also pauses to show a
pre-result overview and let you give free-form feedback before the
character/location bibles are generated - type `/finish` there too when
you're happy with it. Going autonomous at any point (via `--autonomous` or
`/finish` mid-interview) skips this discussion stage as well.

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
