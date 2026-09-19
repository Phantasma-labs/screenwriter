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
