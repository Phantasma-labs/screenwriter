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
from src.output_files import build_output_files

SUPPORTED_CONTEXT_EXTENSIONS = {".md", ".txt", ".pdf"}

STATUS_BADGES = {
    "ingest": "[INGESTING]",
    "interview_ask": "[INTERVIEWING]",
    "interview_wait": "[INTERVIEWING]",
    "researcher": "[RESEARCHING]",
    "outliner": "[OUTLINING]",
    "writer": "[WRITING DRAFT]",
    "reviewer": "[REVIEWING]",
    "overview": "[BUILDING OVERVIEW]",
    "overview_discussion_wait": "[OVERVIEW DISCUSSION]",
    "overview_revise": "[REVISING OVERVIEW]",
    "character_bible": "[BUILDING CHARACTER BIBLE]",
    "location_bible": "[BUILDING LOCATION BIBLE]",
    "bible_reviewer": "[REVIEWING BIBLES]",
    "finalize": "[FINALIZING]",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screenwriter Agent CLI")
    parser.add_argument("--topic", "-t", required=True, help="Premise or prompt")
    parser.add_argument(
        "--skill",
        "-s",
        default="short_film",
        choices=[
            "short_film",
            "documentary",
            "commercial",
            "product_shot",
            "learning_dev",
            "infomedia",
        ],
    )
    parser.add_argument(
        "--files", "-f", nargs="*", default=[], help="Context files (.md, .txt, .pdf)"
    )
    parser.add_argument("--max-revisions", "-r", type=int, default=2)
    parser.add_argument(
        "--output", "-o", default=None, help="Output directory for the four generated files"
    )
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


def resolve_output_dir(output: str | None) -> Path | None:
    if output is None:
        return None
    return Path(output)


def write_outputs(output_dir: Path, result: ScreenplayState) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in build_output_files(result).items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def print_outputs(result: ScreenplayState) -> None:
    print("\n" + "=" * 20 + " OVERVIEW " + "=" * 20)
    print(result["overview"])
    print("\n" + "=" * 20 + " SCREENPLAY (Markdown) " + "=" * 20)
    print(result["screenplay_markdown"])
    print("\n" + "=" * 20 + " SCRIPT " + "=" * 20)
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
                interrupt_payload = chunk["__interrupt__"]
                if not interrupt_payload:
                    continue
                value = interrupt_payload[0].value
                if isinstance(value, dict) and value.get("kind") == "overview_discussion":
                    print("[OVERVIEW PRE-RESULT]")
                    print(value["overview"])
                    feedback = input("Feedback (or /finish to accept and continue): ")
                    finish = is_finish_command(feedback)
                    current_input = Command(
                        resume={"feedback": "" if finish else feedback, "finish": finish}
                    )
                else:
                    question = value
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
    settings = load_settings()  # fail fast on invalid numeric env overrides
    app = build_workflow(enable_search=args.enable_search)

    initial_state = new_initial_state(
        topic=args.topic,
        skill=args.skill,
        file_paths=args.files,
        max_revisions=args.max_revisions,
        max_bible_revisions=settings.max_bible_revisions,
        autonomous=args.autonomous,
    )
    config = {"configurable": {"thread_id": f"cli-{id(args)}"}}
    result = run_interactive(app, initial_state, config)

    output_dir = resolve_output_dir(args.output)
    if output_dir is None:
        print_outputs(result)
    else:
        for path in write_outputs(output_dir, result):
            print(f"Wrote {path}")
    return result


if __name__ == "__main__":
    run()
    sys.exit(0)
