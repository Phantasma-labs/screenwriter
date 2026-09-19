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
