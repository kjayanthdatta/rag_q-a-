"""Central application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root = parent of the `app/` directory.
BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
DOCUMENT_INFO_FILE = VECTOR_STORE_DIR / "document_info.json"

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))

# LLM provider: "openai" or "gemini". Gemini uses the free-tier-friendly
# Google AI Studio API key (https://aistudio.google.com/apikey).
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0")))

SUPPORTED_PROVIDERS = ("openai", "gemini")

# RAG tuning
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K: int = int(os.getenv("TOP_K", "4"))

# 10 MB upload limit
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))

ALLOWED_CONTENT_TYPES = {"application/pdf"}


def ensure_directories() -> None:
    """Create data directories if they do not exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_openai_api_key_or_raise() -> str:
    """Return the OpenAI API key or raise a clear error explaining how to configure it."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file (see .env.example) with OPENAI_API_KEY=your_api_key_here, "
            "or set the environment variable, then restart the application."
        )
    return OPENAI_API_KEY


def get_gemini_api_key_or_raise() -> str:
    """Return the Gemini API key or raise a clear error explaining how to configure it."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/apikey and add "
            "GEMINI_API_KEY=your_key_here to your .env file (see .env.example), "
            "then restart the application."
        )
    return GEMINI_API_KEY


def get_active_provider() -> str:
    """Return the configured provider, validating it is supported."""
    provider = (LLM_PROVIDER or "openai").lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. "
            f"Supported values are: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return provider


def get_provider_api_key_or_raise() -> str:
    """Validate the active provider's API key. Used by the API routes."""
    if get_active_provider() == "gemini":
        return get_gemini_api_key_or_raise()
    return get_openai_api_key_or_raise()
