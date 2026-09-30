"""FAISS vector store persistence helpers."""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app import config
from app.services import embedding_service


def faiss_files_exist() -> bool:
    """Check whether a persisted FAISS index is present."""
    folder = Path(config.VECTOR_STORE_DIR)
    return (folder / "index.faiss").exists() and (folder / "index.pkl").exists()


def vector_store_exists() -> bool:
    return faiss_files_exist()


def build_and_save(chunks: list[Document], source_filename: str) -> int:
    """Build a FAISS index from chunks and persist it (replaces any old index)."""
    config.ensure_directories()
    embeddings = embedding_service.get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(config.VECTOR_STORE_DIR))
    info = {
        "filename": source_filename,
        "chunks": len(chunks),
        "provider": config.get_active_provider(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(config.DOCUMENT_INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return len(chunks)


def load_store() -> FAISS:
    """Load the persisted FAISS index. Raises FileNotFoundError when absent.

    Raises RuntimeError when the index was built with a different provider,
    since embedding dimensions differ between providers.
    """
    if not vector_store_exists():
        raise FileNotFoundError(
            "No vector store found. Please upload and process a PDF first."
        )
    info = load_document_info()
    active = config.get_active_provider()
    indexed_provider = (info or {}).get("provider", "openai")
    if indexed_provider != active:
        raise RuntimeError(
            f"The saved index was built with provider '{indexed_provider}' "
            f"but the active provider is '{active}'. "
            "Embedding dimensions differ between providers, "
            "so please re-upload the PDF to rebuild the index."
        )
    embeddings = embedding_service.get_embeddings()
    return FAISS.load_local(
        str(config.VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_document_info() -> dict | None:
    """Return metadata about the currently indexed document, if any."""
    try:
        return json.loads(Path(config.DOCUMENT_INFO_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
