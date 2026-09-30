"""API tests. External LLM calls are mocked — no real API key needed."""

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app import config

# Ensure a dummy key so app-level key checks pass unless a test overrides them.
config.OPENAI_API_KEY = "test-key"
config.LLM_PROVIDER = "openai"

from app.main import app  # noqa: E402

client = TestClient(app)

MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 60>>stream\nBT /F1 12 Tf 72 720 Td (Hello RAG world) Tj ET\nendstream\nendobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF"
)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_upload_invalid_file_type():
    res = client.post(
        "/upload",
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert res.status_code == 400
    assert "PDF" in res.json()["detail"]


def test_upload_no_file():
    res = client.post("/upload", files={})
    # FastAPI validation returns 422 when the required file part is missing.
    assert res.status_code in (400, 422)


def test_upload_valid_pdf_mocked():
    fake_chunks = ["chunk-1", "chunk-2"]
    with (
        patch("app.services.pdf_service.process_pdf", return_value=fake_chunks),
        patch("app.services.vector_store.build_and_save", return_value=2),
    ):
        res = client.post(
            "/upload",
            files={"file": ("example.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["message"] == "Document processed successfully"
    assert body["filename"] == "example.pdf"
    assert body["chunks"] == 2


def test_upload_empty_pdf_rejected():
    res = client.post(
        "/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert res.status_code == 400


def test_ask_missing_question():
    res = client.post("/ask", json={"question": "   "})
    assert res.status_code in (400, 422)


def test_ask_no_vector_store():
    with patch(
        "app.main.answer_question",
        new_callable=AsyncMock,
        side_effect=FileNotFoundError(
            "No vector store found. Please upload and process a PDF first."
        ),
    ):
        res = client.post("/ask", json={"question": "What is this about?"})
    assert res.status_code == 400
    assert "upload" in res.json()["detail"].lower()


def test_ask_success_mocked():
    fake_result = {
        "answer": "This document is about testing.",
        "sources": [{"page": 1, "content": "Hello RAG world"}],
    }
    with patch(
        "app.main.answer_question", new_callable=AsyncMock, return_value=fake_result
    ):
        res = client.post("/ask", json={"question": "What is this about?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "This document is about testing."
    assert body["sources"][0]["page"] == 1


def test_ask_missing_api_key():
    with patch("app.main.config") as mock_config:
        mock_config.get_provider_api_key_or_raise.side_effect = RuntimeError(
            "GEMINI_API_KEY is not set."
        )
        res = client.post("/ask", json={"question": "Hello?"})
    assert res.status_code == 500


def test_get_active_provider_invalid(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "unknown-provider")
    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        config.get_active_provider()


def test_gemini_factories_use_gemini_models(monkeypatch):
    from app.services import embedding_service

    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "g-test-key")
    with (
        patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as mock_emb,
        patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_chat,
    ):
        embedding_service.get_embeddings()
        embedding_service.get_chat_model()
    mock_emb.assert_called_once_with(
        model=config.GEMINI_EMBEDDING_MODEL, google_api_key="g-test-key"
    )
    mock_chat.assert_called_once_with(
        model=config.GEMINI_CHAT_MODEL,
        google_api_key="g-test-key",
        temperature=config.GEMINI_TEMPERATURE,
    )


def test_openai_factories_use_openai_models(monkeypatch):
    from app.services import embedding_service

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "o-test-key")
    with (
        patch("app.services.embedding_service.OpenAIEmbeddings") as mock_emb,
        patch("app.services.embedding_service.ChatOpenAI") as mock_chat,
    ):
        embedding_service.get_embeddings()
        embedding_service.get_chat_model()
    mock_emb.assert_called_once_with(
        model=config.OPENAI_EMBEDDING_MODEL, api_key="o-test-key"
    )
    mock_chat.assert_called_once_with(
        model=config.OPENAI_CHAT_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
        api_key="o-test-key",
    )


def test_vector_store_rejects_cross_provider_index(monkeypatch):
    from app.services import vector_store

    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    with (
        patch.object(vector_store, "vector_store_exists", return_value=True),
        patch.object(
            vector_store, "load_document_info", return_value={"provider": "openai"}
        ),
    ):
        with pytest.raises(RuntimeError, match="re-upload the PDF"):
            vector_store.load_store()
