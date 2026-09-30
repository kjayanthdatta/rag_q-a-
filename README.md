# AI Document Q&A System

A beginner-friendly Retrieval-Augmented Generation (RAG) app. Upload a PDF, then ask questions answered **only** from the document content.

## Features

- Upload a PDF via web UI or API
- Text extraction with PyPDF (multi-page, whitespace cleaning)
- Overlapping chunking with LangChain (`1000` chars / `200` overlap by default)
- Pluggable LLM provider: **OpenAI** or **Google Gemini** (free-tier friendly) for embeddings + chat, local FAISS vector store
- Similarity search + grounded LLM answers with page-number sources
- Clear "not available in the uploaded document" fallback (no hallucinating)
- Validation: PDF-only, 10 MB limit, safe filenames, helpful JSON errors
- Simple HTML/CSS/JS frontend, Swagger UI at `/docs`
- Pytest suite with mocked LLM provider calls

## Architecture

```text
PDF upload
  -> Extract text (PyPDF, per page)
  -> Clean whitespace
  -> Split into chunks (RecursiveCharacterTextSplitter, page metadata kept)
  -> Embed chunks (OpenAI text-embedding-3-small OR Gemini gemini-embedding-001)
  -> Store in FAISS (data/vector_store/)
  -> User question
  -> Similarity search (top-k = 4)
  -> Retrieved context + strict system prompt
  -> LLM answer (gpt-4o-mini OR gemini-3.8-flash)
  -> Answer + sources [{page, content}]
```

## Technology Stack

- Python 3.11+
- FastAPI + Uvicorn
- LangChain (`langchain`, `langchain-openai`, `langchain-google-genai`, `langchain-community`, `langchain-text-splitters`)
- OpenAI API or Google Gemini API (embeddings + chat, selected via `LLM_PROVIDER`)
- FAISS (`faiss-cpu`) local vector DB
- PyPDF, Pydantic, python-dotenv, python-multipart
- Plain HTML/CSS/JavaScript frontend
- pytest + httpx/TestClient for tests

## Folder Structure

```text
ai-document-qa/
  app/
    main.py               # FastAPI routes: /, /health, /upload, /ask
    config.py             # env vars, paths, limits
    models.py             # Pydantic schemas
    services/
      pdf_service.py      # extract / clean / chunk
      embedding_service.py# OpenAI / Gemini clients
      vector_store.py     # FAISS save / load
      rag_service.py      # similarity search + LLM prompt
    static/
      index.html
      style.css
      script.js
  data/
    uploads/              # saved PDFs (safe uuid names)
    vector_store/         # index.faiss, index.pkl, document_info.json
  tests/
    test_api.py
  .env.example
  requirements.txt
  run.py
  README.md
```

One active document at a time: each new upload replaces the FAISS index.

## Installation (Windows)

```bat
cd D:\rag_q-a-
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Requires Python 3.11+. FAISS on Windows works via the `faiss-cpu` wheel.

## API Key Configuration

1. Copy the example file:

```bat
copy .env.example .env
```

2. Pick a provider and edit `.env`:

**Gemini (free tier, recommended when OpenAI credits run out):**

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key_here
```

Get a free key at **https://aistudio.google.com/apikey** (no credit card required).

**OpenAI:**

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

Never commit `.env`. The app returns a clear error if the active provider's key is missing.

> Switching providers changes embedding dimensions, so re-upload your PDF after
> changing `LLM_PROVIDER` — the app will tell you if the saved index belongs to
> the other provider.

## How to Start

```bat
venv\Scripts\activate
uvicorn app.main:app --reload
```

or:

```bat
venv\Scripts\activate
python run.py
```

Local URL: **http://127.0.0.1:8000**

Swagger UI: **http://127.0.0.1:8000/docs**

## How to Use

1. Open `http://127.0.0.1:8000`
2. Select a PDF, click **Upload & Process**
3. Wait for success (e.g. `example.pdf processed into 25 chunks`)
4. Type a question, click **Ask**
5. Read the answer + source pages

## API Endpoints

### `GET /`
Serves the frontend (`app/static/index.html`).

### `GET /health`
```json
{ "status": "healthy" }
```

### `POST /upload`
Multipart form with a `file` field (PDF only, max 10 MB).

Success `200`:
```json
{ "message": "Document processed successfully", "filename": "example.pdf", "chunks": 25 }
```

Errors: `400` invalid type / empty / no text, `413` too large, `500` missing API key / index failure.

### `POST /ask`
```json
{ "question": "What is this document about?" }
```

Success `200`:
```json
{
  "answer": "...",
  "sources": [{ "page": 3, "content": "..." }]
}
```

Errors: `400` empty question / no document uploaded yet, `500` missing API key, `502` LLM provider error.

## Common Errors and Solutions

| Error | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | `LLM_PROVIDER=openai` but no key | Add `OPENAI_API_KEY` to `.env`, restart |
| `GEMINI_API_KEY is not set` | `LLM_PROVIDER=gemini` but no key | Get a free key at https://aistudio.google.com/apikey, add to `.env`, restart |
| `The saved index was built with provider ...` | Switched `LLM_PROVIDER` | Re-upload the PDF to rebuild the index |
| `Only PDF files are allowed` | Wrong file type | Upload a real `.pdf` |
| `File too large` | > 10 MB | Compress/split the PDF or raise `MAX_UPLOAD_SIZE_BYTES` |
| `No extractable text found` | Scanned-image PDF | Use an OCR tool first |
| `No vector store found` | Asked before uploading | Upload a PDF first |
| `LLM/Embedding service error` | Provider outage / bad key / quota | Check key, billing/quota, network |
| `faiss` install fails | Old Python | Use Python 3.11+ on 64-bit Windows |

## Running Tests

```bat
venv\Scripts\activate
pytest -v
```

Tests mock both providers, so no real API key or network is needed.
