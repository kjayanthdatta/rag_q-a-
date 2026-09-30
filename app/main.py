"""FastAPI application: upload PDFs, ask questions, serve frontend."""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError

from app import config
from app.models import AskRequest, AskResponse, HealthResponse, SourceChunk, UploadResponse
from app.services import pdf_service, vector_store
from app.services.rag_service import answer_question

config.ensure_directories()

app = FastAPI(title="AI Document Q&A System", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="Frontend not found.")
    return FileResponse(str(index))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


def _safe_filename(original: str) -> str:
    suffix = Path(original).suffix.lower()
    return f"{uuid.uuid4().hex}{suffix}"


def _is_provider_api_error(exc: Exception) -> bool:
    """Detect upstream LLM/embedding API errors (OpenAI or Google)."""
    if isinstance(exc, OpenAIError):
        return True
    module = type(exc).__module__.lower()
    name = type(exc).__name__.lower()
    return (
        "google" in module
        or "generativeai" in module
        or "generativeai" in name
        or "googleapi" in name
        or name in {"api_error", "googleapierror", "googleapiaerror",
                    "permissiondenied", "unauthenticated", "resourceexhausted"}
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No PDF file uploaded.")

    original_name = file.filename
    if Path(original_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    if file.content_type and file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    # API key check early so we fail fast with a helpful message.
    try:
        config.get_provider_api_key_or_raise()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > config.MAX_UPLOAD_SIZE_BYTES:
        limit_mb = config.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum allowed size is {limit_mb} MB."
        )
    if not contents[:5].startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    config.ensure_directories()
    safe_name = _safe_filename(original_name)
    saved_path = config.UPLOAD_DIR / safe_name
    try:
        saved_path.write_bytes(contents)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

    try:
        chunks = pdf_service.process_pdf(str(saved_path), source_filename=original_name)
    except ValueError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to extract text from the PDF.")

    try:
        count = vector_store.build_and_save(chunks, source_filename=original_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        if _is_provider_api_error(exc):
            raise HTTPException(status_code=502, detail=f"Embedding service error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to build the vector index.")

    return UploadResponse(
        message="Document processed successfully",
        filename=original_name,
        chunks=count,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        config.get_provider_api_key_or_raise()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        result = await answer_question(question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        if _is_provider_api_error(exc):
            raise HTTPException(status_code=502, detail=f"LLM service error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    sources = [
        SourceChunk(page=s.get("page", 0), content=s.get("content", ""))
        for s in result.get("sources", [])
    ]
    return AskResponse(answer=result.get("answer", ""), sources=sources)
