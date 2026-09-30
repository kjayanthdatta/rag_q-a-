"""PDF text extraction, cleaning and chunking."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract raw text per page. Returns [{'page': int, 'text': str}]."""
    reader = PdfReader(pdf_path)
    pages: list[dict] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})
    return pages


def clean_text(text: str) -> str:
    """Collapse redundant whitespace while preserving basic readability."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_documents(pages: list[dict], source_filename: str) -> list[Document]:
    """Convert non-empty pages into LangChain Documents with page metadata."""
    documents: list[Document] = []
    for entry in pages:
        cleaned = clean_text(entry["text"])
        if not cleaned:
            continue
        documents.append(
            Document(
                page_content=cleaned,
                metadata={"page": entry["page"], "source": source_filename},
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, preserving page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def process_pdf(pdf_path: str, source_filename: str) -> list[Document]:
    """Full pipeline: extract -> clean -> chunk. Raises ValueError if no text."""
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError("The PDF has no pages or could not be read.")
    documents = build_documents(pages, source_filename)
    if not documents:
        raise ValueError("No extractable text found in the PDF. It may be scanned images.")
    return split_documents(documents)
