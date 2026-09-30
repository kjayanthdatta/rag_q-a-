"""RAG question-answering pipeline."""

from langchain_core.prompts import ChatPromptTemplate

from app import config
from app.services import embedding_service, vector_store

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "Answer the user's question using ONLY the provided document context. "
    "If the answer is not present in the context, say that the information "
    "is not available in the uploaded document. Do not invent or assume information."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Document context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)

NOT_FOUND_PHRASES = (
    "not available in the uploaded document",
    "not present in the context",
    "could not be found in the uploaded document",
    "not found in the uploaded document",
)


def _format_context(docs: list) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        source = d.metadata.get("source", "document")
        parts.append(f"[Page {page} | {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


async def answer_question(question: str, top_k: int | None = None) -> dict:
    """Run similarity search + LLM generation. Returns {'answer', 'sources'}."""
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("Question must not be empty.")
    if not vector_store.vector_store_exists():
        raise FileNotFoundError(
            "No vector store found. Please upload and process a PDF first."
        )

    k = top_k or config.TOP_K
    store = vector_store.load_store()
    relevant_docs = store.similarity_search(cleaned, k=k)

    if not relevant_docs:
        return {
            "answer": "The information is not available in the uploaded document.",
            "sources": [],
        }

    context = _format_context(relevant_docs)
    llm = embedding_service.get_chat_model()
    messages = PROMPT.format_messages(context=context, question=cleaned)
    # Async invoke: the Gemini client needs a running event loop, which a
    # sync route (AnyIO worker thread) does not provide.
    response = await llm.ainvoke(messages)
    answer_text = (response.content or "").strip() if response else ""
    if not answer_text:
        answer_text = "The information is not available in the uploaded document."

    sources = []
    # Only attach sources when the model actually used the context.
    lowered = answer_text.lower()
    used_context = not any(phrase in lowered for phrase in NOT_FOUND_PHRASES)
    if used_context:
        for d in relevant_docs:
            try:
                page = int(d.metadata.get("page", 0))
            except (TypeError, ValueError):
                page = 0
            sources.append({"page": page, "content": d.page_content[:1000]})

    return {"answer": answer_text, "sources": sources}
