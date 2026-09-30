"""Embedding and chat model factories (OpenAI or Gemini, per configuration)."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app import config


def get_embeddings():
    """Create the embeddings client for the active provider.

    Raises RuntimeError when the provider's API key is missing.
    """
    provider = config.get_active_provider()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = config.get_gemini_api_key_or_raise()
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL, google_api_key=api_key
        )
    api_key = config.get_openai_api_key_or_raise()
    return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, api_key=api_key)


def get_chat_model():
    """Create the chat model client for the active provider.

    Raises RuntimeError when the provider's API key is missing.
    """
    provider = config.get_active_provider()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.get_gemini_api_key_or_raise()
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL,
            google_api_key=api_key,
            temperature=config.GEMINI_TEMPERATURE,
        )
    api_key = config.get_openai_api_key_or_raise()
    return ChatOpenAI(
        model=config.OPENAI_CHAT_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
        api_key=api_key,
    )
