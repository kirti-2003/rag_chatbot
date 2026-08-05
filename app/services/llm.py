from functools import lru_cache

from langchain_openrouter import ChatOpenRouter

from app.core.config import (
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenRouter:
    """
    Create and cache the OpenRouter chat model.

    The same model instance is reused instead of being
    recreated for every API request.
    """

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. "
            "Add it to the .env file."
        )

    return ChatOpenRouter(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        max_retries=LLM_MAX_RETRIES,
    )