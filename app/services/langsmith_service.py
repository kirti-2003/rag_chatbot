from langsmith import Client

from app.core.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
)


def get_langsmith_client() -> Client | None:
    """
    Create a LangSmith client when tracing is enabled.

    Returns None when LangSmith tracing is disabled.
    """

    if not LANGSMITH_TRACING:
        return None

    return Client(api_key=LANGSMITH_API_KEY)


def get_langsmith_status() -> dict:
    """
    Return safe LangSmith configuration information.

    The API key is intentionally not returned.
    """

    return {
        "tracing_enabled": LANGSMITH_TRACING,
        "project": LANGSMITH_PROJECT,
        "api_key_configured": bool(LANGSMITH_API_KEY),
    }