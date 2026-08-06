from fastapi import APIRouter

from app.services.langsmith_service import get_langsmith_status


router = APIRouter(
    prefix="/observability",
    tags=["Observability"],
)


@router.get("/status")
def observability_status() -> dict:
    """
    Check whether LangSmith tracing is configured.
    """

    return {
        "message": "Observability configuration loaded successfully.",
        "langsmith": get_langsmith_status(),
    }