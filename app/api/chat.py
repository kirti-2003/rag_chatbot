from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from app.services.generation import generate_answer


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
def chat_with_documents(
    request: ChatRequest,
) -> ChatResponse:
    """
    Retrieve relevant document context and generate
    an answer using an OpenRouter-hosted LLM.
    """

    try:
        result = generate_answer(
            question=request.question,
            search_type=request.search_type,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            fetch_k=request.fetch_k,
            lambda_mult=request.lambda_mult,
            document_id=request.document_id,
            filename=request.filename,
        )

        return ChatResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {error}",
        ) from error