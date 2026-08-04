from fastapi import APIRouter, HTTPException, status

from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
)
from app.services.retriever import retrieve_documents


router = APIRouter(
    prefix="/retrieval",
    tags=["Retrieval"],
)


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
def search_documents(
    request: RetrievalRequest,
) -> RetrievalResponse:
    """
    Search indexed document chunks.

    This endpoint performs retrieval only.
    It does not call an LLM or generate a final answer.
    """

    try:
        result = retrieve_documents(
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            fetch_k=request.fetch_k,
            lambda_mult=request.lambda_mult,
            document_id=request.document_id,
            filename=request.filename,
        )

        return RetrievalResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {error}",
        ) from error