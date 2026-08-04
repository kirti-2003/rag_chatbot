from typing import Any, Literal

from pydantic import BaseModel, Field


SearchType = Literal[
    "similarity",
    "similarity_score_threshold",
    "mmr",
]


class RetrievalRequest(BaseModel):
    """
    Data sent by the client to the retrieval endpoint.
    """

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        examples=["How many casual leaves do employees get?"],
    )

    search_type: SearchType = Field(
        default="similarity",
        description=(
            "Retrieval strategy: similarity, "
            "similarity_score_threshold, or mmr."
        ),
    )

    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum number of chunks to return.",
    )

    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum relevance score. Used mainly with "
            "similarity_score_threshold."
        ),
    )

    fetch_k: int = Field(
        default=15,
        ge=1,
        le=100,
        description=(
            "Number of initial candidates fetched before MMR selection."
        ),
    )

    lambda_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "MMR balance. A larger value favors query relevance; "
            "a smaller value favors diversity."
        ),
    )

    document_id: str | None = Field(
        default=None,
        description="Optionally search within one indexed document.",
    )

    filename: str | None = Field(
        default=None,
        description="Optionally filter by the original filename.",
    )


class RetrievedChunk(BaseModel):
    """
    One document chunk returned by the retriever.
    """

    rank: int
    content: str
    score: float | None = None
    metadata: dict[str, Any]


class RetrievalResponse(BaseModel):
    """
    Complete response returned by the retrieval API.
    """

    query: str
    search_type: SearchType
    result_count: int
    results: list[RetrievedChunk]
    context: str