from typing import Any, Literal

from pydantic import BaseModel, Field


SearchType = Literal[
    "similarity",
    "similarity_score_threshold",
    "mmr",
]


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        examples=[
            "What is the password requirement?"
        ],
    )

    search_type: SearchType = Field(
        default="similarity",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )

    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    fetch_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    lambda_mult: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )

    document_id: str | None = None
    filename: str | None = None


class ChatSource(BaseModel):
    filename: str
    page: int | None = None
    chunk_index: int | None = None
    relevance_score: float | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    search_type: SearchType
    sources: list[ChatSource]
    retrieved_chunks: list[dict[str, Any]]