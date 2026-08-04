from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.services.embeddings import get_embedding_model
from app.services.vector_store import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load expensive dependencies once when the server starts.
    """

    get_embedding_model()
    get_vector_store()

    yield


app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "A learning project covering document ingestion, "
        "retrieval and answer generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(documents_router)


@app.get("/")
def root() -> dict:
    return {
        "message": "RAG Chatbot API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
    }