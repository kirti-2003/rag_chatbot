from functools import lru_cache

from langchain_chroma import Chroma

from app.core.config import CHROMA_DIR, COLLECTION_NAME
from app.services.embeddings import get_embedding_model


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Return one reusable Chroma vector-store instance.
    """

    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )