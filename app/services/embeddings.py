from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and cache the embedding model.

    lru_cache prevents the model from being loaded again
    for every API request.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )