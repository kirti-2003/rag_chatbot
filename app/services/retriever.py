from typing import Any

from langchain_core.documents import Document

from app.services.vector_store import get_vector_store


SUPPORTED_SEARCH_TYPES = {
    "similarity",
    "similarity_score_threshold",
    "mmr",
}


def clean_query(query: str) -> str:
    """
    Perform basic query cleaning.

    We do not aggressively modify the query because punctuation,
    technical terms and wording can be important for retrieval.
    """

    cleaned_query = " ".join(query.split())

    if not cleaned_query:
        raise ValueError("The query cannot be empty.")

    return cleaned_query


def build_metadata_filter(
    document_id: str | None = None,
    filename: str | None = None,
) -> dict[str, Any] | None:
    """
    Build a Chroma metadata filter.

    For this first version, only one filter is applied at a time.
    document_id is preferred because it is unique.
    """

    if document_id:
        return {
            "document_id": document_id,
        }

    if filename:
        return {
            "filename": filename,
        }

    return None


def convert_result(
    document: Document,
    rank: int,
    score: float | None = None,
) -> dict[str, Any]:
    """
    Convert a LangChain Document into JSON-friendly output.
    """

    metadata = document.metadata.copy()

    page = metadata.get("page")

    # PDF loaders often store pages using zero-based indexing.
    if isinstance(page, int):
        metadata["display_page"] = page + 1

    return {
        "rank": rank,
        "content": document.page_content,
        "score": round(score, 4) if score is not None else None,
        "metadata": metadata,
    }


def format_context(results: list[dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into one context string.

    Later, this context will be inserted into the LLM prompt.
    """

    context_parts: list[str] = []

    for result in results:
        metadata = result["metadata"]

        filename = metadata.get("filename", "Unknown source")
        page = metadata.get("display_page", "Unknown")
        rank = result["rank"]

        source_header = (
            f"[Source {rank}: {filename}, page {page}]"
        )

        context_parts.append(
            f"{source_header}\n{result['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_by_similarity(
    query: str,
    top_k: int,
    metadata_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Return the top-K semantically similar chunks.

    This version also returns relevance scores.
    """

    vector_store = get_vector_store()

    search_kwargs: dict[str, Any] = {
        "k": top_k,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    matches = vector_store.similarity_search_with_relevance_scores(
        query=query,
        **search_kwargs,
    )

    results: list[dict[str, Any]] = []

    for rank, (document, score) in enumerate(matches, start=1):
        results.append(
            convert_result(
                document=document,
                rank=rank,
                score=score,
            )
        )

    return results


def retrieve_with_threshold(
    query: str,
    top_k: int,
    score_threshold: float,
    metadata_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Return only chunks whose relevance score meets the threshold.
    """

    vector_store = get_vector_store()

    search_kwargs: dict[str, Any] = {
        "k": top_k,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    matches = vector_store.similarity_search_with_relevance_scores(
        query=query,
        **search_kwargs,
    )

    filtered_matches = [
        (document, score)
        for document, score in matches
        if score >= score_threshold
    ]

    results: list[dict[str, Any]] = []

    for rank, (document, score) in enumerate(
        filtered_matches,
        start=1,
    ):
        results.append(
            convert_result(
                document=document,
                rank=rank,
                score=score,
            )
        )

    return results


def retrieve_by_mmr(
    query: str,
    top_k: int,
    fetch_k: int,
    lambda_mult: float,
    metadata_filter: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant but diverse chunks using MMR.

    Chroma's MMR method does not return a relevance score,
    so score is returned as None in this mode.
    """

    vector_store = get_vector_store()

    search_kwargs: dict[str, Any] = {
        "k": top_k,
        "fetch_k": max(fetch_k, top_k),
        "lambda_mult": lambda_mult,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    documents = vector_store.max_marginal_relevance_search(
        query=query,
        **search_kwargs,
    )

    results: list[dict[str, Any]] = []

    for rank, document in enumerate(documents, start=1):
        results.append(
            convert_result(
                document=document,
                rank=rank,
                score=None,
            )
        )

    return results


def retrieve_documents(
    query: str,
    search_type: str = "similarity",
    top_k: int = 4,
    score_threshold: float | None = None,
    fetch_k: int = 15,
    lambda_mult: float = 0.5,
    document_id: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Main retriever function.

    It selects the requested search strategy and returns
    document chunks plus a formatted context string.
    """

    cleaned_query = clean_query(query)

    if search_type not in SUPPORTED_SEARCH_TYPES:
        raise ValueError(
            f"Unsupported search type: {search_type}"
        )

    metadata_filter = build_metadata_filter(
        document_id=document_id,
        filename=filename,
    )

    if search_type == "similarity":
        results = retrieve_by_similarity(
            query=cleaned_query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    elif search_type == "similarity_score_threshold":
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else 0.5
        )

        results = retrieve_with_threshold(
            query=cleaned_query,
            top_k=top_k,
            score_threshold=effective_threshold,
            metadata_filter=metadata_filter,
        )

    else:
        results = retrieve_by_mmr(
            query=cleaned_query,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            metadata_filter=metadata_filter,
        )

    context = format_context(results)

    return {
        "query": cleaned_query,
        "search_type": search_type,
        "result_count": len(results),
        "results": results,
        "context": context,
    }