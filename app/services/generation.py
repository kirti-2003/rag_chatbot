from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from app.services.llm import get_llm
from app.services.retriever import retrieve_documents


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant for a document-based
question-answering system.

Follow these rules:

1. Answer using only the supplied context.
2. Do not use unsupported outside knowledge.
3. If the context does not contain the answer, say:
   "I could not find the answer in the uploaded documents."
4. Keep the answer clear and concise.
5. Do not invent policies, numbers, names or dates.
6. When possible, mention the source filename and page.
""".strip(),
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}

Provide the answer based only on the context above.
""".strip(),
        ),
    ]
)


@traceable(
    name="Generate RAG Answer",
    run_type="chain",
    tags=["rag", "generation"],
)
def generate_answer(
    question: str,
    search_type: str = "similarity",
    top_k: int = 3,
    score_threshold: float | None = None,
    fetch_k: int = 10,
    lambda_mult: float = 0.7,
    document_id: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Run the complete retrieval and generation flow.

    Question
      -> retrieve relevant chunks
      -> build prompt
      -> call OpenRouter LLM
      -> return the generated answer
    """

    retrieval_result = retrieve_documents(
        query=question,
        search_type=search_type,
        top_k=top_k,
        score_threshold=score_threshold,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        document_id=document_id,
        filename=filename,
    )

    retrieved_results = retrieval_result["results"]
    context = retrieval_result["context"]

    if not retrieved_results or not context.strip():
        return {
            "question": question,
            "answer": (
                "I could not find the answer in the "
                "uploaded documents."
            ),
            "search_type": search_type,
            "sources": [],
            "retrieved_chunks": [],
        }

    llm = get_llm()

    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = build_sources(retrieved_results)

    return {
        "question": question,
        "answer": answer.strip(),
        "search_type": search_type,
        "sources": sources,
        "retrieved_chunks": retrieved_results,
    }


@traceable(
    name="Build RAG Sources",
    run_type="chain",
    tags=["rag", "sources"],
)
def build_sources(
    retrieved_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a clean source list from retrieved metadata.
    """

    sources: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, Any, Any]] = set()

    for result in retrieved_results:
        metadata = result["metadata"]

        filename = metadata.get(
            "filename",
            "Unknown source",
        )

        page = metadata.get("display_page")
        chunk_index = metadata.get("chunk_index")

        source_key = (
            filename,
            page,
            chunk_index,
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "filename": filename,
                "page": page,
                "chunk_index": chunk_index,
                "relevance_score": result.get("score"),
            }
        )

    return sources