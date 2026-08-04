import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.services.vector_store import get_vector_store


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    The hash acts like a unique fingerprint for the file.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_pdf(file_path: Path) -> list[Document]:
    """
    Extract text from a PDF.

    PyMuPDFLoader normally creates one LangChain Document
    object for each PDF page.
    """

    loader = PyMuPDFLoader(str(file_path))
    documents = loader.load()

    return documents


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Remove empty pages and clean basic whitespace.
    """

    cleaned_documents: list[Document] = []

    for document in documents:
        cleaned_text = document.page_content.strip()

        if not cleaned_text:
            continue

        # Replace null characters that can sometimes appear
        # during PDF extraction.
        cleaned_text = cleaned_text.replace("\x00", " ")

        document.page_content = cleaned_text
        cleaned_documents.append(document)

    return cleaned_documents


def add_document_metadata(
    documents: list[Document],
    file_path: Path,
    original_filename: str,
    file_hash: str,
    document_id: str,
) -> list[Document]:
    """
    Add metadata that will later help us filter,
    identify and cite retrieved chunks.
    """

    ingestion_time = datetime.now(timezone.utc).isoformat()

    for document in documents:
        existing_metadata = document.metadata.copy()

        document.metadata = {
            **existing_metadata,
            "document_id": document_id,
            "filename": original_filename,
            "stored_filename": file_path.name,
            "file_type": "pdf",
            "file_hash": file_hash,
            "ingested_at": ingestion_time,
        }

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Divide PDF pages into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["character_count"] = len(chunk.page_content)

    return chunks


def create_chunk_ids(
    chunks: list[Document],
    document_id: str,
) -> list[str]:
    """
    Create a unique ID for each chunk.
    """

    chunk_ids = []

    for index, chunk in enumerate(chunks):
        page_number = chunk.metadata.get("page", 0)

        chunk_id = (
            f"{document_id}"
            f"-page-{page_number}"
            f"-chunk-{index}"
        )

        chunk_ids.append(chunk_id)

    return chunk_ids


def ingest_pdf(
    file_path: Path,
    original_filename: str,
) -> dict:
    """
    Complete data-ingestion pipeline.

    PDF
      -> documents
      -> clean text
      -> metadata
      -> chunks
      -> embeddings
      -> Chroma
    """

    document_id = str(uuid4())
    file_hash = calculate_file_hash(file_path)

    # 1. Load the PDF
    documents = load_pdf(file_path)

    if not documents:
        raise ValueError("No pages could be extracted from the PDF.")

    # 2. Remove empty content
    documents = clean_documents(documents)

    if not documents:
        raise ValueError(
            "The PDF does not contain readable text. "
            "It may be a scanned document requiring OCR."
        )

    # 3. Add useful metadata
    documents = add_document_metadata(
        documents=documents,
        file_path=file_path,
        original_filename=original_filename,
        file_hash=file_hash,
        document_id=document_id,
    )

    # 4. Split pages into chunks
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    # 5. Create unique chunk IDs
    chunk_ids = create_chunk_ids(
        chunks=chunks,
        document_id=document_id,
    )

    # 6. Generate embeddings and save everything in Chroma
    vector_store = get_vector_store()

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    return {
        "document_id": document_id,
        "filename": original_filename,
        "file_hash": file_hash,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "status": "indexed",
    }