from pathlib import Path


# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directory where uploaded PDFs will be stored
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

# Directory where Chroma stores vector data
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# Name of the Chroma collection
COLLECTION_NAME = "rag_documents"

# Embedding model used for both documents and queries
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Upload restrictions
MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
}


# Create directories automatically when the application starts
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)