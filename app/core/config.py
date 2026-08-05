from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()



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
CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

# Upload restrictions
MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
}

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free",
)

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 500
LLM_MAX_RETRIES = 2

# Create directories automatically when the application starts
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)