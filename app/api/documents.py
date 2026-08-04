import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import (
    ALLOWED_CONTENT_TYPES,
    MAX_FILE_SIZE_MB,
    UPLOAD_DIR,
)
from app.services.ingestion import ingest_pdf


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


def sanitize_filename(filename: str) -> str:
    """
    Remove unsafe characters from a filename.
    """

    filename = Path(filename).name

    safe_filename = re.sub(
        pattern=r"[^a-zA-Z0-9._-]",
        repl="_",
        string=filename,
    )

    return safe_filename


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    file: UploadFile = File(...),
) -> dict:
    """
    Upload and index a PDF document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file must have a filename.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are currently supported.",
        )

    original_filename = sanitize_filename(file.filename)

    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file must have a .pdf extension.",
        )

    file_bytes = await file.read()

    max_file_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    if len(file_bytes) > max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size cannot exceed "
                f"{MAX_FILE_SIZE_MB} MB."
            ),
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # Give the stored file a unique name so two users can upload
    # files that have the same original filename.
    stored_filename = (
        f"{uuid4()}-{original_filename}"
    )

    stored_path = UPLOAD_DIR / stored_filename

    try:
        stored_path.write_bytes(file_bytes)

        result = ingest_pdf(
            file_path=stored_path,
            original_filename=original_filename,
        )

        return {
            "message": "Document ingested successfully.",
            **result,
        }

    except ValueError as error:
        stored_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    except Exception as error:
        stored_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {error}",
        ) from error

    finally:
        await file.close()