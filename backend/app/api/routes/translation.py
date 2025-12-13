"""Translation API routes for Urdu language support."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import check_rate_limit, get_current_user_optional
from app.db.postgres import get_db_session
from app.models.user import User
from app.services.translation import TranslationService

router = APIRouter(prefix="/translation", tags=["translation"])


# Request/Response schemas
class TranslateTextRequest(BaseModel):
    """Single text translation request."""

    text: str = Field(..., max_length=10000)
    target_language: str = Field(default="ur", pattern="^(ur|en)$")
    chapter_id: str | None = None


class TranslateTextResponse(BaseModel):
    """Translation response."""

    original_text: str
    translated_text: str
    target_language: str
    cached: bool


class TranslateChapterRequest(BaseModel):
    """Chapter translation request."""

    chapter_id: str
    title: str
    content: str
    sections: list[dict] | None = None


class TranslationStatusResponse(BaseModel):
    """Translation status response."""

    chapter_id: str
    total_segments: int
    completed: int
    pending: int
    failed: int
    completion_percentage: float


class QueueTranslationRequest(BaseModel):
    """Request to queue chapter for background translation."""

    chapter_id: str
    content_segments: list[str]
    target_language: str = "ur"


class QueueTranslationResponse(BaseModel):
    """Response after queuing translation."""

    chapter_id: str
    queued_segments: int
    content_hashes: list[str]


# Routes
@router.post(
    "/translate",
    response_model=TranslateTextResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def translate_text(
    body: TranslateTextRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
) -> TranslateTextResponse:
    """
    Translate a text segment to the target language.

    - Supports English to Urdu translation
    - Results are cached for performance
    - Preserves markdown formatting and code blocks
    """
    service = TranslationService(db)

    # Check cache first
    content_hash = service.generate_content_hash(body.text, body.target_language)
    cached = await service.get_cached_translation(content_hash)

    if cached:
        return TranslateTextResponse(
            original_text=body.text,
            translated_text=cached.translated_text,
            target_language=body.target_language,
            cached=True,
        )

    # Perform translation
    translated = await service.translate_text(
        text=body.text,
        target_language=body.target_language,
        chapter_id=body.chapter_id,
    )

    return TranslateTextResponse(
        original_text=body.text,
        translated_text=translated,
        target_language=body.target_language,
        cached=False,
    )


@router.post(
    "/translate/chapter",
    response_model=dict,
    dependencies=[Depends(check_rate_limit)],
)
async def translate_chapter(
    body: TranslateChapterRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
) -> dict:
    """
    Translate an entire chapter's content.

    - Translates title, content, and sections
    - Uses caching for previously translated content
    - May take longer for large chapters
    """
    service = TranslationService(db)

    chapter_content = {
        "chapter_id": body.chapter_id,
        "title": body.title,
        "content": body.content,
        "sections": body.sections or [],
    }

    translated = await service.translate_chapter_content(chapter_content)

    return {
        "chapter_id": body.chapter_id,
        "translated": translated,
        "target_language": "ur",
    }


@router.get(
    "/status/{chapter_id}",
    response_model=TranslationStatusResponse,
)
async def get_translation_status(
    chapter_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> TranslationStatusResponse:
    """Get translation status for a specific chapter."""
    service = TranslationService(db)
    status = await service.get_translation_status(chapter_id)

    return TranslationStatusResponse(**status)


@router.post(
    "/queue",
    response_model=QueueTranslationResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def queue_chapter_translation(
    body: QueueTranslationRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
) -> QueueTranslationResponse:
    """
    Queue chapter segments for background translation.

    Use this for large chapters to avoid timeout issues.
    Check status with GET /translation/status/{chapter_id}
    """
    service = TranslationService(db)

    hashes = await service.queue_chapter_translation(
        chapter_id=body.chapter_id,
        content_segments=body.content_segments,
        target_language=body.target_language,
    )

    return QueueTranslationResponse(
        chapter_id=body.chapter_id,
        queued_segments=len(hashes),
        content_hashes=hashes,
    )


@router.get("/languages")
async def get_supported_languages() -> dict:
    """Get list of supported languages for translation."""
    return {
        "source_languages": [
            {"code": "en", "name": "English", "native_name": "English"},
        ],
        "target_languages": [
            {"code": "ur", "name": "Urdu", "native_name": "اردو", "direction": "rtl"},
        ],
    }
