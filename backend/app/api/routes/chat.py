"""
Chat API routes for the RAG chatbot.

Provides endpoints for:
- POST /api/chat - Send a message and get a response
- GET /api/chat/history - Get chat history for a session
- POST /api/chat/feedback - Submit feedback for a response
- DELETE /api/chat/{session_id} - Clear a chat session
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db_session, get_session_maker
from app.services.rag.chat import get_chat_service, ChatService
from app.services.rag.embeddings import get_embeddings_service, EmbeddingsService

router = APIRouter(prefix="/chat", tags=["chat"])


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    chapter: str | None = Field(default=None, description="Current chapter context")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatSource(BaseModel):
    """Source information for a chat response."""

    chapter: str
    section: str
    score: float
    content: str | None = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str
    sources: list[ChatSource]
    session_id: str
    tokens_used: int = 0


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""

    session_id: str
    message_index: int = Field(..., ge=0)
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)


class CollectionStatsResponse(BaseModel):
    """Response model for collection stats."""

    collection_name: str
    vectors_count: int | None = None
    points_count: int | None = None
    status: str | None = None
    error: str | None = None


# Dependencies
def get_chat_svc() -> ChatService:
    """Dependency for chat service."""
    return get_chat_service()


def get_embeddings_svc() -> EmbeddingsService:
    """Dependency for embeddings service."""
    return get_embeddings_service()


# Routes
@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_svc)],
):
    """
    Send a message to the RAG chatbot.

    The chatbot will search the textbook content for relevant information
    and generate a response based on the retrieved context.
    """
    if request.stream:
        # Return streaming response
        async def generate():
            async for chunk in chat_service.chat_stream(
                query=request.message,
                session_id=request.session_id,
                chapter=request.chapter,
            ):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )

    # Non-streaming response
    response = chat_service.chat(
        query=request.message,
        session_id=request.session_id,
        chapter=request.chapter,
    )

    return ChatResponse(
        message=response.message,
        sources=[
            ChatSource(
                chapter=s.get("chapter", ""),
                section=s.get("section", ""),
                score=s.get("score", 0.0),
                content=s.get("content"),
            )
            for s in response.sources
        ],
        session_id=response.session_id,
        tokens_used=response.tokens_used,
    )


@router.post("/stream")
async def send_message_stream(
    request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_svc)],
):
    """
    Send a message and receive a streaming response.

    This endpoint always streams the response, regardless of the `stream` field.
    """
    async def generate():
        async for chunk in chat_service.chat_stream(
            query=request.message,
            session_id=request.session_id,
            chapter=request.chapter,
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )


@router.delete("/{session_id}")
async def clear_session(
    session_id: str,
    chat_service: Annotated[ChatService, Depends(get_chat_svc)],
):
    """Clear the chat history for a session."""
    deleted = chat_service.delete_conversation(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "Session cleared"}


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_svc)],
):
    """
    Submit feedback for a chat response.

    This helps improve the chatbot's responses over time.
    """
    session_maker = get_session_maker()
    async with session_maker() as db:
        # Get messages for this session and find the one at message_index
        history = await chat_service.get_history_from_db(db, request.session_id, limit=100)
        if request.message_index >= len(history):
            raise HTTPException(status_code=404, detail="Message not found")

        message = history[request.message_index]
        await chat_service.save_feedback(
            db, str(message.id), request.rating, request.comment
        )

    return {
        "status": "success",
        "message": "Feedback saved",
        "session_id": request.session_id,
        "rating": request.rating,
    }


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    chat_service: Annotated[ChatService, Depends(get_chat_svc)],
    limit: int = Query(20, ge=1, le=100),
):
    """Get chat history for a session."""
    session_maker = get_session_maker()
    async with session_maker() as db:
        history = await chat_service.get_history_from_db(db, session_id, limit=limit)
        return {
            "session_id": session_id,
            "messages": [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "chapter": m.chapter,
                    "sources": m.sources,
                    "created_at": m.created_at.isoformat(),
                }
                for m in history
            ],
            "count": len(history),
        }


@router.get("/stats", response_model=CollectionStatsResponse)
async def get_stats(
    embeddings_service: Annotated[EmbeddingsService, Depends(get_embeddings_svc)],
):
    """Get statistics about the vector collection."""
    stats = embeddings_service.get_collection_stats()
    return CollectionStatsResponse(**stats)

@router.get("/search")
async def search_content(
    q: Annotated[str, Query(..., min_length=1, max_length=500, description="Search query")],
    limit: int = Query(5, ge=1, le=20),
    chapter: str | None = Query(None),
    embeddings_service: Annotated[
        EmbeddingsService,
        Depends(get_embeddings_svc),
    ] = None,
):
    results = embeddings_service.search(
        query=q,
        limit=limit,
        chapter_filter=chapter,
    )

    return {
        "query": q,
        "results": [r.to_dict() for r in results],
        "count": len(results),
    }

