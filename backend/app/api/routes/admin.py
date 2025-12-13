"""
Admin API routes for backend management tasks.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["admin"])


class IngestionStatus(BaseModel):
    status: str
    message: str
    chunks_processed: int = 0
    total_chunks: int = 0


# Global state for tracking ingestion
_ingestion_state = {
    "running": False,
    "chunks_processed": 0,
    "total_chunks": 0,
    "error": None,
}


def run_ingestion_task(docs_path: str, batch_size: int = 10):
    """Background task to run ingestion."""
    global _ingestion_state

    try:
        from pathlib import Path
        from app.services.rag.ingestion import BookIngester
        from app.services.rag.embeddings import get_embeddings_service
        from app.db.qdrant import get_qdrant_client, get_collection_name
        from qdrant_client.models import Distance, VectorParams

        _ingestion_state["running"] = True
        _ingestion_state["error"] = None

        # Initialize ingester
        path = Path(docs_path).resolve()
        ingester = BookIngester(docs_path=path, chunk_size=1000, chunk_overlap=200)

        # Collect chunks
        chunks = list(ingester.ingest_all())
        _ingestion_state["total_chunks"] = len(chunks)

        # Ensure collection exists
        client = get_qdrant_client()
        collection_name = get_collection_name()
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

        # Get embeddings service and upload
        embeddings_service = get_embeddings_service()
        total = embeddings_service.upsert_chunks(chunks=chunks, batch_size=batch_size)

        _ingestion_state["chunks_processed"] = total
        _ingestion_state["running"] = False

    except Exception as e:
        _ingestion_state["error"] = str(e)
        _ingestion_state["running"] = False


@router.post("/ingest", response_model=IngestionStatus)
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    docs_path: str = Query(default="/app/docs", description="Path to docs directory"),
    batch_size: int = Query(default=10, description="Batch size for embeddings"),
):
    """Trigger book content ingestion into vector database."""
    global _ingestion_state

    if _ingestion_state["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already running")

    # Reset state
    _ingestion_state = {
        "running": True,
        "chunks_processed": 0,
        "total_chunks": 0,
        "error": None,
    }

    # Run in background
    background_tasks.add_task(run_ingestion_task, docs_path, batch_size)

    return IngestionStatus(
        status="started",
        message=f"Ingestion started with docs_path={docs_path}, batch_size={batch_size}",
    )


@router.get("/ingest/status", response_model=IngestionStatus)
async def get_ingestion_status():
    """Get current ingestion status."""
    global _ingestion_state

    if _ingestion_state["error"]:
        return IngestionStatus(
            status="error",
            message=_ingestion_state["error"],
            chunks_processed=_ingestion_state["chunks_processed"],
            total_chunks=_ingestion_state["total_chunks"],
        )

    if _ingestion_state["running"]:
        return IngestionStatus(
            status="running",
            message="Ingestion in progress",
            chunks_processed=_ingestion_state["chunks_processed"],
            total_chunks=_ingestion_state["total_chunks"],
        )

    return IngestionStatus(
        status="completed" if _ingestion_state["chunks_processed"] > 0 else "idle",
        message="Ingestion completed" if _ingestion_state["chunks_processed"] > 0 else "No ingestion running",
        chunks_processed=_ingestion_state["chunks_processed"],
        total_chunks=_ingestion_state["total_chunks"],
    )
