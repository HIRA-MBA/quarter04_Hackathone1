from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings

# Module-level client instance
_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance with lazy initialization."""
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            timeout=120,  # Increased timeout for cloud connections
        )
    return _qdrant_client


def reset_qdrant_client() -> None:
    """Reset the Qdrant client (useful for reconnection after errors)."""
    global _qdrant_client
    _qdrant_client = None


async def ensure_collection_exists(
    collection_name: str | None = None,
    vector_size: int = 1536,  # OpenAI text-embedding-3-small dimension
) -> None:
    """Ensure the vector collection exists, create if not."""
    settings = get_settings()
    client = get_qdrant_client()
    name = collection_name or settings.qdrant_collection

    collections = client.get_collections().collections
    if not any(c.name == name for c in collections):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def get_collection_name() -> str:
    """Get the configured collection name."""
    return get_settings().qdrant_collection
