from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
    )


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
