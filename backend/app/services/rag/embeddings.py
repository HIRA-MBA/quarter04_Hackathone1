"""
Embeddings service for RAG chatbot.

Handles OpenAI embedding generation and Qdrant vector operations.
"""

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from app.config import get_settings
from app.db.qdrant import get_qdrant_client, get_collection_name, ensure_collection_exists
from app.services.rag.ingestion import TextChunk


@dataclass
class SearchResult:
    """A search result from vector similarity search."""

    content: str
    chapter: str
    section: str
    score: float
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "chapter": self.chapter,
            "section": self.section,
            "score": self.score,
            "metadata": self.metadata,
        }


class EmbeddingsService:
    """Service for generating and managing embeddings."""

    def __init__(self):
        settings = get_settings()
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.qdrant_client = get_qdrant_client()
        self.collection_name = get_collection_name()
        self.embedding_model = settings.embedding_model
        self.embedding_dimension = 1536  # text-embedding-3-small

    async def initialize(self) -> None:
        """Initialize the embeddings service and ensure collection exists."""
        await ensure_collection_exists(
            collection_name=self.collection_name,
            vector_size=self.embedding_dimension,
        )

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def upsert_chunks(self, chunks: list[TextChunk], batch_size: int = 100) -> int:
        """Upsert text chunks with their embeddings into Qdrant."""
        import time
        total_upserted = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(chunks), batch_size), 1):
            batch = chunks[i : i + batch_size]
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            # Generate embeddings for batch
            texts = [chunk.content for chunk in batch]
            embeddings = self.generate_embeddings_batch(texts)

            # Create points for Qdrant
            points = [
                PointStruct(
                    id=hash(chunk.chunk_id) & 0xFFFFFFFFFFFFFFFF,  # Positive int64
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                )
                for chunk, embedding in zip(batch, embeddings)
            ]

            # Upsert to Qdrant with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"   Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise

            total_upserted += len(points)

        return total_upserted

    def search(
        self,
        query: str,
        limit: int = 5,
        chapter_filter: str | None = None,
        score_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Search for similar content using vector similarity."""
        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Build filter if chapter specified
        query_filter = None
        if chapter_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="chapter",
                        match=MatchValue(value=chapter_filter),
                    )
                ]
            )

        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        return [
            SearchResult(
                content=result.payload.get("content", ""),
                chapter=result.payload.get("chapter", ""),
                section=result.payload.get("section", ""),
                score=result.score,
                metadata={
                    k: v
                    for k, v in result.payload.items()
                    if k not in ("content", "chapter", "section")
                },
            )
            for result in results
        ]

    def search_with_context(
        self,
        query: str,
        limit: int = 5,
        chapter_filter: str | None = None,
    ) -> tuple[list[SearchResult], str]:
        """Search and return results with formatted context string."""
        results = self.search(
            query=query,
            limit=limit,
            chapter_filter=chapter_filter,
        )

        if not results:
            return results, ""

        # Format context for LLM
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result.chapter} - {result.section}]\n{result.content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return results, context

    def delete_by_chapter(self, chapter: str) -> int:
        """Delete all vectors for a specific chapter."""
        # Get points matching the chapter
        scroll_result = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="chapter",
                        match=MatchValue(value=chapter),
                    )
                ]
            ),
            limit=10000,
        )

        point_ids = [point.id for point in scroll_result[0]]

        if point_ids:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )

        return len(point_ids)

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e),
            }


# Singleton instance
_embeddings_service: EmbeddingsService | None = None


def get_embeddings_service() -> EmbeddingsService:
    """Get embeddings service singleton."""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
    return _embeddings_service
