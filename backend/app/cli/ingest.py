#!/usr/bin/env python3
"""
CLI tool for ingesting book content into the vector database.

Usage:
    python -m app.cli.ingest --docs-path ../docs
    python -m app.cli.ingest --docs-path ../docs --stats-only
    python -m app.cli.ingest --docs-path ../docs --chapter module-1-ros2/ch01-welcome-first-node
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.rag.ingestion import BookIngester
from app.services.rag.embeddings import get_embeddings_service


def main():
    parser = argparse.ArgumentParser(
        description="Ingest book content into the vector database"
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        required=True,
        help="Path to the docs directory containing Markdown files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of text chunks (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't ingest",
    )
    parser.add_argument(
        "--chapter",
        type=str,
        help="Only ingest a specific chapter (e.g., module-1-ros2/ch01-welcome-first-node)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embedding generation (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but don't upload to vector database",
    )

    args = parser.parse_args()

    # Resolve docs path
    docs_path = Path(args.docs_path).resolve()
    if not docs_path.exists():
        print(f"Error: Docs path does not exist: {docs_path}")
        sys.exit(1)

    print(f"📚 Book Ingestion Tool")
    print(f"   Docs path: {docs_path}")
    print(f"   Chunk size: {args.chunk_size}")
    print(f"   Chunk overlap: {args.chunk_overlap}")
    print()

    # Initialize ingester
    ingester = BookIngester(
        docs_path=docs_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Show stats
    if args.stats_only:
        stats = ingester.get_stats()
        print(f"📊 Ingestion Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Total characters: {stats['total_characters']:,}")
        print()
        print("📑 Chapters:")
        for ch in stats["chapters"]:
            print(f"   - {ch['chapter']}: {ch['chunks']} chunks")
        return

    # Get embeddings service
    embeddings_service = get_embeddings_service()

    # Show current collection stats
    collection_stats = embeddings_service.get_collection_stats()
    print(f"🗄️  Vector Collection: {collection_stats.get('collection_name', 'unknown')}")
    if "error" not in collection_stats:
        print(f"   Current vectors: {collection_stats.get('vectors_count', 0)}")
    print()

    # Collect chunks
    if args.chapter:
        # Find the specific chapter file
        chapter_file = None
        for f in ingester.get_chapter_files():
            if args.chapter in str(f):
                chapter_file = f
                break

        if not chapter_file:
            print(f"Error: Chapter not found: {args.chapter}")
            sys.exit(1)

        print(f"📖 Ingesting chapter: {args.chapter}")
        chunks = list(ingester.ingest_file(chapter_file))
    else:
        print(f"📖 Ingesting all chapters...")
        chunks = list(ingester.ingest_all())

    print(f"   Found {len(chunks)} chunks")
    print()

    if args.dry_run:
        print("🔍 Dry run - showing sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Chapter: {chunk.chapter}")
            print(f"Section: {chunk.section}")
            print(f"Content preview: {chunk.content[:200]}...")
        print("\n✅ Dry run complete. No data uploaded.")
        return

    # Upload to vector database
    print(f"⬆️  Uploading to vector database (batch size: {args.batch_size})...")

    try:
        total_uploaded = embeddings_service.upsert_chunks(
            chunks=chunks,
            batch_size=args.batch_size,
        )
        print(f"   Uploaded {total_uploaded} vectors")
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        sys.exit(1)

    # Show final stats
    final_stats = embeddings_service.get_collection_stats()
    print()
    print(f"✅ Ingestion complete!")
    print(f"   Total vectors in collection: {final_stats.get('vectors_count', 'unknown')}")


if __name__ == "__main__":
    main()
