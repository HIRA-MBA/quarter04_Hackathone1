"""
Text ingestion pipeline for RAG chatbot.

Parses Markdown content from book chapters and creates semantic chunks
suitable for vector embedding and retrieval.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class TextChunk:
    """A semantic chunk of text with metadata."""

    content: str
    chapter: str
    section: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Generate unique ID for this chunk."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.chapter}_{self.section}_{self.chunk_index}_{content_hash}"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chapter": self.chapter,
            "section": self.section,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


class MarkdownParser:
    """Parse Markdown files and extract structured content."""

    # Regex patterns for Markdown elements
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
    MERMAID_PATTERN = re.compile(r"```mermaid\n.*?```", re.DOTALL)

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Extract YAML frontmatter from content."""
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            return {}, content

        frontmatter_text = match.group(1)
        remaining_content = content[match.end() :]

        # Simple YAML parsing for common fields
        metadata = {}
        for line in frontmatter_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"').strip("'")

        return metadata, remaining_content

    def extract_sections(self, content: str) -> list[dict]:
        """Extract sections based on headings."""
        sections = []
        lines = content.split("\n")
        current_section = {"heading": "Introduction", "level": 1, "content": []}

        for line in lines:
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                # Save previous section if it has content
                if current_section["content"]:
                    current_section["content"] = "\n".join(current_section["content"]).strip()
                    if current_section["content"]:
                        sections.append(current_section)

                # Start new section
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                current_section = {"heading": heading, "level": level, "content": []}
            else:
                current_section["content"].append(line)

        # Don't forget the last section
        if current_section["content"]:
            current_section["content"] = "\n".join(current_section["content"]).strip()
            if current_section["content"]:
                sections.append(current_section)

        return sections

    def clean_content(self, content: str) -> str:
        """Clean content while preserving important text."""
        # Remove Mermaid diagrams (keep placeholder text)
        content = self.MERMAID_PATTERN.sub("[Diagram]", content)

        # Keep code blocks but mark them
        # This helps the LLM understand context
        content = re.sub(
            r"```(\w+)\n(.*?)```",
            r"[Code Example (\1):\n\2]",
            content,
            flags=re.DOTALL,
        )

        # Clean up excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content.strip()

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break

                # Fall back to sentence boundary
                elif (sentence_end := text.rfind(". ", start, end)) > start + self.chunk_size // 2:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks


class BookIngester:
    """Ingest book content from Markdown files."""

    def __init__(
        self,
        docs_path: Path | str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.docs_path = Path(docs_path)
        self.parser = MarkdownParser(chunk_size, chunk_overlap)

    def get_chapter_files(self) -> list[Path]:
        """Get all Markdown files from docs directory."""
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_path}")

        files = []
        for pattern in ["**/*.md", "**/*.mdx"]:
            files.extend(self.docs_path.glob(pattern))

        # Sort by path for consistent ordering
        return sorted(files)

    def extract_chapter_name(self, file_path: Path) -> str:
        """Extract chapter identifier from file path."""
        # Get relative path from docs directory
        rel_path = file_path.relative_to(self.docs_path)
        # Use path without extension as chapter name
        return str(rel_path.with_suffix("")).replace("\\", "/")

    def ingest_file(self, file_path: Path) -> Iterator[TextChunk]:
        """Ingest a single Markdown file into chunks."""
        content = file_path.read_text(encoding="utf-8")
        chapter = self.extract_chapter_name(file_path)

        # Parse frontmatter
        metadata, content = self.parser.parse_frontmatter(content)

        # Extract sections
        sections = self.parser.extract_sections(content)

        chunk_index = 0
        for section in sections:
            section_name = section["heading"]
            section_content = self.parser.clean_content(section["content"])

            if not section_content:
                continue

            # Chunk the section content
            chunks = self.parser.chunk_text(section_content)

            for chunk_text in chunks:
                yield TextChunk(
                    content=chunk_text,
                    chapter=chapter,
                    section=section_name,
                    chunk_index=chunk_index,
                    metadata={
                        "title": metadata.get("title", ""),
                        "sidebar_label": metadata.get("sidebar_label", ""),
                        "heading_level": section["level"],
                        "file_path": str(file_path),
                    },
                )
                chunk_index += 1

    def ingest_all(self) -> Iterator[TextChunk]:
        """Ingest all book content."""
        files = self.get_chapter_files()

        for file_path in files:
            yield from self.ingest_file(file_path)

    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        files = self.get_chapter_files()
        total_chunks = 0
        total_chars = 0
        chapters = []

        for file_path in files:
            chapter_chunks = list(self.ingest_file(file_path))
            total_chunks += len(chapter_chunks)
            total_chars += sum(len(c.content) for c in chapter_chunks)
            chapters.append({
                "chapter": self.extract_chapter_name(file_path),
                "chunks": len(chapter_chunks),
            })

        return {
            "total_files": len(files),
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "chapters": chapters,
        }
