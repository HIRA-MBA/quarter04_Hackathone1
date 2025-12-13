from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class TranslationCache(Base, UUIDMixin, TimestampMixin):
    """Cache for translated content."""

    __tablename__ = "translation_cache"

    # Source identification
    source_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_language: Mapped[str] = mapped_column(String(10))
    target_language: Mapped[str] = mapped_column(String(10))

    # Content
    source_text: Mapped[str] = mapped_column(Text)
    translated_text: Mapped[str] = mapped_column(Text)

    # Metadata
    chapter: Mapped[str | None] = mapped_column(String(100), nullable=True)
    section: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Quality
    is_reviewed: Mapped[bool] = mapped_column(default=False)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
