from uuid import UUID

from sqlalchemy import ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class UserPreference(Base, UUIDMixin, TimestampMixin):
    """User preferences for personalization."""

    __tablename__ = "user_preferences"

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))

    # Language preference
    language: Mapped[str] = mapped_column(String(10), default="en")

    # Background questionnaire responses
    experience_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
    background: Mapped[str | None] = mapped_column(Text, nullable=True)
    goals: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Programming language proficiency (Python, C++, JavaScript)
    # Format: {"python": "intermediate", "cpp": "beginner", "javascript": "none"}

    programming_languages: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),  # This makes Alembic happy
    )
    # Progress tracking
    completed_chapters: Mapped[dict] = mapped_column(JSONB, default=dict)
    bookmarks: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Personalization settings
    theme: Mapped[str] = mapped_column(String(20), default="system")
    font_size: Mapped[str] = mapped_column(String(20), default="medium")

    # Relationships
    user: Mapped["User"] = relationship(back_populates="preferences")


from app.models.user import User  # noqa: E402
