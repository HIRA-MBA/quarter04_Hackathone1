from uuid import UUID

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class ChatHistory(Base, UUIDMixin, TimestampMixin):
    """Chat history for RAG conversations."""

    __tablename__ = "chat_history"

    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    session_id: Mapped[str] = mapped_column(String(64), index=True)

    # Message content
    role: Mapped[str] = mapped_column(String(20))  # user, assistant, system
    content: Mapped[str] = mapped_column(Text)

    # Context
    chapter: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sources: Mapped[list] = mapped_column(JSONB, default=list)

    # Feedback
    feedback_rating: Mapped[int | None] = mapped_column(nullable=True)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User | None"] = relationship(back_populates="chat_history")


from app.models.user import User  # noqa: E402
