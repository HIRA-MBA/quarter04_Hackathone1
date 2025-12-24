"""
Chat service for RAG chatbot.

Handles conversation management, context injection, and response generation
using OpenAI's chat completion API.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.chat_history import ChatHistory
from app.services.rag.embeddings import SearchResult, get_embeddings_service

SYSTEM_PROMPT = """You are an expert AI tutor for a textbook on Physical AI and Humanoid Robotics. Your knowledge comes exclusively from the textbook content provided in the context.

Guidelines:
1. ONLY answer based on the provided context from the textbook
2. If the context doesn't contain relevant information, say "I don't have information about that in the textbook. Could you ask about a topic covered in the chapters?"
3. When referencing information, mention the chapter and section it comes from
4. Provide clear, educational explanations suitable for students learning robotics
5. If asked about code, explain the concepts and refer to the code examples in the textbook
6. Be encouraging and supportive of the student's learning journey

Remember: Never make up information. Only use what's in the provided context."""


def build_personalized_system_prompt(preferences: dict | None) -> str:
    """Build a personalized system prompt based on user preferences."""
    base = SYSTEM_PROMPT

    if not preferences:
        return base

    personalization_lines = ["\n\nUser Personalization (adapt your responses accordingly):"]

    # Experience level adjustments
    level = preferences.get("experience_level", "beginner")

    if level == "beginner":
        personalization_lines.extend([
            "- Use simple, step-by-step explanations",
            "- Avoid jargon or explain technical terms when used",
            "- Provide analogies to everyday concepts when helpful",
            "- Be encouraging and patient with explanations",
            "- Break down complex concepts into smaller parts",
        ])
    elif level == "intermediate":
        personalization_lines.extend([
            "- Balance clarity with technical depth",
            "- Use standard robotics terminology",
            "- Reference related concepts the user might know",
            "- Provide context for why things work the way they do",
        ])
    else:  # advanced
        personalization_lines.extend([
            "- Be concise and technically precise",
            "- Focus on implementation details and edge cases",
            "- Reference advanced concepts directly without over-explaining basics",
            "- Discuss trade-offs and optimization strategies",
        ])

    # Programming language context
    langs = preferences.get("programming_languages", {})
    if langs:
        known_langs = []
        for lang, proficiency in langs.items():
            if proficiency and proficiency != "none":
                lang_name = {"python": "Python", "cpp": "C++", "javascript": "JavaScript"}.get(
                    lang, lang
                )
                known_langs.append(f"{lang_name} ({proficiency})")

        if known_langs:
            personalization_lines.append(f"- User knows: {', '.join(known_langs)}")
            personalization_lines.append(
                "- When explaining code concepts, relate them to the user's known languages"
            )

    return base + "\n".join(personalization_lines)


@dataclass
class ChatMessage:
    """A message in a chat conversation."""

    role: str  # system, user, assistant
    content: str


@dataclass
class ChatResponse:
    """Response from the chat service."""

    message: str
    sources: list[dict]
    session_id: str
    tokens_used: int = 0


@dataclass
class ConversationContext:
    """Maintains conversation state."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[ChatMessage] = field(default_factory=list)
    current_chapter: str | None = None
    user_preferences: dict | None = None

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append(ChatMessage(role=role, content=content))

    def get_messages_for_api(self, include_system: bool = True) -> list[dict]:
        """Get messages formatted for OpenAI API."""
        messages = []
        if include_system:
            # Use personalized prompt if preferences are available
            system_prompt = build_personalized_system_prompt(self.user_preferences)
            messages.append({"role": "system", "content": system_prompt})

        for msg in self.messages[-10:]:  # Keep last 10 messages for context
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


class ChatService:
    """Service for handling RAG-powered chat conversations."""

    def __init__(self):
        settings = get_settings()
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.chat_model
        self.embeddings_service = get_embeddings_service()
        self._conversations: dict[str, ConversationContext] = {}

    def get_or_create_conversation(
        self,
        session_id: str | None = None,
        chapter: str | None = None,
        user_preferences: dict | None = None,
    ) -> ConversationContext:
        """Get existing conversation or create new one."""
        if session_id and session_id in self._conversations:
            conv = self._conversations[session_id]
            if chapter:
                conv.current_chapter = chapter
            # Update preferences if provided
            if user_preferences:
                conv.user_preferences = user_preferences
            return conv

        conv = ConversationContext(
            session_id=session_id or str(uuid4()),
            current_chapter=chapter,
            user_preferences=user_preferences,
        )
        self._conversations[conv.session_id] = conv
        return conv

    def _build_context_prompt(self, query: str, context: str, sources: list[SearchResult]) -> str:
        """Build the user prompt with retrieved context."""
        if not context:
            return f"""User Question: {query}

Note: No relevant content was found in the textbook for this question."""

        source_list = "\n".join(f"- {s.chapter}: {s.section}" for s in sources)

        return f"""Based on the following textbook content:

{context}

---
Sources consulted:
{source_list}

---
User Question: {query}

Please provide a helpful answer based on the textbook content above."""

    def chat(
        self,
        query: str,
        session_id: str | None = None,
        chapter: str | None = None,
        user_preferences: dict | None = None,
    ) -> ChatResponse:
        """Process a chat message and return response."""
        # Get or create conversation
        conv = self.get_or_create_conversation(session_id, chapter, user_preferences)

        # Retrieve relevant context
        sources, context = self.embeddings_service.search_with_context(
            query=query,
            limit=5,
            chapter_filter=conv.current_chapter,
        )

        # Build prompt with context
        user_prompt = self._build_context_prompt(query, context, sources)

        # Add user message to conversation
        conv.add_message("user", query)

        # Prepare messages for API
        messages = conv.get_messages_for_api()
        # Replace the last user message with the context-enhanced version
        messages[-1] = {"role": "user", "content": user_prompt}

        # Call OpenAI
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )

        assistant_message = response.choices[0].message.content or ""

        # Add assistant response to conversation
        conv.add_message("assistant", assistant_message)

        return ChatResponse(
            message=assistant_message,
            sources=[s.to_dict() for s in sources],
            session_id=conv.session_id,
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )

    async def chat_stream(
        self,
        query: str,
        session_id: str | None = None,
        chapter: str | None = None,
        user_preferences: dict | None = None,
    ) -> AsyncIterator[str]:
        """Process a chat message and stream the response."""
        # Get or create conversation
        conv = self.get_or_create_conversation(session_id, chapter, user_preferences)

        # Retrieve relevant context
        sources, context = self.embeddings_service.search_with_context(
            query=query,
            limit=5,
            chapter_filter=conv.current_chapter,
        )

        # Build prompt with context
        user_prompt = self._build_context_prompt(query, context, sources)

        # Add user message to conversation
        conv.add_message("user", query)

        # Prepare messages for API
        messages = conv.get_messages_for_api()
        messages[-1] = {"role": "user", "content": user_prompt}

        # Stream response
        full_response = []
        stream = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response.append(content)
                yield content

        # Add complete response to conversation
        conv.add_message("assistant", "".join(full_response))

        # Yield sources at the end
        yield "\n\n---\nSources:\n"
        for source in sources:
            yield f"- {source.chapter}: {source.section}\n"

    def clear_conversation(self, session_id: str) -> bool:
        """Clear a conversation's history."""
        if session_id in self._conversations:
            self._conversations[session_id].clear()
            return True
        return False

    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation entirely."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False

    async def save_message_to_db(
        self,
        db: AsyncSession,
        session_id: str,
        role: str,
        content: str,
        chapter: str | None = None,
        sources: list | None = None,
    ) -> ChatHistory:
        """Persist a chat message to the database."""
        record = ChatHistory(
            session_id=session_id,
            role=role,
            content=content,
            chapter=chapter,
            sources=sources or [],
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return record

    async def get_history_from_db(
        self,
        db: AsyncSession,
        session_id: str,
        limit: int = 20,
    ) -> list[ChatHistory]:
        """Load chat history from the database."""
        result = await db.execute(
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def save_feedback(
        self,
        db: AsyncSession,
        message_id: str,
        rating: int,
        text: str | None = None,
    ) -> bool:
        """Save feedback for a message."""
        from uuid import UUID

        result = await db.execute(select(ChatHistory).where(ChatHistory.id == UUID(message_id)))
        record = result.scalar_one_or_none()
        if record:
            record.feedback_rating = rating
            record.feedback_text = text
            await db.commit()
            return True
        return False


# Singleton instance
_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """Get chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
