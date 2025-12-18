"""Translation service for Urdu language support using OpenAI."""

import hashlib

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.translation_cache import TranslationCache

settings = get_settings()


class TranslationService:
    """Service for translating content to Urdu using OpenAI."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    @staticmethod
    def generate_content_hash(content: str, target_language: str) -> str:
        """Generate a hash for caching purposes."""
        hash_input = f"{target_language}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    async def get_cached_translation(
        self,
        content_hash: str,
    ) -> TranslationCache | None:
        """Get a cached translation if available."""
        result = await self.db.execute(
            select(TranslationCache).where(
                TranslationCache.content_hash == content_hash,
                TranslationCache.status == "completed",
            )
        )
        return result.scalar_one_or_none()

    async def save_translation_cache(
        self,
        content_hash: str,
        source_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
        chapter_id: str | None = None,
    ) -> TranslationCache:
        """Save a translation to the cache."""
        cache_entry = TranslationCache(
            content_hash=content_hash,
            source_text=source_text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            chapter_id=chapter_id,
            status="completed",
        )
        self.db.add(cache_entry)
        await self.db.commit()
        await self.db.refresh(cache_entry)
        return cache_entry

    async def translate_text(
        self,
        text: str,
        target_language: str = "ur",
        source_language: str = "en",
        chapter_id: str | None = None,
        use_cache: bool = True,
    ) -> str:
        """
        Translate text to the target language.

        Args:
            text: The source text to translate
            target_language: Target language code (default: 'ur' for Urdu)
            source_language: Source language code (default: 'en' for English)
            chapter_id: Optional chapter identifier for cache organization
            use_cache: Whether to use cache (default: True)

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        # Check cache first
        content_hash = self.generate_content_hash(text, target_language)

        if use_cache:
            cached = await self.get_cached_translation(content_hash)
            if cached:
                return cached.translated_text

        # Translate using OpenAI
        language_names = {
            "ur": "Urdu",
            "en": "English",
        }
        target_lang_name = language_names.get(target_language, target_language)

        system_prompt = f"""You are a professional translator specializing in technical and educational content.
Translate the following text from English to {target_lang_name}.

Guidelines:
1. Maintain technical accuracy - keep code snippets, commands, and technical terms unchanged
2. Preserve markdown formatting (headers, lists, code blocks, links)
3. Keep proper nouns and brand names in their original form
4. Use formal academic style appropriate for a textbook
5. Ensure the translation sounds natural to native {target_lang_name} speakers
6. Preserve any HTML tags or special formatting markers

For Urdu specifically:
- Use proper Urdu script (not Roman Urdu)
- Technical terms can be kept in English with Urdu transliteration in parentheses where helpful
- Maintain right-to-left text flow for prose content
- Code examples should remain left-to-right"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this text:\n\n{text}"},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            translated_text = response.choices[0].message.content or text

            # Cache the translation
            if use_cache:
                await self.save_translation_cache(
                    content_hash=content_hash,
                    source_text=text,
                    translated_text=translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    chapter_id=chapter_id,
                )

            return translated_text

        except Exception as e:
            # Log error and return original text
            print(f"Translation error: {e}")
            return text

    async def translate_chapter_content(
        self,
        chapter_content: dict,
        target_language: str = "ur",
    ) -> dict:
        """
        Translate structured chapter content.

        Args:
            chapter_content: Dict with 'title', 'sections', 'content' keys
            target_language: Target language code

        Returns:
            Dict with translated content
        """
        translated = {}

        # Translate title
        if "title" in chapter_content:
            translated["title"] = await self.translate_text(
                chapter_content["title"],
                target_language,
                chapter_id=chapter_content.get("chapter_id"),
            )

        # Translate main content
        if "content" in chapter_content:
            translated["content"] = await self.translate_text(
                chapter_content["content"],
                target_language,
                chapter_id=chapter_content.get("chapter_id"),
            )

        # Translate sections
        if "sections" in chapter_content:
            translated["sections"] = []
            for section in chapter_content["sections"]:
                translated_section = {
                    "id": section.get("id"),
                    "title": await self.translate_text(
                        section.get("title", ""),
                        target_language,
                        chapter_id=chapter_content.get("chapter_id"),
                    ),
                    "content": await self.translate_text(
                        section.get("content", ""),
                        target_language,
                        chapter_id=chapter_content.get("chapter_id"),
                    ),
                }
                translated["sections"].append(translated_section)

        return translated

    async def get_translation_status(self, chapter_id: str) -> dict:
        """Get translation status for a chapter."""
        result = await self.db.execute(
            select(TranslationCache).where(
                TranslationCache.chapter_id == chapter_id,
            )
        )
        entries = result.scalars().all()

        total = len(entries)
        completed = sum(1 for e in entries if e.status == "completed")
        pending = sum(1 for e in entries if e.status == "pending")
        failed = sum(1 for e in entries if e.status == "failed")

        return {
            "chapter_id": chapter_id,
            "total_segments": total,
            "completed": completed,
            "pending": pending,
            "failed": failed,
            "completion_percentage": round(completed / total * 100, 1) if total > 0 else 0,
        }

    async def queue_chapter_translation(
        self,
        chapter_id: str,
        content_segments: list[str],
        target_language: str = "ur",
    ) -> list[str]:
        """
        Queue chapter segments for background translation.

        Args:
            chapter_id: Chapter identifier
            content_segments: List of content segments to translate
            target_language: Target language code

        Returns:
            List of content hashes for tracking
        """
        hashes = []

        for segment in content_segments:
            content_hash = self.generate_content_hash(segment, target_language)

            # Check if already cached
            existing = await self.get_cached_translation(content_hash)
            if existing:
                hashes.append(content_hash)
                continue

            # Create pending entry
            cache_entry = TranslationCache(
                content_hash=content_hash,
                source_text=segment,
                translated_text="",
                source_language="en",
                target_language=target_language,
                chapter_id=chapter_id,
                status="pending",
            )
            self.db.add(cache_entry)
            hashes.append(content_hash)

        await self.db.commit()
        return hashes
