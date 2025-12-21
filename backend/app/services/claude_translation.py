"""Claude-powered translation service for English-Urdu translation."""

import hashlib
from typing import Literal

import anthropic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.translation_cache import TranslationCache

settings = get_settings()


class ClaudeTranslationService:
    """Service for translating content using Claude (Anthropic)."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    @staticmethod
    def generate_content_hash(content: str, target_language: str) -> str:
        """Generate a hash for caching purposes."""
        hash_input = f"claude:{target_language}:{content}"
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
        target_language: Literal["ur", "en"] = "ur",
        source_language: Literal["en", "ur"] = "en",
        chapter_id: str | None = None,
        use_cache: bool = True,
    ) -> str:
        """
        Translate text using Claude.

        Args:
            text: The source text to translate
            target_language: Target language code ('ur' for Urdu, 'en' for English)
            source_language: Source language code ('en' for English, 'ur' for Urdu)
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

        # Build translation prompt
        language_names = {
            "ur": "Urdu",
            "en": "English",
        }
        source_lang_name = language_names.get(source_language, source_language)
        target_lang_name = language_names.get(target_language, target_language)

        # Specialized system prompt for Claude
        system_prompt = self._build_system_prompt(source_lang_name, target_lang_name)

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Translate this text:\n\n{text}",
                    }
                ],
            )

            translated_text = message.content[0].text if message.content else text

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

        except anthropic.APIError as e:
            print(f"Claude translation error: {e}")
            return text

    def _build_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """Build the system prompt for translation."""
        base_prompt = f"""You are an expert translator specializing in educational and technical content translation between {source_lang} and {target_lang}.

Your task is to translate text while following these strict guidelines:

## Core Translation Rules

1. **Accuracy First**: Preserve the exact meaning and intent of the source text
2. **Natural Flow**: The translation should read naturally to native speakers
3. **Consistency**: Use consistent terminology throughout

## Technical Content Handling

- **Code Blocks**: Keep all code snippets exactly as-is, do not translate
- **Technical Terms**: Keep widely-used technical terms in English (API, CPU, GPU, ROS, etc.)
- **Commands**: Keep shell commands, file paths, and package names unchanged
- **Variables/Functions**: Never translate code identifiers

## Formatting Preservation

- Preserve all markdown formatting (headers, lists, bold, italic, links)
- Keep HTML tags intact
- Maintain paragraph structure and line breaks
- Preserve numbering and bullet points"""

        if target_lang == "Urdu":
            urdu_specific = """

## Urdu-Specific Guidelines

1. **Script**: Always use proper Urdu Nastaliq/Naskh script (never Roman Urdu)
2. **Technical Terms**: You may add transliteration in parentheses for complex terms
   - Example: "machine learning (مشین لرننگ)"
3. **Direction**: Prose content flows right-to-left, code stays left-to-right
4. **Numbers**: Use standard Arabic numerals (0-9)
5. **Formality**: Use formal academic Urdu appropriate for textbooks
6. **Diacritics**: Use diacritical marks (اعراب) only when necessary for clarity"""
            base_prompt += urdu_specific

        elif target_lang == "English":
            english_specific = """

## English-Specific Guidelines

1. **Clarity**: Use clear, standard American English
2. **Idioms**: Translate meaning rather than word-for-word
3. **Register**: Match the formality level of the source
4. **Cultural Context**: Add brief clarifying notes if needed for cultural references"""
            base_prompt += english_specific

        base_prompt += """

## Output Format

Return ONLY the translated text. Do not include:
- Explanations of your translation choices
- Notes about the translation process
- The original text
- Any meta-commentary

Begin translation immediately."""

        return base_prompt

    async def translate_with_context(
        self,
        text: str,
        context: str,
        target_language: Literal["ur", "en"] = "ur",
        source_language: Literal["en", "ur"] = "en",
    ) -> str:
        """
        Translate text with additional context for better accuracy.

        Args:
            text: The source text to translate
            context: Additional context about the content (e.g., chapter topic)
            target_language: Target language code
            source_language: Source language code

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text

        language_names = {
            "ur": "Urdu",
            "en": "English",
        }
        target_lang_name = language_names.get(target_language, target_language)

        system_prompt = self._build_system_prompt(
            language_names.get(source_language, source_language),
            target_lang_name,
        )

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nTranslate this text:\n\n{text}",
                    }
                ],
            )

            return message.content[0].text if message.content else text

        except anthropic.APIError as e:
            print(f"Claude translation error: {e}")
            return text

    async def translate_batch(
        self,
        texts: list[str],
        target_language: Literal["ur", "en"] = "ur",
        source_language: Literal["en", "ur"] = "en",
        chapter_id: str | None = None,
    ) -> list[str]:
        """
        Translate multiple text segments efficiently.

        Args:
            texts: List of text segments to translate
            target_language: Target language code
            source_language: Source language code
            chapter_id: Optional chapter identifier

        Returns:
            List of translated texts
        """
        results = []
        for text in texts:
            translated = await self.translate_text(
                text=text,
                target_language=target_language,
                source_language=source_language,
                chapter_id=chapter_id,
            )
            results.append(translated)
        return results

    async def translate_chapter_content(
        self,
        chapter_content: dict,
        target_language: Literal["ur", "en"] = "ur",
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
        chapter_id = chapter_content.get("chapter_id")

        # Translate title
        if "title" in chapter_content:
            translated["title"] = await self.translate_text(
                chapter_content["title"],
                target_language,
                chapter_id=chapter_id,
            )

        # Translate main content
        if "content" in chapter_content:
            translated["content"] = await self.translate_text(
                chapter_content["content"],
                target_language,
                chapter_id=chapter_id,
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
                        chapter_id=chapter_id,
                    ),
                    "content": await self.translate_text(
                        section.get("content", ""),
                        target_language,
                        chapter_id=chapter_id,
                    ),
                }
                translated["sections"].append(translated_section)

        return translated
