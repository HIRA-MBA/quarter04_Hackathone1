/**
 * Translation API client for Urdu language support.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface TranslationResponse {
  original_text: string;
  translated_text: string;
  target_language: string;
  cached: boolean;
}

export interface TranslationStatus {
  chapter_id: string;
  total_segments: number;
  completed: number;
  pending: number;
  failed: number;
  completion_percentage: number;
}

export interface SupportedLanguage {
  code: string;
  name: string;
  native_name: string;
  direction?: string;
}

class TranslationApiClient {
  private baseUrl: string;
  private cache: Map<string, string> = new Map();

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Generate a cache key for local caching.
   */
  private getCacheKey(text: string, targetLanguage: string): string {
    return `${targetLanguage}:${text.substring(0, 100)}`;
  }

  /**
   * Translate a text segment.
   */
  async translateText(
    text: string,
    targetLanguage: string = 'ur',
    chapterId?: string
  ): Promise<TranslationResponse> {
    // Check local cache first
    const cacheKey = this.getCacheKey(text, targetLanguage);
    const cached = this.cache.get(cacheKey);
    if (cached) {
      return {
        original_text: text,
        translated_text: cached,
        target_language: targetLanguage,
        cached: true,
      };
    }

    const response = await fetch(`${this.baseUrl}/api/translation/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        target_language: targetLanguage,
        chapter_id: chapterId,
      }),
    });

    if (!response.ok) {
      throw new Error('Translation failed');
    }

    const result: TranslationResponse = await response.json();

    // Store in local cache
    this.cache.set(cacheKey, result.translated_text);

    return result;
  }

  /**
   * Translate entire chapter content.
   */
  async translateChapter(
    chapterId: string,
    title: string,
    content: string,
    sections?: Array<{ id: string; title: string; content: string }>
  ): Promise<{
    chapter_id: string;
    translated: {
      title?: string;
      content?: string;
      sections?: Array<{ id: string; title: string; content: string }>;
    };
    target_language: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/translation/translate/chapter`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        chapter_id: chapterId,
        title,
        content,
        sections,
      }),
    });

    if (!response.ok) {
      throw new Error('Chapter translation failed');
    }

    return response.json();
  }

  /**
   * Get translation status for a chapter.
   */
  async getTranslationStatus(chapterId: string): Promise<TranslationStatus> {
    const response = await fetch(
      `${this.baseUrl}/api/translation/status/${encodeURIComponent(chapterId)}`
    );

    if (!response.ok) {
      throw new Error('Failed to get translation status');
    }

    return response.json();
  }

  /**
   * Get supported languages.
   */
  async getSupportedLanguages(): Promise<{
    source_languages: SupportedLanguage[];
    target_languages: SupportedLanguage[];
  }> {
    const response = await fetch(`${this.baseUrl}/api/translation/languages`);

    if (!response.ok) {
      throw new Error('Failed to get languages');
    }

    return response.json();
  }

  /**
   * Clear local cache.
   */
  clearCache(): void {
    this.cache.clear();
  }
}

// Export singleton instance
export const translationApi = new TranslationApiClient();

// Export class for testing
export { TranslationApiClient };
