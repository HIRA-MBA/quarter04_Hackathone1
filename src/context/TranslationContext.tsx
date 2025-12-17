/**
 * Translation context provider for managing language state across the app.
 */

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { translationApi } from '../services/translation';

type Language = 'en' | 'ur';

interface TranslationContextValue {
  language: Language;
  setLanguage: (lang: Language) => void;
  translateText: (text: string, chapterId?: string) => Promise<string>;
  isTranslating: boolean;
}

const TranslationContext = createContext<TranslationContextValue | null>(null);

interface TranslationProviderProps {
  children: ReactNode;
}

export function TranslationProvider({ children }: TranslationProviderProps): React.ReactElement {
  const [language, setLanguageState] = useState<Language>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('preferred_language') as Language) || 'en';
    }
    return 'en';
  });
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationCache] = useState<Map<string, string>>(new Map());

  // Apply RTL/LTR styling when language changes
  useEffect(() => {
    if (typeof document !== 'undefined') {
      if (language === 'ur') {
        document.documentElement.setAttribute('dir', 'rtl');
        document.documentElement.setAttribute('lang', 'ur');
        document.body.classList.add('urdu-active');
      } else {
        document.documentElement.setAttribute('dir', 'ltr');
        document.documentElement.setAttribute('lang', 'en');
        document.body.classList.remove('urdu-active');
      }
    }
  }, [language]);

  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang);
    if (typeof window !== 'undefined') {
      localStorage.setItem('preferred_language', lang);
    }
  }, []);

  const translateText = useCallback(async (text: string, chapterId?: string): Promise<string> => {
    if (language === 'en' || !text || text.trim().length === 0) {
      return text;
    }

    // Check local cache first
    const cacheKey = `${language}:${text.substring(0, 100)}`;
    const cached = translationCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    setIsTranslating(true);
    try {
      const result = await translationApi.translateText(text, 'ur', chapterId);
      translationCache.set(cacheKey, result.translated_text);
      return result.translated_text;
    } catch (error) {
      console.error('Translation error:', error);
      return text; // Return original text on error
    } finally {
      setIsTranslating(false);
    }
  }, [language, translationCache]);

  return (
    <TranslationContext.Provider
      value={{
        language,
        setLanguage,
        translateText,
        isTranslating,
      }}
    >
      {children}
    </TranslationContext.Provider>
  );
}

export function useTranslation(): TranslationContextValue {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslation must be used within a TranslationProvider');
  }
  return context;
}

export { TranslationContext };
