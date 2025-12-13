/**
 * Language toggle component for switching between English and Urdu.
 */

import React, { useState, useEffect } from 'react';
import { translationApi } from '../../services/translation';
import styles from './UrduToggle.module.css';

interface UrduToggleProps {
  onLanguageChange?: (language: 'en' | 'ur') => void;
  initialLanguage?: 'en' | 'ur';
}

export default function UrduToggle({
  onLanguageChange,
  initialLanguage = 'en',
}: UrduToggleProps): React.ReactElement {
  const [language, setLanguage] = useState<'en' | 'ur'>(initialLanguage);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Apply RTL styling when Urdu is selected
    if (language === 'ur') {
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ur');
    } else {
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', 'en');
    }
  }, [language]);

  const handleToggle = async () => {
    const newLanguage = language === 'en' ? 'ur' : 'en';
    setIsLoading(true);

    try {
      setLanguage(newLanguage);
      onLanguageChange?.(newLanguage);

      // Store preference in localStorage
      localStorage.setItem('preferred_language', newLanguage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <button
        type="button"
        className={`${styles.toggleButton} ${language === 'ur' ? styles.urduActive : ''}`}
        onClick={handleToggle}
        disabled={isLoading}
        aria-label={`Switch to ${language === 'en' ? 'Urdu' : 'English'}`}
        title={language === 'en' ? 'اردو میں دیکھیں' : 'View in English'}
      >
        <span className={styles.languageIcon}>
          {language === 'en' ? (
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          ) : (
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
            </svg>
          )}
        </span>
        <span className={styles.languageText}>
          {language === 'en' ? 'اردو' : 'EN'}
        </span>
        {isLoading && <span className={styles.loading} />}
      </button>
    </div>
  );
}

/**
 * Hook for managing translation state in a component.
 */
export function useTranslation() {
  const [language, setLanguage] = useState<'en' | 'ur'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('preferred_language') as 'en' | 'ur') || 'en';
    }
    return 'en';
  });

  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedContent, setTranslatedContent] = useState<Map<string, string>>(new Map());

  const translateText = async (text: string, chapterId?: string): Promise<string> => {
    if (language === 'en') {
      return text;
    }

    // Check cache
    const cached = translatedContent.get(text);
    if (cached) {
      return cached;
    }

    setIsTranslating(true);
    try {
      const result = await translationApi.translateText(text, 'ur', chapterId);
      setTranslatedContent((prev) => new Map(prev).set(text, result.translated_text));
      return result.translated_text;
    } catch (error) {
      console.error('Translation error:', error);
      return text;
    } finally {
      setIsTranslating(false);
    }
  };

  const setPreferredLanguage = (newLanguage: 'en' | 'ur') => {
    setLanguage(newLanguage);
    if (typeof window !== 'undefined') {
      localStorage.setItem('preferred_language', newLanguage);
    }
  };

  return {
    language,
    isTranslating,
    translateText,
    setLanguage: setPreferredLanguage,
  };
}
