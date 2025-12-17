/**
 * Language toggle component for switching between English and Urdu.
 */

import React, { useState } from 'react';
import { useTranslation } from '../../context/TranslationContext';
import styles from './UrduToggle.module.css';

interface UrduToggleProps {
  onLanguageChange?: (language: 'en' | 'ur') => void;
}

export default function UrduToggle({
  onLanguageChange,
}: UrduToggleProps): React.ReactElement {
  const { language, setLanguage, isTranslating } = useTranslation();
  const [isLoading, setIsLoading] = useState(false);

  const handleToggle = async () => {
    const newLanguage = language === 'en' ? 'ur' : 'en';
    setIsLoading(true);

    try {
      setLanguage(newLanguage);
      onLanguageChange?.(newLanguage);
    } finally {
      setIsLoading(false);
    }
  };

  const loading = isLoading || isTranslating;

  return (
    <div className={styles.container}>
      <button
        type="button"
        className={`${styles.toggleButton} ${language === 'ur' ? styles.urduActive : ''}`}
        onClick={handleToggle}
        disabled={loading}
        aria-label={`Switch to ${language === 'en' ? 'Urdu' : 'English'}`}
        title={language === 'en' ? 'اردو میں دیکھیں' : 'View in English'}
      >
        <span className={styles.languageIcon}>
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
        </span>
        <span className={styles.languageText}>
          {language === 'en' ? 'اردو' : 'EN'}
        </span>
        {loading && <span className={styles.loading} />}
      </button>
    </div>
  );
}

/**
 * Hook for managing translation state in a component.
 * Re-exported from context for convenience.
 */
export { useTranslation } from '../../context/TranslationContext';
