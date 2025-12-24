/**
 * TranslateButton component for translating chapter content to Urdu.
 * Replaces the page content with translated version when activated.
 */

import React, { useState, useCallback } from 'react';
import { useTranslation } from '../../context/TranslationContext';
import { translationApi } from '../../services/translation';
import styles from './TranslateButton.module.css';

interface TranslateButtonProps {
  chapterId: string;
}

export default function TranslateButton({
  chapterId,
}: TranslateButtonProps): React.ReactElement {
  const { language, setLanguage } = useTranslation();
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isUrdu = language === 'ur';

  const handleToggleLanguage = useCallback(async () => {
    if (isTranslating) return;

    setError(null);

    if (isUrdu) {
      // Switch back to English
      setLanguage('en');
      return;
    }

    // Start translation process
    setIsTranslating(true);

    try {
      // Get the main content element
      const contentElement = document.querySelector('.markdown');
      if (!contentElement) {
        throw new Error('Content not found');
      }

      // Get text content for translation (excluding code blocks and buttons)
      const textNodes: { element: Element; originalText: string }[] = [];
      const walker = document.createTreeWalker(
        contentElement,
        NodeFilter.SHOW_ELEMENT,
        {
          acceptNode: (node: Element) => {
            const tagName = node.tagName?.toLowerCase();
            // Skip code blocks, pre, buttons, and our own components
            if (
              tagName === 'code' ||
              tagName === 'pre' ||
              tagName === 'button' ||
              tagName === 'script' ||
              tagName === 'style' ||
              node.classList?.contains(styles.container)
            ) {
              return NodeFilter.FILTER_REJECT;
            }
            return NodeFilter.FILTER_ACCEPT;
          },
        }
      );

      // Collect paragraphs and headings for translation
      const elementsToTranslate: Element[] = [];
      let currentNode = walker.nextNode() as Element | null;
      while (currentNode) {
        const tagName = currentNode.tagName?.toLowerCase();
        if (
          (tagName === 'p' ||
            tagName === 'li' ||
            tagName === 'h1' ||
            tagName === 'h2' ||
            tagName === 'h3' ||
            tagName === 'h4') &&
          currentNode.textContent?.trim()
        ) {
          elementsToTranslate.push(currentNode);
        }
        currentNode = walker.nextNode() as Element | null;
      }

      // Translate each element
      for (const element of elementsToTranslate.slice(0, 20)) {
        // Limit to first 20 elements
        const originalText = element.textContent || '';
        if (originalText.length > 10) {
          // Only translate substantial text
          try {
            const result = await translationApi.translateText(
              originalText,
              'ur',
              chapterId
            );
            // Store original and update with translation
            element.setAttribute('data-original-text', originalText);
            element.textContent = result.translated_text;
          } catch {
            // Continue with other elements if one fails
            console.warn('Failed to translate element:', originalText.substring(0, 50));
          }
        }
      }

      // Switch to Urdu mode
      setLanguage('ur');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Translation failed');
    } finally {
      setIsTranslating(false);
    }
  }, [isUrdu, isTranslating, setLanguage, chapterId]);

  // Handle switching back to English
  const handleShowEnglish = useCallback(() => {
    // Restore original text
    const contentElement = document.querySelector('.markdown');
    if (contentElement) {
      const elementsWithOriginal = contentElement.querySelectorAll('[data-original-text]');
      elementsWithOriginal.forEach((element) => {
        const originalText = element.getAttribute('data-original-text');
        if (originalText) {
          element.textContent = originalText;
          element.removeAttribute('data-original-text');
        }
      });
    }
    setLanguage('en');
  }, [setLanguage]);

  return (
    <div className={styles.container}>
      {error && (
        <span className={styles.error} title={error}>
          !
        </span>
      )}
      <button
        type="button"
        className={`${styles.translateButton} ${isUrdu ? styles.active : ''}`}
        onClick={isUrdu ? handleShowEnglish : handleToggleLanguage}
        disabled={isTranslating}
        title={isUrdu ? 'Show in English' : 'Translate to Urdu'}
      >
        {isTranslating ? (
          <>
            <span className={styles.spinner} />
            <span>Translating...</span>
          </>
        ) : isUrdu ? (
          <>
            <span className={styles.icon}>EN</span>
            <span>English</span>
          </>
        ) : (
          <>
            <span className={styles.icon}>اردو</span>
            <span>Translate</span>
          </>
        )}
      </button>
    </div>
  );
}
