/**
 * Component that translates its text content when Urdu is selected.
 */

import React, { useState, useEffect, useMemo, ReactNode, ElementType } from 'react';
import { useTranslation } from '../../context/TranslationContext';

interface TranslatableContentProps {
  children: ReactNode;
  chapterId?: string;
  as?: ElementType;
  className?: string;
}

/**
 * Wraps text content and translates it when Urdu language is selected.
 * Only translates text nodes, preserves HTML structure.
 */
export default function TranslatableContent({
  children,
  chapterId,
  as: Component = 'span',
  className,
}: TranslatableContentProps): React.ReactElement {
  const { language, translateText, isTranslating } = useTranslation();
  const [translatedContent, setTranslatedContent] = useState<string | null>(null);

  // Extract text from children using useMemo (not setState in effect)
  const originalText = useMemo(() => {
    if (typeof children === 'string') {
      return children;
    } else if (React.isValidElement(children)) {
      const childProps = children.props as { children?: unknown };
      if (typeof childProps.children === 'string') {
        return childProps.children;
      }
    }
    return '';
  }, [children]);

  // Handle translation when language or text changes
  useEffect(() => {
    let cancelled = false;

    if (language === 'ur' && originalText) {
      translateText(originalText, chapterId).then((result) => {
        if (!cancelled) {
          setTranslatedContent(result);
        }
      });
    }

    // Cleanup: reset translation when not in Urdu mode
    return () => {
      cancelled = true;
      if (language !== 'ur') {
        setTranslatedContent(null);
      }
    };
  }, [language, originalText, chapterId, translateText]);

  // If we have translated content and language is Urdu, show it
  if (language === 'ur' && translatedContent) {
    return (
      <Component className={className} dir="rtl">
        {translatedContent}
      </Component>
    );
  }

  // Show loading state
  if (language === 'ur' && isTranslating) {
    return (
      <Component className={className}>
        {children}
        <span style={{ opacity: 0.5 }}> (translating...)</span>
      </Component>
    );
  }

  // Default: show original content
  if (typeof children === 'string') {
    return <Component className={className}>{children}</Component>;
  }

  return <>{children}</>;
}

export { TranslatableContent };
