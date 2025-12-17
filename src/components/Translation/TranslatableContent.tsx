/**
 * Component that translates its text content when Urdu is selected.
 */

import React, { useState, useEffect, ReactNode } from 'react';
import { useTranslation } from '../../context/TranslationContext';

interface TranslatableContentProps {
  children: ReactNode;
  chapterId?: string;
  as?: keyof JSX.IntrinsicElements;
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
  const [originalText, setOriginalText] = useState<string>('');

  // Extract text from children
  useEffect(() => {
    if (typeof children === 'string') {
      setOriginalText(children);
    } else if (React.isValidElement(children) && typeof children.props.children === 'string') {
      setOriginalText(children.props.children);
    }
  }, [children]);

  // Translate when language changes
  useEffect(() => {
    if (language === 'ur' && originalText) {
      translateText(originalText, chapterId).then(setTranslatedContent);
    } else {
      setTranslatedContent(null);
    }
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
