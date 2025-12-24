/**
 * Swizzled DocItem/Content component.
 * Adds PersonalizeChapter button and TranslateButton at the top of each doc page.
 */

import React from 'react';
import clsx from 'clsx';
import { ThemeClassNames } from '@docusaurus/theme-common';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import Heading from '@theme/Heading';
import MDXContent from '@theme/MDXContent';
import type { Props } from '@theme/DocItem/Content';

import PersonalizeChapter from '@site/src/components/Personalization/PersonalizeChapter';
import TranslateButton from '@site/src/components/Translation/TranslateButton';

function useSyntheticTitle(): string | null {
  const { metadata, frontMatter, contentTitle } = useDoc();
  const shouldRender =
    !frontMatter.hide_title && typeof contentTitle === 'undefined';
  if (!shouldRender) {
    return null;
  }
  return metadata.title;
}

export default function DocItemContent({ children }: Props): React.ReactElement {
  const syntheticTitle = useSyntheticTitle();
  const { metadata } = useDoc();

  // Extract chapter ID from the doc path (e.g., "module-1-ros2/ch01-welcome-first-node")
  const chapterId = metadata.id || '';
  const chapterTitle = metadata.title || '';

  // Check if this is a chapter (not front/back matter)
  const isChapter = chapterId.includes('/ch') || chapterId.startsWith('ch');

  return (
    <div className={clsx(ThemeClassNames.docs.docMarkdown, 'markdown')}>
      {syntheticTitle ? (
        <header>
          <Heading as="h1">{syntheticTitle}</Heading>
        </header>
      ) : null}

      {/* Add personalization and translation buttons for chapters */}
      {isChapter && (
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <PersonalizeChapter chapterId={chapterId} chapterTitle={chapterTitle} />
            <TranslateButton chapterId={chapterId} />
          </div>
        </div>
      )}

      <MDXContent>{children}</MDXContent>
    </div>
  );
}
