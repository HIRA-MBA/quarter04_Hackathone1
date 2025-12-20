/**
 * DifficultyContent component - conditionally renders content based on user's difficulty settings.
 * Use these components in MDX files to show/hide content based on user experience level.
 *
 * Usage in MDX:
 * ```mdx
 * import { AdvancedContent, BeginnerTip, ExpandableCode } from '@site/src/components/Personalization/DifficultyContent';
 *
 * <BeginnerTip>
 *   This is a helpful tip for beginners!
 * </BeginnerTip>
 *
 * <AdvancedContent>
 *   This advanced content is only shown to intermediate/advanced users.
 * </AdvancedContent>
 *
 * <ExpandableCode title="Full implementation">
 *   ```python
 *   # Detailed code example
 *   ```
 * </ExpandableCode>
 * ```
 */

import React from 'react';
import { useDifficultyAdjustment } from './ChapterRenderer';
import styles from './DifficultyContent.module.css';

interface DifficultyContentProps {
  children: React.ReactNode;
}

interface ExpandableCodeProps {
  children: React.ReactNode;
  title?: string;
  defaultExpanded?: boolean;
}

/**
 * AdvancedContent - only shown to intermediate/advanced users.
 * Hidden for beginners to reduce cognitive load.
 */
export function AdvancedContent({ children }: DifficultyContentProps): React.ReactElement | null {
  const { showAdvanced } = useDifficultyAdjustment();

  if (!showAdvanced) {
    return (
      <div className={styles.hiddenAdvanced}>
        <span className={styles.hiddenLabel}>Advanced content available</span>
        <button
          className={styles.showButton}
          onClick={(e) => {
            const parent = (e.target as HTMLElement).closest(`.${styles.hiddenAdvanced}`);
            if (parent) {
              parent.classList.add(styles.revealed);
            }
          }}
        >
          Show anyway
        </button>
        <div className={styles.hiddenContent}>{children}</div>
      </div>
    );
  }

  return <div className={styles.advancedContent}>{children}</div>;
}

/**
 * BeginnerTip - helpful tips shown only to beginners.
 * Automatically hidden for intermediate/advanced users.
 */
export function BeginnerTip({ children }: DifficultyContentProps): React.ReactElement | null {
  const { showBeginnerTips } = useDifficultyAdjustment();

  if (!showBeginnerTips) {
    return null;
  }

  return (
    <div className={styles.beginnerTip}>
      <span className={styles.tipIcon}>💡</span>
      <div className={styles.tipContent}>{children}</div>
    </div>
  );
}

/**
 * ExpandableCode - collapsible code block, auto-expanded for beginners.
 * Collapsed by default for advanced users who want a quick scan.
 */
export function ExpandableCode({
  children,
  title = 'Show code',
  defaultExpanded,
}: ExpandableCodeProps): React.ReactElement {
  const { expandCodeExamples } = useDifficultyAdjustment();
  const [isExpanded, setIsExpanded] = React.useState(
    defaultExpanded ?? expandCodeExamples
  );

  return (
    <div className={styles.expandableCode}>
      <button
        className={styles.expandButton}
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <span className={styles.expandIcon}>{isExpanded ? '▼' : '▶'}</span>
        {title}
      </button>
      {isExpanded && <div className={styles.codeContent}>{children}</div>}
    </div>
  );
}

/**
 * PaceAware - wrapper that adds visual hints based on suggested pace.
 */
export function PaceAware({ children }: DifficultyContentProps): React.ReactElement {
  const { suggestedPace } = useDifficultyAdjustment();

  return (
    <div className={`${styles.paceAware} ${styles[suggestedPace] || ''}`}>
      {children}
    </div>
  );
}

export default { AdvancedContent, BeginnerTip, ExpandableCode, PaceAware };
