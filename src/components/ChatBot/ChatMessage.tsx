import React from 'react';
import styles from './ChatBot.module.css';

export interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    chapter: string;
    section: string;
  }>;
  isLoading?: boolean;
}

export default function ChatMessage({
  role,
  content,
  sources,
  isLoading,
}: ChatMessageProps): JSX.Element {
  return (
    <div className={`${styles.message} ${styles[role]}`}>
      <div className={styles.messageAvatar}>
        {role === 'user' ? '👤' : '🤖'}
      </div>
      <div className={styles.messageContent}>
        {isLoading ? (
          <div className={styles.loadingDots}>
            <span></span>
            <span></span>
            <span></span>
          </div>
        ) : (
          <>
            <div className={styles.messageText}>{content}</div>
            {sources && sources.length > 0 && (
              <div className={styles.sources}>
                <span className={styles.sourcesLabel}>Sources:</span>
                <ul className={styles.sourcesList}>
                  {sources.map((source, index) => (
                    <li key={index}>
                      {source.chapter}: {source.section}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
