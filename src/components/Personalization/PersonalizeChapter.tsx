/**
 * PersonalizeChapter component - adds a personalization button at the top of chapters.
 * Generates a custom introduction based on user's background and experience level.
 */

import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { userApi } from '../../services/user';
import styles from './PersonalizeChapter.module.css';

interface PersonalizeChapterProps {
  chapterId: string;
  chapterTitle: string;
}

export default function PersonalizeChapter({
  chapterId,
  chapterTitle,
}: PersonalizeChapterProps): React.ReactElement | null {
  const { user, isAuthenticated, preferences } = useAuth();
  const [personalizedIntro, setPersonalizedIntro] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userLevel, setUserLevel] = useState<string | null>(null);

  // Only show for authenticated users with completed questionnaire
  if (!isAuthenticated || !user) {
    return null;
  }

  // Check if user has completed the questionnaire
  const hasCompletedQuestionnaire = preferences?.experience_level;

  const handlePersonalize = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await userApi.getPersonalizedIntro(chapterId, chapterTitle);
      setPersonalizedIntro(result.personalized_intro);
      setUserLevel(result.user_level);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to personalize');
    } finally {
      setLoading(false);
    }
  };

  const handleDismiss = () => {
    setPersonalizedIntro(null);
    setUserLevel(null);
  };

  if (!hasCompletedQuestionnaire) {
    return (
      <div className={styles.container}>
        <div className={styles.setupPrompt}>
          <span className={styles.icon}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </span>
          <span>Complete your profile to get personalized chapter introductions</span>
          <a href="/auth/questionnaire" className={styles.setupLink}>
            Set up now
          </a>
        </div>
      </div>
    );
  }

  if (personalizedIntro) {
    return (
      <div className={styles.container}>
        <div className={styles.personalizedBox}>
          <div className={styles.header}>
            <div className={styles.headerLeft}>
              <span className={styles.sparkle}>&#10024;</span>
              <span className={styles.title}>Personalized for You</span>
              {userLevel && (
                <span className={styles.levelBadge}>{userLevel}</span>
              )}
            </div>
            <button
              type="button"
              className={styles.dismissButton}
              onClick={handleDismiss}
              aria-label="Dismiss"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <div className={styles.introContent}>
            {personalizedIntro.split('\n').map((paragraph, index) => (
              <p key={index}>{paragraph}</p>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {error && (
        <div className={styles.error}>
          {error}
          <button
            type="button"
            className={styles.retryButton}
            onClick={handlePersonalize}
          >
            Retry
          </button>
        </div>
      )}
      <button
        type="button"
        className={styles.personalizeButton}
        onClick={handlePersonalize}
        disabled={loading}
      >
        {loading ? (
          <>
            <span className={styles.spinner} />
            Personalizing...
          </>
        ) : (
          <>
            <span className={styles.sparkle}>&#10024;</span>
            Personalize this chapter for me
          </>
        )}
      </button>
    </div>
  );
}
