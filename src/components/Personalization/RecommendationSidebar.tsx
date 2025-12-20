/**
 * RecommendationSidebar component - displays personalized chapter recommendations.
 * Shows recommended next chapters based on user's progress and experience level.
 */

import React, { useEffect, useState } from 'react';
import { authApi } from '../../services/auth';
import { userApi, Recommendation } from '../../services/user';
import ProgressTracker from './ProgressTracker';
import styles from './RecommendationSidebar.module.css';

interface RecommendationSidebarProps {
  currentChapterId?: string;
  maxRecommendations?: number;
  showProgress?: boolean;
}

export default function RecommendationSidebar({
  currentChapterId,
  maxRecommendations = 3,
  showProgress = true,
}: RecommendationSidebarProps): React.ReactElement | null {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadRecommendations = async () => {
      const user = await authApi.getCurrentUser();
      if (user) {
        setIsAuthenticated(true);
        try {
          const recs = await userApi.getRecommendations();
          // Filter out current chapter if viewing one
          const filtered = currentChapterId
            ? recs.filter((r) => r.chapter_id !== currentChapterId)
            : recs;
          setRecommendations(filtered.slice(0, maxRecommendations));
        } catch (error) {
          console.error('Failed to load recommendations:', error);
        }
      }
      setLoading(false);
    };

    loadRecommendations();
  }, [currentChapterId, maxRecommendations]);

  if (!isAuthenticated || loading) {
    return null;
  }

  const getDifficultyColor = (match: number): string => {
    if (match >= 0.9) return '#4caf50';
    if (match >= 0.7) return '#ff9800';
    return '#f44336';
  };

  const getDifficultyLabel = (match: number): string => {
    if (match >= 0.9) return 'Perfect fit';
    if (match >= 0.7) return 'Good match';
    return 'Challenge';
  };

  return (
    <aside className={styles.sidebar}>
      {/* Recommendations section */}
      <div className={styles.section}>
        <h4 className={styles.sectionTitle}>Recommended Next</h4>
        {recommendations.length === 0 ? (
          <p className={styles.emptyState}>
            Complete more chapters to get personalized recommendations.
          </p>
        ) : (
          <ul className={styles.recommendationList}>
            {recommendations.map((rec) => (
              <li key={rec.chapter_id} className={styles.recommendationItem}>
                <a href={`/docs/${rec.chapter_id}`} className={styles.recommendationLink}>
                  <div className={styles.recommendationHeader}>
                    <span className={styles.recommendationTitle}>{rec.title}</span>
                    <span
                      className={styles.difficultyBadge}
                      style={{ backgroundColor: getDifficultyColor(rec.difficulty_match) }}
                      title={`${Math.round(rec.difficulty_match * 100)}% match`}
                    >
                      {getDifficultyLabel(rec.difficulty_match)}
                    </span>
                  </div>
                  <span className={styles.recommendationReason}>{rec.reason}</span>
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Progress tracker */}
      {showProgress && (
        <div className={styles.section}>
          <ProgressTracker compact={false} showModules={false} />
        </div>
      )}

      {/* Quick actions */}
      <div className={styles.quickActions}>
        <a href="/docs/front-matter/introduction" className={styles.quickAction}>
          Start from beginning
        </a>
        <a href="/docs/back-matter/glossary" className={styles.quickAction}>
          View Glossary
        </a>
      </div>
    </aside>
  );
}

export { RecommendationSidebar };
