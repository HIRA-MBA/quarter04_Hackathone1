/**
 * Personalized chapter renderer that adapts content based on user profile and progress.
 */

import React, { useEffect, useState, createContext, useContext } from 'react';
import { authApi } from '../../services/auth';
import { userApi, UserPreferences, ProgressSummary, Recommendation, DifficultyAdjustment } from '../../services/user';
import styles from './ChapterRenderer.module.css';

/**
 * Context for difficulty adjustment - allows child components to access content filtering.
 */
interface DifficultyContextValue {
  adjustment: DifficultyAdjustment | null;
  showAdvanced: boolean;
  showBeginnerTips: boolean;
  expandCodeExamples: boolean;
  suggestedPace: string;
}

const DifficultyContext = createContext<DifficultyContextValue>({
  adjustment: null,
  showAdvanced: true,
  showBeginnerTips: false,
  expandCodeExamples: true,
  suggestedPace: 'normal',
});

export const useDifficultyAdjustment = () => useContext(DifficultyContext);

interface ChapterRendererProps {
  chapterId: string;
  children: React.ReactNode;
}

interface ChapterContext {
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  prerequisites: string[];
  estimatedTime: string;
}

// Chapter metadata mapping
const CHAPTER_CONTEXT: Record<string, ChapterContext> = {
  'ch01-welcome-first-node': {
    difficulty: 'beginner',
    prerequisites: [],
    estimatedTime: '45 min',
  },
  'ch02-sensors-perception': {
    difficulty: 'beginner',
    prerequisites: ['ch01'],
    estimatedTime: '60 min',
  },
  'ch03-ros2-architecture': {
    difficulty: 'intermediate',
    prerequisites: ['ch01', 'ch02'],
    estimatedTime: '90 min',
  },
  'ch04-urdf-humanoid': {
    difficulty: 'intermediate',
    prerequisites: ['ch01', 'ch02', 'ch03'],
    estimatedTime: '75 min',
  },
  'ch05-edge-capstone': {
    difficulty: 'advanced',
    prerequisites: ['ch01', 'ch02', 'ch03', 'ch04'],
    estimatedTime: '120 min',
  },
  'ch06-gazebo-physics': {
    difficulty: 'intermediate',
    prerequisites: ['ch01', 'ch04'],
    estimatedTime: '60 min',
  },
  'ch07-unity-capstone': {
    difficulty: 'advanced',
    prerequisites: ['ch06'],
    estimatedTime: '120 min',
  },
  'ch08-isaac-sim': {
    difficulty: 'intermediate',
    prerequisites: ['ch06'],
    estimatedTime: '75 min',
  },
  'ch09-isaac-ros-gpu': {
    difficulty: 'advanced',
    prerequisites: ['ch08'],
    estimatedTime: '90 min',
  },
  'ch10-nav-rl-sim2real': {
    difficulty: 'advanced',
    prerequisites: ['ch08', 'ch09'],
    estimatedTime: '120 min',
  },
  'ch11-humanoid-locomotion': {
    difficulty: 'advanced',
    prerequisites: ['ch04', 'ch10'],
    estimatedTime: '90 min',
  },
  'ch12-dexterous-manipulation': {
    difficulty: 'advanced',
    prerequisites: ['ch11'],
    estimatedTime: '90 min',
  },
  'ch13-vision-language-action': {
    difficulty: 'advanced',
    prerequisites: ['ch11', 'ch12'],
    estimatedTime: '105 min',
  },
  'ch14-capstone-humanoid': {
    difficulty: 'advanced',
    prerequisites: ['ch11', 'ch12', 'ch13'],
    estimatedTime: '180 min',
  },
};

export default function ChapterRenderer({
  chapterId,
  children,
}: ChapterRendererProps): React.ReactElement {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);
  const [progress, setProgress] = useState<ProgressSummary | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [difficultyAdjustment, setDifficultyAdjustment] = useState<DifficultyAdjustment | null>(null);
  // eslint-disable-next-line react-hooks/purity
  const [startTime, setStartTime] = useState<number>(() => Date.now());
  const [chapterProgress, setChapterProgress] = useState(0);

  const context = CHAPTER_CONTEXT[chapterId] || {
    difficulty: 'intermediate',
    prerequisites: [],
    estimatedTime: '60 min',
  };

  // Extract chapter number from chapterId (e.g., "ch01-welcome-first-node" -> 1)
  const getChapterNumber = (id: string): number => {
    const match = id.match(/ch(\d+)/);
    return match ? parseInt(match[1], 10) : 1;
  };

  // Load user data
  useEffect(() => {
    const loadUserData = async () => {
      const user = await authApi.getCurrentUser();
      if (user) {
        setIsAuthenticated(true);
        try {
          const chapterNum = getChapterNumber(chapterId);
          const [prefs, prog, recs, diffAdj] = await Promise.all([
            userApi.getPreferences(),
            userApi.getProgressSummary(),
            userApi.getRecommendations(),
            userApi.getDifficultyAdjustment(chapterNum).catch(() => null),
          ]);
          setPreferences(prefs);
          setProgress(prog);
          setRecommendations(recs);
          setDifficultyAdjustment(diffAdj);

          // Get current chapter progress
          const chapterData = prefs.completed_chapters[chapterId];
          if (chapterData) {
            setChapterProgress(chapterData.progress_percentage);
          }
        } catch (error) {
          console.error('Failed to load user data:', error);
        }
      }
    };

    loadUserData();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStartTime(Date.now());
  }, [chapterId]);

  // Track reading progress via scroll
  useEffect(() => {
    const handleScroll = () => {
      const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrolled = window.scrollY;
      const progress = Math.min(Math.round((scrolled / scrollHeight) * 100), 100);
      setChapterProgress(progress);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Save progress on unmount or chapter change
  useEffect(() => {
    return () => {
      if (isAuthenticated) {
        const timeSpent = Math.round((Date.now() - startTime) / 1000);
        userApi
          .updateProgress(chapterId, chapterProgress >= 90, chapterProgress, timeSpent)
          .catch(console.error);
      }
    };
  }, [chapterId, isAuthenticated, chapterProgress, startTime]);

  const getDifficultyBadge = () => {
    const colors = {
      beginner: { bg: '#e8f5e9', color: '#2e7d32' },
      intermediate: { bg: '#fff3e0', color: '#e65100' },
      advanced: { bg: '#fce4ec', color: '#c2185b' },
    };
    const style = colors[context.difficulty];
    return (
      <span
        className={styles.difficultyBadge}
        style={{ backgroundColor: style.bg, color: style.color }}
      >
        {context.difficulty.charAt(0).toUpperCase() + context.difficulty.slice(1)}
      </span>
    );
  };

  const getExperienceMatch = (): 'good' | 'stretch' | 'review' | null => {
    if (!preferences?.experience_level) return null;

    const levelMap = { beginner: 0, intermediate: 1, advanced: 2 };
    const userLevel = levelMap[preferences.experience_level as keyof typeof levelMap] ?? 1;
    const chapterLevel = levelMap[context.difficulty];

    if (chapterLevel === userLevel) return 'good';
    if (chapterLevel > userLevel) return 'stretch';
    return 'review';
  };

  const match = getExperienceMatch();

  // Prepare difficulty context value
  const difficultyContextValue: DifficultyContextValue = {
    adjustment: difficultyAdjustment,
    showAdvanced: difficultyAdjustment?.show_advanced_content ?? true,
    showBeginnerTips: difficultyAdjustment?.show_beginner_tips ?? false,
    expandCodeExamples: difficultyAdjustment?.expand_code_examples ?? true,
    suggestedPace: difficultyAdjustment?.suggested_pace ?? 'normal',
  };

  // Get pace indicator text and icon
  const getPaceIndicator = () => {
    if (!difficultyAdjustment) return null;
    const paceMap: Record<string, { text: string; icon: string }> = {
      very_slow: { text: 'Take it slow', icon: '🐢' },
      slow: { text: 'Careful pace', icon: '🚶' },
      normal: { text: 'Normal pace', icon: '🏃' },
      fast: { text: 'Quick review', icon: '⚡' },
    };
    return paceMap[difficultyAdjustment.suggested_pace] || null;
  };

  const paceIndicator = getPaceIndicator();

  return (
    <DifficultyContext.Provider value={difficultyContextValue}>
      <div className={styles.chapterWrapper}>
        {/* Chapter metadata bar */}
        <div className={styles.metadataBar}>
          <div className={styles.metadataLeft}>
            {getDifficultyBadge()}
            <span className={styles.estimatedTime}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <circle cx="12" cy="12" r="10" strokeWidth="2" />
                <polyline points="12 6 12 12 16 14" strokeWidth="2" />
              </svg>
              {context.estimatedTime}
            </span>
            {paceIndicator && (
              <span className={styles.paceIndicator} title={`Suggested: ${paceIndicator.text}`}>
                {paceIndicator.icon} {paceIndicator.text}
              </span>
            )}
          </div>

          {isAuthenticated && (
            <div className={styles.metadataRight}>
              {/* Progress indicator */}
              <div className={styles.progressContainer}>
                <div
                  className={styles.progressBar}
                  style={{ width: `${chapterProgress}%` }}
                />
                <span className={styles.progressText}>{chapterProgress}%</span>
              </div>
            </div>
          )}
        </div>

        {/* Experience match indicator */}
        {match && (
          <div className={`${styles.matchBanner} ${styles[match]}`}>
            {match === 'good' && (
              <>
                <strong>Good match!</strong> This chapter aligns with your experience level.
              </>
            )}
            {match === 'stretch' && (
              <>
                <strong>Challenge ahead!</strong> This is above your indicated level.
                Take your time and use the chatbot for help.
              </>
            )}
            {match === 'review' && (
              <>
                <strong>Review material</strong> You might already know this.
                Great for reinforcing fundamentals!
              </>
            )}
          </div>
        )}

        {/* Beginner tips banner (only shown based on difficulty adjustment) */}
        {difficultyContextValue.showBeginnerTips && (
          <div className={styles.beginnerTipsBanner}>
            <strong>Tip:</strong> This chapter includes expanded explanations and examples.
            Don't hesitate to use the chatbot if you need help!
          </div>
        )}

        {/* Main chapter content */}
        <div className={styles.chapterContent}>{children}</div>

        {/* Sidebar with recommendations (for authenticated users) */}
        {isAuthenticated && recommendations.length > 0 && (
          <aside className={styles.sidebar}>
            <h4>Recommended Next</h4>
            <ul className={styles.recommendationList}>
              {recommendations.slice(0, 3).map((rec) => (
                <li key={rec.chapter_id}>
                  <a href={`/docs/${rec.chapter_id}`}>
                    <strong>{rec.title}</strong>
                    <span>{rec.reason}</span>
                  </a>
                </li>
              ))}
            </ul>

            {progress && (
              <div className={styles.overallProgress}>
                <h4>Your Progress</h4>
                <div className={styles.progressCircle}>
                  <svg viewBox="0 0 36 36">
                    <path
                      className={styles.progressCircleBg}
                      d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path
                      className={styles.progressCircleFill}
                      strokeDasharray={`${progress.completion_percentage}, 100`}
                      d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                  </svg>
                  <span>{Math.round(progress.completion_percentage)}%</span>
                </div>
                <p>
                  {progress.completed_chapters} of {progress.total_chapters} chapters
                </p>
              </div>
            )}
          </aside>
        )}
      </div>
    </DifficultyContext.Provider>
  );
}

/**
 * Export individual components for flexibility
 */
export { ChapterRenderer };
