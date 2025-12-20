/**
 * ProgressTracker component - standalone progress tracking UI.
 * Displays overall book progress and module-level progress.
 */

import React, { useEffect, useState } from 'react';
import { authApi } from '../../services/auth';
import { userApi, ProgressSummary, UserPreferences } from '../../services/user';
import styles from './ProgressTracker.module.css';

interface ProgressTrackerProps {
  compact?: boolean;
  showModules?: boolean;
}

interface ModuleProgress {
  name: string;
  chapters: number[];
  completed: number;
  total: number;
}

const MODULES: ModuleProgress[] = [
  { name: 'ROS 2 Fundamentals', chapters: [1, 2, 3, 4, 5], completed: 0, total: 5 },
  { name: 'Digital Twin', chapters: [6, 7], completed: 0, total: 2 },
  { name: 'NVIDIA Isaac', chapters: [8, 9, 10], completed: 0, total: 3 },
  { name: 'VLA & Capstone', chapters: [11, 12, 13, 14], completed: 0, total: 4 },
];

export default function ProgressTracker({
  compact = false,
  showModules = true,
}: ProgressTrackerProps): React.ReactElement | null {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [progress, setProgress] = useState<ProgressSummary | null>(null);
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);
  const [moduleProgress, setModuleProgress] = useState<ModuleProgress[]>(MODULES);

  useEffect(() => {
    const loadProgress = async () => {
      const user = await authApi.getCurrentUser();
      if (user) {
        setIsAuthenticated(true);
        try {
          const [prog, prefs] = await Promise.all([
            userApi.getProgressSummary(),
            userApi.getPreferences(),
          ]);
          setProgress(prog);
          setPreferences(prefs);

          // Calculate module-level progress
          if (prefs?.completed_chapters) {
            const completedChapters = new Set(
              Object.entries(prefs.completed_chapters)
                .filter(([_, data]) => data.completed)
                .map(([key]) => {
                  const match = key.match(/ch(\d+)/);
                  return match ? parseInt(match[1], 10) : 0;
                })
            );

            const updatedModules = MODULES.map((module) => ({
              ...module,
              completed: module.chapters.filter((ch) => completedChapters.has(ch)).length,
            }));
            setModuleProgress(updatedModules);
          }
        } catch (error) {
          console.error('Failed to load progress:', error);
        }
      }
    };

    loadProgress();
  }, []);

  if (!isAuthenticated || !progress) {
    return null;
  }

  if (compact) {
    return (
      <div className={styles.compactProgress}>
        <div className={styles.compactCircle}>
          <svg viewBox="0 0 36 36">
            <path
              className={styles.circleBg}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
            <path
              className={styles.circleFill}
              strokeDasharray={`${progress.completion_percentage}, 100`}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
          </svg>
          <span className={styles.compactPercent}>{Math.round(progress.completion_percentage)}%</span>
        </div>
        <span className={styles.compactLabel}>
          {progress.completed_chapters}/{progress.total_chapters}
        </span>
      </div>
    );
  }

  return (
    <div className={styles.progressTracker}>
      <h3 className={styles.title}>Your Progress</h3>

      {/* Overall progress circle */}
      <div className={styles.overallProgress}>
        <div className={styles.progressCircle}>
          <svg viewBox="0 0 36 36">
            <path
              className={styles.circleBg}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
            <path
              className={styles.circleFill}
              strokeDasharray={`${progress.completion_percentage}, 100`}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
          </svg>
          <span className={styles.percentText}>{Math.round(progress.completion_percentage)}%</span>
        </div>
        <p className={styles.chapterCount}>
          {progress.completed_chapters} of {progress.total_chapters} chapters completed
        </p>
      </div>

      {/* Module-level progress */}
      {showModules && (
        <div className={styles.moduleProgress}>
          <h4 className={styles.moduleTitle}>By Module</h4>
          {moduleProgress.map((module, idx) => (
            <div key={idx} className={styles.moduleItem}>
              <div className={styles.moduleHeader}>
                <span className={styles.moduleName}>{module.name}</span>
                <span className={styles.moduleCount}>
                  {module.completed}/{module.total}
                </span>
              </div>
              <div className={styles.moduleBar}>
                <div
                  className={styles.moduleBarFill}
                  style={{ width: `${(module.completed / module.total) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Current position */}
      {progress.current_module && (
        <div className={styles.currentPosition}>
          <span className={styles.currentLabel}>Currently on:</span>
          <span className={styles.currentModule}>
            {progress.current_module.replace('module-', 'Module ')}
          </span>
        </div>
      )}

      {/* Next recommendation */}
      {progress.recommended_next && (
        <div className={styles.nextUp}>
          <span className={styles.nextLabel}>Next up:</span>
          <a
            href={`/docs/${progress.recommended_next}`}
            className={styles.nextLink}
          >
            {progress.recommended_next.replace(/-/g, ' ')}
          </a>
        </div>
      )}
    </div>
  );
}

export { ProgressTracker };
