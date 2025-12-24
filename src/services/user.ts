/**
 * User API client for profile, preferences, and progress tracking.
 */

import { authApi } from './auth';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export type ProficiencyLevel = 'none' | 'beginner' | 'intermediate' | 'advanced';

export interface ProgrammingLanguages {
  python: ProficiencyLevel;
  cpp: ProficiencyLevel;
  javascript: ProficiencyLevel;
}

export interface QuestionnaireData {
  experience_level: 'beginner' | 'intermediate' | 'advanced';
  background: string;
  goals: string;
  programming_experience?: string;
  robotics_experience?: string;
  preferred_learning_style?: string;
  programming_languages?: ProgrammingLanguages;
}

export interface UserPreferences {
  language: string;
  experience_level: string | null;
  background: string | null;
  goals: string | null;
  programming_languages: ProgrammingLanguages | null;
  completed_chapters: Record<string, ChapterProgress>;
  bookmarks: Record<string, Bookmark>;
  theme: string;
  font_size: string;
}

export interface ChapterProgress {
  completed: boolean;
  progress_percentage: number;
  time_spent_seconds: number;
}

export interface Bookmark {
  chapter_id: string;
  section_id: string | null;
  note: string | null;
}

export interface ProgressSummary {
  total_chapters: number;
  completed_chapters: number;
  completion_percentage: number;
  current_module: string | null;
  recommended_next: string | null;
}

export interface Recommendation {
  chapter_id: string;
  title: string;
  reason: string;
  difficulty_match: number;
}

export interface DifficultyAdjustment {
  chapter: number;
  user_level: string;
  chapter_difficulty: string;
  show_advanced_content: boolean;
  show_beginner_tips: boolean;
  expand_code_examples: boolean;
  suggested_pace: 'very_slow' | 'slow' | 'normal' | 'fast';
}

class UserApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Submit background questionnaire.
   */
  async submitQuestionnaire(data: QuestionnaireData): Promise<UserPreferences> {
    const response = await fetch(`${this.baseUrl}/api/user/questionnaire`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authApi.getAuthHeaders(),
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to submit questionnaire' }));
      throw new Error(error.detail || 'Failed to submit questionnaire');
    }

    return response.json();
  }

  /**
   * Get user preferences.
   */
  async getPreferences(): Promise<UserPreferences> {
    const response = await fetch(`${this.baseUrl}/api/user/preferences`, {
      headers: {
        ...authApi.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to get preferences');
    }

    return response.json();
  }

  /**
   * Update user preferences.
   */
  async updatePreferences(
    preferences: Partial<{ language: string; theme: string; font_size: string }>
  ): Promise<UserPreferences> {
    const response = await fetch(`${this.baseUrl}/api/user/preferences`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        ...authApi.getAuthHeaders(),
      },
      body: JSON.stringify(preferences),
    });

    if (!response.ok) {
      throw new Error('Failed to update preferences');
    }

    return response.json();
  }

  /**
   * Update chapter progress.
   */
  async updateProgress(
    chapterId: string,
    completed: boolean,
    progressPercentage: number,
    timeSpentSeconds: number
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/user/progress`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authApi.getAuthHeaders(),
      },
      body: JSON.stringify({
        chapter_id: chapterId,
        completed,
        progress_percentage: progressPercentage,
        time_spent_seconds: timeSpentSeconds,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to update progress');
    }
  }

  /**
   * Get progress summary.
   */
  async getProgressSummary(): Promise<ProgressSummary> {
    const response = await fetch(`${this.baseUrl}/api/user/progress`, {
      headers: {
        ...authApi.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to get progress');
    }

    return response.json();
  }

  /**
   * Add a bookmark.
   */
  async addBookmark(
    chapterId: string,
    sectionId?: string,
    note?: string
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/user/bookmarks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authApi.getAuthHeaders(),
      },
      body: JSON.stringify({
        chapter_id: chapterId,
        section_id: sectionId,
        note,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to add bookmark');
    }
  }

  /**
   * Remove a bookmark.
   */
  async removeBookmark(bookmarkKey: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/user/bookmarks/${encodeURIComponent(bookmarkKey)}`,
      {
        method: 'DELETE',
        headers: {
          ...authApi.getAuthHeaders(),
        },
      }
    );

    if (!response.ok) {
      throw new Error('Failed to remove bookmark');
    }
  }

  /**
   * Get personalized recommendations.
   */
  async getRecommendations(): Promise<Recommendation[]> {
    const response = await fetch(`${this.baseUrl}/api/user/recommendations`, {
      headers: {
        ...authApi.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to get recommendations');
    }

    return response.json();
  }

  /**
   * Get difficulty adjustment for a chapter.
   */
  async getDifficultyAdjustment(chapterNum: number): Promise<DifficultyAdjustment> {
    const response = await fetch(`${this.baseUrl}/api/user/difficulty/${chapterNum}`, {
      headers: {
        ...authApi.getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to get difficulty adjustment');
    }

    return response.json();
  }

  /**
   * Get personalized chapter introduction.
   */
  async getPersonalizedIntro(
    chapterId: string,
    chapterTitle: string
  ): Promise<{ chapter_id: string; personalized_intro: string; user_level: string }> {
    const response = await fetch(`${this.baseUrl}/api/user/personalize-chapter`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authApi.getAuthHeaders(),
      },
      body: JSON.stringify({
        chapter_id: chapterId,
        chapter_title: chapterTitle,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to personalize' }));
      throw new Error(error.detail || 'Failed to get personalized introduction');
    }

    return response.json();
  }
}

// Export singleton instance
export const userApi = new UserApiClient();

// Export class for testing
export { UserApiClient };
