"""Personalization service for chapter recommendations and difficulty adjustment."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.preference import UserPreference

# Chapter dependencies and prerequisites
CHAPTER_PREREQUISITES = {
    1: [],  # Ch01: Welcome - no prerequisites
    2: [1],  # Ch02: Sensors - requires Ch01
    3: [1, 2],  # Ch03: ROS2 Architecture - requires Ch01, Ch02
    4: [3],  # Ch04: URDF - requires Ch03
    5: [1, 2, 3, 4],  # Ch05: Edge Capstone - requires all Module 1
    6: [3],  # Ch06: Gazebo - requires ROS2 basics
    7: [5, 6],  # Ch07: Unity Capstone - requires Ch05, Ch06
    8: [6],  # Ch08: Isaac Sim - requires Gazebo basics
    9: [8],  # Ch09: Isaac ROS GPU - requires Isaac Sim
    10: [8, 9],  # Ch10: Nav RL Sim2Real - requires Isaac chapters
    11: [5],  # Ch11: Humanoid Locomotion - requires edge basics
    12: [11],  # Ch12: Dexterous Manipulation - requires locomotion
    13: [11, 12],  # Ch13: VLA - requires locomotion and manipulation
    14: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # Ch14: Final Capstone
}

# Chapter difficulty levels
CHAPTER_DIFFICULTY = {
    1: "beginner",
    2: "beginner",
    3: "intermediate",
    4: "intermediate",
    5: "advanced",  # Capstone
    6: "intermediate",
    7: "advanced",  # Capstone
    8: "intermediate",
    9: "advanced",
    10: "advanced",
    11: "advanced",
    12: "advanced",
    13: "advanced",
    14: "advanced",  # Final Capstone
}

# Module groupings
MODULES = {
    1: [1, 2, 3, 4, 5],  # ROS2 Fundamentals
    2: [6, 7],  # Digital Twin
    3: [8, 9, 10],  # NVIDIA Isaac
    4: [11, 12, 13, 14],  # VLA & Capstone
}


class PersonalizationService:
    """Service for personalized learning recommendations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_preferences(self, user_id: UUID) -> UserPreference | None:
        """Get user preferences by user ID."""
        result = await self.db.execute(
            select(UserPreference).where(UserPreference.user_id == user_id)
        )
        return result.scalar_one_or_none()

    def get_completed_chapters(self, preferences: UserPreference | None) -> list[int]:
        """Extract list of completed chapter numbers."""
        if not preferences or not preferences.completed_chapters:
            return []
        return [
            int(ch) for ch, completed in preferences.completed_chapters.items()
            if completed
        ]

    def get_next_recommended_chapters(
        self,
        preferences: UserPreference | None,
        limit: int = 3
    ) -> list[dict]:
        """Get recommended next chapters based on progress."""
        completed = set(self.get_completed_chapters(preferences))
        experience_level = preferences.experience_level if preferences else "beginner"

        recommendations = []

        for chapter in range(1, 15):
            if chapter in completed:
                continue

            # Check if prerequisites are met
            prereqs = CHAPTER_PREREQUISITES.get(chapter, [])
            prereqs_met = all(p in completed for p in prereqs)

            if not prereqs_met:
                continue

            # Get chapter difficulty
            difficulty = CHAPTER_DIFFICULTY.get(chapter, "intermediate")

            # Calculate priority score
            priority = self._calculate_priority(
                chapter, difficulty, experience_level, completed
            )

            recommendations.append({
                "chapter": chapter,
                "difficulty": difficulty,
                "prerequisites_met": True,
                "priority": priority,
                "module": self._get_module_for_chapter(chapter),
            })

        # Sort by priority and return top recommendations
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        return recommendations[:limit]

    def _calculate_priority(
        self,
        chapter: int,
        difficulty: str,
        experience_level: str,
        completed: set[int]
    ) -> float:
        """Calculate recommendation priority score."""
        priority = 10.0

        # Prefer chapters that match user's experience level
        difficulty_match = {
            "beginner": {"beginner": 1.5, "intermediate": 1.0, "advanced": 0.5},
            "intermediate": {"beginner": 0.8, "intermediate": 1.5, "advanced": 1.0},
            "advanced": {"beginner": 0.5, "intermediate": 1.0, "advanced": 1.5},
        }
        level_multiplier = difficulty_match.get(
            experience_level, {}
        ).get(difficulty, 1.0)
        priority *= level_multiplier

        # Prefer sequential progression within modules
        module = self._get_module_for_chapter(chapter)
        if module:
            module_chapters = MODULES[module]
            chapter_index = module_chapters.index(chapter)
            if chapter_index > 0:
                prev_chapter = module_chapters[chapter_index - 1]
                if prev_chapter in completed:
                    priority *= 1.3  # Boost for continuing a module

        # Lower priority for capstone chapters (5, 7, 14)
        if chapter in [5, 7, 14]:
            priority *= 0.8  # Capstones should come after regular chapters

        return round(priority, 2)

    def _get_module_for_chapter(self, chapter: int) -> int | None:
        """Get the module number for a chapter."""
        for module, chapters in MODULES.items():
            if chapter in chapters:
                return module
        return None

    def get_difficulty_adjustment(
        self,
        chapter: int,
        preferences: UserPreference | None
    ) -> dict:
        """Get content difficulty adjustment for a chapter."""
        experience_level = preferences.experience_level if preferences else "beginner"
        chapter_difficulty = CHAPTER_DIFFICULTY.get(chapter, "intermediate")

        # Determine what content to show/hide
        show_advanced = experience_level in ["intermediate", "advanced"]
        show_beginner_tips = experience_level == "beginner"
        expand_code_examples = experience_level != "advanced"

        adjustment = {
            "chapter": chapter,
            "user_level": experience_level,
            "chapter_difficulty": chapter_difficulty,
            "show_advanced_content": show_advanced,
            "show_beginner_tips": show_beginner_tips,
            "expand_code_examples": expand_code_examples,
            "suggested_pace": self._get_suggested_pace(experience_level, chapter_difficulty),
        }

        return adjustment

    def _get_suggested_pace(self, user_level: str, chapter_difficulty: str) -> str:
        """Suggest reading pace based on user level and chapter difficulty."""
        if user_level == "beginner":
            if chapter_difficulty == "beginner":
                return "normal"
            elif chapter_difficulty == "intermediate":
                return "slow"
            else:
                return "very_slow"
        elif user_level == "intermediate":
            if chapter_difficulty == "beginner":
                return "fast"
            elif chapter_difficulty == "intermediate":
                return "normal"
            else:
                return "slow"
        else:  # advanced
            if chapter_difficulty == "advanced":
                return "normal"
            else:
                return "fast"

    def get_learning_path(
        self,
        preferences: UserPreference | None,
        target_chapter: int | None = None
    ) -> list[dict]:
        """Get optimal learning path to reach a target chapter or complete the book."""
        completed = set(self.get_completed_chapters(preferences))
        target = target_chapter or 14  # Default to completing the book

        path = []
        to_complete = set()

        # Find all chapters needed to reach target
        def add_prerequisites(ch: int):
            if ch in completed or ch in to_complete:
                return
            prereqs = CHAPTER_PREREQUISITES.get(ch, [])
            for prereq in prereqs:
                add_prerequisites(prereq)
            to_complete.add(ch)

        add_prerequisites(target)

        # Sort by chapter number for sequential path
        ordered = sorted(to_complete)

        for chapter in ordered:
            path.append({
                "chapter": chapter,
                "difficulty": CHAPTER_DIFFICULTY.get(chapter, "intermediate"),
                "module": self._get_module_for_chapter(chapter),
                "is_capstone": chapter in [5, 7, 14],
            })

        return path

    async def update_chapter_completion(
        self,
        user_id: UUID,
        chapter: int,
        completed: bool = True
    ) -> UserPreference | None:
        """Update chapter completion status."""
        preferences = await self.get_user_preferences(user_id)
        if not preferences:
            return None

        if preferences.completed_chapters is None:
            preferences.completed_chapters = {}

        preferences.completed_chapters[str(chapter)] = completed
        await self.db.commit()
        await self.db.refresh(preferences)

        return preferences

    def get_progress_summary(self, preferences: UserPreference | None) -> dict:
        """Get overall progress summary."""
        completed = self.get_completed_chapters(preferences)
        total_chapters = 14

        # Calculate module progress
        module_progress = {}
        for module, chapters in MODULES.items():
            completed_in_module = len([c for c in chapters if c in completed])
            module_progress[module] = {
                "completed": completed_in_module,
                "total": len(chapters),
                "percentage": round(completed_in_module / len(chapters) * 100, 1),
            }

        return {
            "total_completed": len(completed),
            "total_chapters": total_chapters,
            "overall_percentage": round(len(completed) / total_chapters * 100, 1),
            "module_progress": module_progress,
            "next_recommendations": self.get_next_recommended_chapters(preferences),
        }
