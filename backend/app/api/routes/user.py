"""User API routes for profile, preferences, and progress tracking."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.postgres import get_db_session
from app.models.preference import UserPreference
from app.models.user import User
from app.services.personalization import PersonalizationService

router = APIRouter(prefix="/user", tags=["user"])


# Request/Response schemas
class ProgrammingLanguagesSchema(BaseModel):
    """Programming language proficiency levels."""

    python: str = Field(default="none", pattern="^(none|beginner|intermediate|advanced)$")
    cpp: str = Field(default="none", pattern="^(none|beginner|intermediate|advanced)$")
    javascript: str = Field(default="none", pattern="^(none|beginner|intermediate|advanced)$")


class QuestionnaireRequest(BaseModel):
    """Background questionnaire request schema."""

    experience_level: str = Field(
        ...,
        description="User's experience level",
        examples=["beginner", "intermediate", "advanced"],
    )
    background: str = Field(
        ...,
        max_length=2000,
        description="User's background (education, work experience)",
    )
    goals: str = Field(
        ...,
        max_length=2000,
        description="Learning goals and objectives",
    )
    # Programming language proficiency
    programming_languages: ProgrammingLanguagesSchema | None = Field(
        None,
        description="Programming language proficiency levels",
    )
    # Additional questionnaire fields
    programming_experience: str | None = Field(
        None,
        description="Programming languages familiar with (deprecated, use programming_languages)",
    )
    robotics_experience: str | None = Field(
        None,
        description="Prior robotics experience",
    )
    preferred_learning_style: str | None = Field(
        None,
        description="Visual, hands-on, reading, etc.",
    )


class PreferencesRequest(BaseModel):
    """User preferences update request."""

    language: str | None = Field(None, pattern="^(en|ur)$")
    theme: str | None = Field(None, pattern="^(light|dark|system)$")
    font_size: str | None = Field(None, pattern="^(small|medium|large)$")


class ChapterProgressRequest(BaseModel):
    """Chapter progress update request."""

    chapter_id: str
    completed: bool
    progress_percentage: int = Field(ge=0, le=100)
    time_spent_seconds: int = Field(ge=0)


class BookmarkRequest(BaseModel):
    """Bookmark request schema."""

    chapter_id: str
    section_id: str | None = None
    note: str | None = Field(None, max_length=500)


class PreferencesResponse(BaseModel):
    """User preferences response."""

    language: str
    experience_level: str | None
    background: str | None
    goals: str | None
    programming_languages: dict | None
    completed_chapters: dict
    bookmarks: dict
    theme: str
    font_size: str

    class Config:
        from_attributes = True


class ProgressSummaryResponse(BaseModel):
    """Progress summary response."""

    total_chapters: int
    completed_chapters: int
    completion_percentage: float
    current_module: str | None
    recommended_next: str | None


class RecommendationResponse(BaseModel):
    """Personalized recommendation response."""

    chapter_id: str
    title: str
    reason: str
    difficulty_match: float


class DifficultyAdjustmentResponse(BaseModel):
    """Difficulty adjustment response for personalized content."""

    chapter: int
    user_level: str
    chapter_difficulty: str
    show_advanced_content: bool
    show_beginner_tips: bool
    expand_code_examples: bool
    suggested_pace: str


# Routes
@router.post("/questionnaire", response_model=PreferencesResponse)
async def submit_questionnaire(
    body: QuestionnaireRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> PreferencesResponse:
    """Submit background questionnaire for personalization."""
    # Get or create user preferences
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    if not preference:
        preference = UserPreference(
            user_id=current_user.id,
            language="en",
            completed_chapters={},
            bookmarks={},
        )
        db.add(preference)

    # Update questionnaire fields
    preference.experience_level = body.experience_level
    preference.background = body.background
    preference.goals = body.goals

    # Store programming languages proficiency
    if body.programming_languages:
        preference.programming_languages = body.programming_languages.model_dump()

    # Store additional responses in background field
    if body.robotics_experience or body.preferred_learning_style:
        additional_info = []
        if body.robotics_experience:
            additional_info.append(f"Robotics experience: {body.robotics_experience}")
        if body.preferred_learning_style:
            additional_info.append(f"Learning style: {body.preferred_learning_style}")
        if additional_info and preference.background:
            preference.background = f"{body.background}\n\n{'; '.join(additional_info)}"

    await db.commit()
    await db.refresh(preference)

    return PreferencesResponse(
        language=preference.language,
        experience_level=preference.experience_level,
        background=preference.background,
        goals=preference.goals,
        programming_languages=preference.programming_languages,
        completed_chapters=preference.completed_chapters or {},
        bookmarks=preference.bookmarks or {},
        theme=preference.theme,
        font_size=preference.font_size,
    )


@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> PreferencesResponse:
    """Get user preferences."""
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    if not preference:
        # Return defaults
        return PreferencesResponse(
            language="en",
            experience_level=None,
            background=None,
            goals=None,
            programming_languages=None,
            completed_chapters={},
            bookmarks={},
            theme="system",
            font_size="medium",
        )

    return PreferencesResponse(
        language=preference.language,
        experience_level=preference.experience_level,
        background=preference.background,
        goals=preference.goals,
        programming_languages=preference.programming_languages,
        completed_chapters=preference.completed_chapters or {},
        bookmarks=preference.bookmarks or {},
        theme=preference.theme,
        font_size=preference.font_size,
    )


@router.patch("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    body: PreferencesRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> PreferencesResponse:
    """Update user preferences."""
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    if not preference:
        preference = UserPreference(
            user_id=current_user.id,
            language="en",
            completed_chapters={},
            bookmarks={},
        )
        db.add(preference)

    # Update only provided fields
    if body.language is not None:
        preference.language = body.language
    if body.theme is not None:
        preference.theme = body.theme
    if body.font_size is not None:
        preference.font_size = body.font_size

    await db.commit()
    await db.refresh(preference)

    return PreferencesResponse(
        language=preference.language,
        experience_level=preference.experience_level,
        background=preference.background,
        goals=preference.goals,
        programming_languages=preference.programming_languages,
        completed_chapters=preference.completed_chapters or {},
        bookmarks=preference.bookmarks or {},
        theme=preference.theme,
        font_size=preference.font_size,
    )


@router.post("/progress", response_model=dict)
async def update_chapter_progress(
    body: ChapterProgressRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """Update progress for a specific chapter."""
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    if not preference:
        preference = UserPreference(
            user_id=current_user.id,
            language="en",
            completed_chapters={},
            bookmarks={},
        )
        db.add(preference)

    # Update chapter progress
    completed_chapters = dict(preference.completed_chapters or {})
    completed_chapters[body.chapter_id] = {
        "completed": body.completed,
        "progress_percentage": body.progress_percentage,
        "time_spent_seconds": body.time_spent_seconds,
    }
    preference.completed_chapters = completed_chapters

    await db.commit()

    return {"message": "Progress updated", "chapter_id": body.chapter_id}


@router.get("/progress", response_model=ProgressSummaryResponse)
async def get_progress_summary(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> ProgressSummaryResponse:
    """Get overall progress summary."""
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    # Total chapters in the book
    total_chapters = 14

    if not preference or not preference.completed_chapters:
        return ProgressSummaryResponse(
            total_chapters=total_chapters,
            completed_chapters=0,
            completion_percentage=0.0,
            current_module="module-1",
            recommended_next="ch01-welcome-first-node",
        )

    completed = preference.completed_chapters
    completed_count = sum(1 for ch in completed.values() if ch.get("completed", False))

    # Determine current module based on progress
    chapter_modules = {
        "ch01": "module-1",
        "ch02": "module-1",
        "ch03": "module-1",
        "ch04": "module-1",
        "ch05": "module-1",
        "ch06": "module-2",
        "ch07": "module-2",
        "ch08": "module-3",
        "ch09": "module-3",
        "ch10": "module-3",
        "ch11": "module-4",
        "ch12": "module-4",
        "ch13": "module-4",
        "ch14": "module-4",
    }

    # Find first incomplete chapter
    chapter_order = [f"ch{i:02d}" for i in range(1, 15)]
    recommended_next = None
    current_module = "module-1"

    for ch_prefix in chapter_order:
        ch_key = next(
            (k for k in completed if k.startswith(ch_prefix)),
            None,
        )
        if not ch_key or not completed.get(ch_key, {}).get("completed", False):
            recommended_next = ch_prefix
            current_module = chapter_modules.get(ch_prefix, "module-1")
            break

    return ProgressSummaryResponse(
        total_chapters=total_chapters,
        completed_chapters=completed_count,
        completion_percentage=round(completed_count / total_chapters * 100, 1),
        current_module=current_module,
        recommended_next=recommended_next,
    )


@router.post("/bookmarks", response_model=dict)
async def add_bookmark(
    body: BookmarkRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """Add a bookmark to a chapter."""
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    if not preference:
        preference = UserPreference(
            user_id=current_user.id,
            language="en",
            completed_chapters={},
            bookmarks={},
        )
        db.add(preference)

    # Add bookmark
    bookmarks = dict(preference.bookmarks or {})
    bookmark_key = f"{body.chapter_id}:{body.section_id or 'main'}"
    bookmarks[bookmark_key] = {
        "chapter_id": body.chapter_id,
        "section_id": body.section_id,
        "note": body.note,
    }
    preference.bookmarks = bookmarks

    await db.commit()

    return {"message": "Bookmark added", "bookmark_key": bookmark_key}


@router.delete("/bookmarks/{bookmark_key}", response_model=dict)
async def remove_bookmark(
    bookmark_key: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """Remove a bookmark."""
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    if not preference or not preference.bookmarks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    bookmarks = dict(preference.bookmarks)
    if bookmark_key not in bookmarks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    del bookmarks[bookmark_key]
    preference.bookmarks = bookmarks

    await db.commit()

    return {"message": "Bookmark removed"}


@router.get("/recommendations", response_model=list[RecommendationResponse])
async def get_recommendations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[RecommendationResponse]:
    """Get personalized chapter recommendations based on profile and progress."""
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    # Default recommendations for new users
    recommendations = []

    # Chapter info for recommendations
    chapters = [
        ("ch01-welcome-first-node", "Welcome to ROS 2: Your First Robot Node", "beginner"),
        ("ch02-sensors-perception", "Sensors and Perception", "beginner"),
        ("ch03-ros2-architecture", "ROS 2 Architecture Deep Dive", "intermediate"),
        ("ch04-urdf-humanoid", "URDF and Humanoid Modeling", "intermediate"),
        ("ch05-edge-capstone", "Edge Computing Capstone", "advanced"),
        ("ch06-gazebo-physics", "Gazebo and Physics Simulation", "intermediate"),
        ("ch07-unity-capstone", "Unity Digital Twin Capstone", "advanced"),
        ("ch08-isaac-sim", "NVIDIA Isaac Sim Introduction", "intermediate"),
        ("ch09-isaac-ros-gpu", "Isaac ROS GPU Acceleration", "advanced"),
        ("ch10-nav-rl-sim2real", "Navigation, RL, and Sim2Real", "advanced"),
        ("ch11-humanoid-locomotion", "Humanoid Locomotion", "advanced"),
        ("ch12-dexterous-manipulation", "Dexterous Manipulation", "advanced"),
        ("ch13-vision-language-action", "Vision-Language-Action Models", "advanced"),
        ("ch14-capstone-humanoid", "Final Humanoid Capstone", "advanced"),
    ]

    user_level = "beginner"
    if preference and preference.experience_level:
        user_level = preference.experience_level

    completed = preference.completed_chapters if preference else {}

    # Find next chapters based on experience level
    level_priority = {"beginner": 0, "intermediate": 1, "advanced": 2}
    user_priority = level_priority.get(user_level, 0)

    for chapter_id, title, difficulty in chapters:
        # Skip completed chapters
        if any(chapter_id in k for k in completed):
            ch_data = next(
                (v for k, v in completed.items() if chapter_id in k),
                {},
            )
            if ch_data.get("completed", False):
                continue

        diff_priority = level_priority.get(difficulty, 0)
        difficulty_match = 1.0 - abs(diff_priority - user_priority) * 0.3

        # Generate reason based on context
        reason = "Continue your learning journey"
        if diff_priority == user_priority:
            reason = f"Perfect match for your {user_level} level"
        elif diff_priority < user_priority:
            reason = "Quick review to solidify fundamentals"
        else:
            reason = "Challenge yourself with advanced material"

        recommendations.append(
            RecommendationResponse(
                chapter_id=chapter_id,
                title=title,
                reason=reason,
                difficulty_match=round(difficulty_match, 2),
            )
        )

        if len(recommendations) >= 5:
            break

    return recommendations


@router.get("/difficulty/{chapter_num}", response_model=DifficultyAdjustmentResponse)
async def get_difficulty_adjustment(
    chapter_num: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> DifficultyAdjustmentResponse:
    """Get difficulty adjustment settings for a specific chapter."""
    if chapter_num < 1 or chapter_num > 14:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chapter number must be between 1 and 14",
        )

    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    service = PersonalizationService(db)
    adjustment = service.get_difficulty_adjustment(chapter_num, preference)

    return DifficultyAdjustmentResponse(**adjustment)


class PersonalizeChapterRequest(BaseModel):
    """Request to personalize chapter introduction."""

    chapter_id: str = Field(..., description="Chapter identifier")
    chapter_title: str = Field(..., description="Chapter title")


class PersonalizeChapterResponse(BaseModel):
    """Personalized chapter introduction response."""

    chapter_id: str
    personalized_intro: str
    user_level: str


@router.post("/personalize-chapter", response_model=PersonalizeChapterResponse)
async def personalize_chapter(
    body: PersonalizeChapterRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> PersonalizeChapterResponse:
    """
    Generate a personalized introduction for a chapter based on user's background.

    Uses AI to create a custom intro that relates chapter content to the user's
    experience level and known programming languages.
    """
    from openai import AsyncOpenAI

    from app.config import get_settings

    settings = get_settings()

    # Get user preferences
    result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    preference = result.scalar_one_or_none()

    if not preference:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please complete the questionnaire first",
        )

    # Build context from user preferences
    user_level = preference.experience_level or "beginner"
    programming_langs = preference.programming_languages or {}
    background = preference.background or ""

    known_langs = []
    for lang, level in programming_langs.items():
        if level and level != "none":
            lang_name = {"python": "Python", "cpp": "C++", "javascript": "JavaScript"}.get(
                lang, lang
            )
            known_langs.append(f"{lang_name} ({level})")

    # Create prompt for personalization
    prompt = f"""Create a brief, personalized introduction (2-3 paragraphs) for a robotics textbook chapter.

Chapter: {body.chapter_title}

User Profile:
- Experience Level: {user_level}
- Programming Background: {", ".join(known_langs) if known_langs else "Not specified"}
- Background: {background[:200] if background else "Not specified"}

Guidelines:
1. Address the user's current skill level
2. Connect the chapter content to their known programming languages if relevant
3. Set clear expectations for what they'll learn
4. Be encouraging and practical
5. Keep it concise (max 150 words)

Write the introduction directly without any preamble."""

    # Call OpenAI to generate personalized intro
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert robotics educator creating personalized learning experiences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        personalized_intro = response.choices[0].message.content or ""

        return PersonalizeChapterResponse(
            chapter_id=body.chapter_id,
            personalized_intro=personalized_intro.strip(),
            user_level=user_level,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate personalized intro: {str(e)}",
        ) from e
